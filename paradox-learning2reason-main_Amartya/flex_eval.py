# flex_eval.py
import argparse
import pprint
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

from safe_eval import safe_evaluate
from dataset import LogicDataset


# ------------------ encoder helpers ------------------

def _get_encoder(model):
    if hasattr(model, "bert") and hasattr(model.bert, "encoder"):
        return model.bert.encoder, "bert"
    if hasattr(model, "roberta") and hasattr(model.roberta, "encoder"):
        return model.roberta.encoder, "roberta"
    raise RuntimeError("Supported only for BERT/RoBERTa-style encoders.")


def apply_tail_k(model, k: int):
    enc, _ = _get_encoder(model)
    L = len(enc.layer)
    k = max(1, min(k, L))
    enc.layer = torch.nn.ModuleList(enc.layer[-k:])
    model.config.num_hidden_layers = k
    print(f"[Eval] Using tail {k} layers (indices {list(range(L-k, L))})")


def apply_repeat_last(model, n: int):
    enc, _ = _get_encoder(model)
    L = len(enc.layer)
    n = max(1, n)
    last = enc.layer[-1]
    enc.layer = torch.nn.ModuleList([last for _ in range(n)])
    model.config.num_hidden_layers = n
    print(f"[Eval] Repeating last block {n}× (original L={L})")


def _parse_indices(s: str, L: int, one_based: bool):
    raw = [x.strip() for x in s.split(",") if x.strip()]
    idx = []
    for r in raw:
        i = int(r)
        if i < 0:
            i = L + i  # negative indexing from end
        else:
            if one_based:
                i = i - 1
        if not (0 <= i < L):
            raise ValueError(f"Layer index {i} out of range 0..{L-1} (after indexing conversion)")
        idx.append(i)
    return idx


def apply_pick_layers(model, indices):
    enc, _ = _get_encoder(model)
    L = len(enc.layer)
    enc.layer = torch.nn.ModuleList([enc.layer[i] for i in indices])
    model.config.num_hidden_layers = len(indices)
    print(f"[Eval] Using encoder layers (0-based): {indices} of original {L}")


def enable_bridge_layernorm(model, device):
    # Adds a LayerNorm before the first kept block to recondition embeddings.
    if not hasattr(model, "_bridge_ln"):
        model._bridge_ln = torch.nn.LayerNorm(model.config.hidden_size).to(device)
    enc, _ = _get_encoder(model)
    orig_forward = enc.forward

    def bridged_forward(hidden_states, **kw):
        return orig_forward(model._bridge_ln(hidden_states), **kw)

    enc.forward = bridged_forward
    print("[Eval] Enabled LayerNorm bridge before first kept block")


# ------------------ main ------------------

def main():
    p = argparse.ArgumentParser()

    # Core paths
    p.add_argument("--model_name_or_path", required=True, type=str)
    p.add_argument("--tokenizer_name", required=True, type=str)
    p.add_argument("--val_file_path", required=True, type=str)

    # Model/data flags
    p.add_argument("--model_type", default="bert", type=str)     # used by safe_evaluate for token_type_ids
    p.add_argument("--do_lower_case", action="store_true")
    p.add_argument("--cache_dir", default=None, type=str)
    p.add_argument("--eval_batch_size", default=16, type=int)
    p.add_argument("--max_length", default=512, type=int)
    p.add_argument("--device", default="cpu", type=str)

    # Depth reporting window (same semantics as your safe_eval)
    p.add_argument("--group_by_which_depth", default="depth", type=str)
    p.add_argument("--limit_report_depth", default=-1, type=int)
    p.add_argument("--limit_report_max_depth", default=1000, type=int)

    # Dataset passthroughs used by LogicDataset
    p.add_argument("--keep_only_negative", action="store_true")
    p.add_argument("--skip_long_examples", action="store_true")
    p.add_argument("--limit_example_num", default=-1, type=int)
    p.add_argument("--ignore_fact", action="store_true")
    p.add_argument("--ignore_both", action="store_true")
    p.add_argument("--ignore_query", action="store_true")
    p.add_argument("--shorten_input", action="store_true")
    p.add_argument("--shrink_ratio", default=1, type=int)
    p.add_argument("--further_split", action="store_true")
    p.add_argument("--further_further_split", action="store_true")
    p.add_argument("--max_depth_during_train", default=1000, type=int)

    # Layer control (choose exactly one mode)
    sub = p.add_subparsers(dest="mode", required=True)

    tail = sub.add_parser("tail_k", help="Keep the last k consecutive layers")
    tail.add_argument("--k", type=int, required=True)

    rep = sub.add_parser("repeat_last", help="Repeat the last block n times")
    rep.add_argument("--n", type=int, required=True)

    pick = sub.add_parser("pick_layers", help="Keep only the listed layers")
    pick.add_argument("--layers", type=str, required=True,
                      help="Comma-separated indices. Default 1-based, negative ok (e.g., -1 for last)")
    pick.add_argument("--zero_based", action="store_true",
                      help="Interpret --layers as 0-based if set")

    # Optional LN bridge
    p.add_argument("--bridge_ln", action="store_true", help="Enable LayerNorm bridge")

    args = p.parse_args()
    args.device = torch.device(args.device)

    # ---- Tokenizer ----
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, do_lower_case=args.do_lower_case, cache_dir=args.cache_dir
    )
    # be robust to casing in your dataset
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[AND]", "[THEN]", "[and]", "[then]"]}
    )
    if added:
        print(f"Added {added} special tokens to tokenizer.")

    # ---- Dataset (by depth) ----
    print("Loading datasets (by depth)...")
    datasets = LogicDataset.initialize_from_file_by_depth(args.val_file_path, args)
    for d in datasets.values():
        d.tokenizer = tokenizer   # force shared tokenizer
        d.max_length = args.max_length
    depths = sorted(list(datasets.keys()))
    print("Depths found:", depths)

    # ---- Model ----
    print("Loading model...")
    config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, config=config, cache_dir=args.cache_dir
    )

    # Resize embeddings if tokenizer grew
    emb_sz = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > emb_sz:
        print(f"Resizing embeddings: {emb_sz} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    # Layer manipulation
    enc, _ = _get_encoder(model)
    total_layers = len(enc.layer)

    if args.mode == "tail_k":
        apply_tail_k(model, args.k)
    elif args.mode == "repeat_last":
        apply_repeat_last(model, args.n)
    elif args.mode == "pick_layers":
        indices = _parse_indices(args.layers, total_layers, one_based=not args.zero_based)
        apply_pick_layers(model, indices)

    if args.bridge_ln:
        enable_bridge_layernorm(model, args.device)

    model.to(args.device)
    model.eval()

    # ---- Eval per depth ----
    all_results = {}
    for depth in depths:
        if depth > args.limit_report_max_depth:
            continue
        print(f"\nDepth {depth}")
        res = safe_evaluate(args, model, tokenizer, datasets[depth])
        all_results[depth] = res
        print(f"Accuracy: {res['acc']:.4f}")

    # Overall (within window)
    selected = [all_results[d]["acc"] for d in all_results
                if args.limit_report_depth <= d <= args.limit_report_max_depth]
    overall = sum(selected) / len(selected) if selected else 0.0

    print("\n===== Summary =====")
    if args.mode == "tail_k":
        print(f"Mode: tail_k (k={args.k})")
    elif args.mode == "repeat_last":
        print(f"Mode: repeat_last (n={args.n})")
    else:
        print(f"Mode: pick_layers ({'0-based' if args.zero_based else '1-based'}: {args.layers})")
    print(f"Overall Acc (depth {args.limit_report_depth}-{args.limit_report_max_depth}): {overall:.4f}")
    print("\nPer-depth results:")
    pprint.pprint(all_results)


if __name__ == "__main__":
    main()
