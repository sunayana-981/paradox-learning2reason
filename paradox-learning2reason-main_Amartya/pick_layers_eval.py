# pick_layers_eval.py
import argparse
import json
import pprint
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

from safe_eval import safe_evaluate
from dataset import LogicDataset  # make sure this matches your file name

# ---- layer selection helpers ----
def parse_indices(s, L, one_based=True):
    raw = [x.strip() for x in s.split(",") if x.strip()]
    idx = []
    for r in raw:
        i = int(r)
        if i < 0:  # negative indexing (e.g., -1 last)
            i = L + i
        else:
            if one_based:
                i = i - 1
        if not (0 <= i < L):
            raise ValueError(f"Layer index {i} out of range 0..{L-1}")
        idx.append(i)
    # keep order as provided (e.g., 8,12 -> [7,11])
    return idx

def select_layers_for_eval(model, indices):
    if hasattr(model, "bert") and hasattr(model.bert, "encoder"):
        enc = model.bert.encoder
    elif hasattr(model, "roberta") and hasattr(model.roberta, "encoder"):
        enc = model.roberta.encoder
    else:
        raise RuntimeError("Layer picking implemented for BERT/RoBERTa-style encoders only.")
    L = len(enc.layer)
    enc.layer = torch.nn.ModuleList([enc.layer[i] for i in indices])
    model.config.num_hidden_layers = len(indices)
    print(f"[Eval] Using encoder layers (0-based): {indices} of original {L}")

def add_bridge_layernorm(model, device):
    # Apply a LayerNorm to embeddings before the first kept block (reduces distro mismatch)
    if not hasattr(model, "_bridge_ln"):
        model._bridge_ln = torch.nn.LayerNorm(model.config.hidden_size).to(device)
    enc = model.bert.encoder if hasattr(model, "bert") else model.roberta.encoder
    orig_forward = enc.forward

    def bridged_forward(hidden_states, **kw):
        hidden_states = model._bridge_ln(hidden_states)
        return orig_forward(hidden_states, **kw)

    enc.forward = bridged_forward
    print("[Eval] Enabled LayerNorm bridge before first kept block")

def main():
    p = argparse.ArgumentParser()
    # core paths
    p.add_argument("--model_name_or_path", required=True, type=str)
    p.add_argument("--tokenizer_name", required=True, type=str)
    p.add_argument("--val_file_path", required=True, type=str)
    # model/data flags
    p.add_argument("--model_type", default="bert", type=str)         # bert/roberta
    p.add_argument("--do_lower_case", action="store_true")
    p.add_argument("--cache_dir", default=None, type=str)
    p.add_argument("--eval_batch_size", default=16, type=int)
    p.add_argument("--max_length", default=512, type=int)
    p.add_argument("--device", default="cpu", type=str)
    p.add_argument("--group_by_which_depth", default="depth", type=str)
    p.add_argument("--limit_report_depth", default=-1, type=int)
    p.add_argument("--limit_report_max_depth", default=1000, type=int)
    # dataset passthroughs
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
    # layer picking
    p.add_argument("--pick_layers", default="8,12", type=str,
                   help="Layers to keep (comma-separated). 1-based indexing by default.")
    p.add_argument("--one_based", action="store_true", default=True,
                   help="Interpret --pick_layers as 1-based indices (default). Use --no-one_based for 0-based.")
    p.add_argument("--no-one_based", dest="one_based", action="store_false")
    p.add_argument("--bridge_ln", action="store_true",
                   help="Add a LayerNorm bridge before first kept block.")
    args = p.parse_args()
    args.device = torch.device(args.device)

    # ---- tokenizer ----
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, do_lower_case=args.do_lower_case, cache_dir=args.cache_dir
    )
    # be robust to casing
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[AND]", "[THEN]", "[and]", "[then]"]}
    )
    if added:
        print(f"Added {added} special tokens to tokenizer.")

    # ---- dataset (by depth) ----
    print("Loading datasets (by depth)...")
    datasets = LogicDataset.initialize_from_file_by_depth(args.val_file_path, args)
    for d in datasets.values():
        # force the shared tokenizer + max_length
        d.tokenizer = tokenizer
        d.max_length = args.max_length

    depths = sorted(list(datasets.keys()))
    print("Depths found:", depths)

    # ---- model ----
    print("Loading model...")
    config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, config=config, cache_dir=args.cache_dir
    )

    # resize embeddings if tokenizer grew
    emb_sz = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > emb_sz:
        print(f"Resizing embeddings: {emb_sz} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    # pick layers
    total_layers = (len(model.bert.encoder.layer)
                    if hasattr(model, "bert") else len(model.roberta.encoder.layer))
    picked = parse_indices(args.pick_layers, total_layers, one_based=args.one_based)
    select_layers_for_eval(model, picked)
    if args.bridge_ln:
        add_bridge_layernorm(model, args.device)

    model.to(args.device)
    model.eval()

    # ---- eval per depth ----
    all_results = {}
    for depth in depths:
        if depth > args.limit_report_max_depth:
            continue
        print(f"\nDepth {depth}")
        res = safe_evaluate(args, model, tokenizer, datasets[depth])
        all_results[depth] = res
        print(f"Accuracy: {res['acc']:.4f}")

    # overall (within depth window)
    sel = [all_results[d]["acc"] for d in all_results
           if args.limit_report_depth <= d <= args.limit_report_max_depth]
    overall = sum(sel) / len(sel) if sel else 0.0

    print("\n===== Summary (Picked Layers) =====")
    print(f"Layers kept (0-based): {picked}")
    print(f"Overall Acc (depth {args.limit_report_depth}-{args.limit_report_max_depth}): {overall:.4f}")
    print("\nPer-depth results:")
    pprint.pprint(all_results)

if __name__ == "__main__":
    main()
