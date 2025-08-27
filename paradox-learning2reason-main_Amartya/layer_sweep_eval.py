# layer_sweep_eval.py
import argparse
import json
import pprint
import csv
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

from safe_eval import safe_evaluate
from dataset import LogicDataset  # same as in safe_eval.py


def truncate_layers_for_eval(model, eval_num_layers: int):
    """In-place truncate encoder layers for BERT/RoBERTa-style models."""
    if eval_num_layers is None:
        return

    if hasattr(model, "bert") and hasattr(model.bert, "encoder"):
        enc = model.bert.encoder
        L = len(enc.layer)
        n = max(1, min(eval_num_layers, L))
        enc.layer = torch.nn.ModuleList(enc.layer[:n])
        model.config.num_hidden_layers = n
        print(f"[Eval] Using {n}/{L} BERT layers")
    elif hasattr(model, "roberta") and hasattr(model.roberta, "encoder"):
        enc = model.roberta.encoder
        L = len(enc.layer)
        n = max(1, min(eval_num_layers, L))
        enc.layer = torch.nn.ModuleList(enc.layer[:n])
        model.config.num_hidden_layers = n
        print(f"[Eval] Using {n}/{L} RoBERTa layers")
    else:
        print("[Eval] Layer truncation not implemented for this model; skipping.")


def parse_layer_list(s: str):
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    # Core paths / types
    parser.add_argument("--model_name_or_path", required=True, type=str)
    parser.add_argument("--tokenizer_name", required=True, type=str)
    parser.add_argument("--val_file_path", required=True, type=str)
    parser.add_argument("--model_type", default="bert", type=str)

    # Eval / device
    parser.add_argument("--eval_batch_size", default=16, type=int)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--device", default="cpu", type=str)

    # Depth reporting (same semantics as your safe_eval defaults)
    parser.add_argument("--limit_report_depth", default=-1, type=int)
    parser.add_argument("--limit_report_max_depth", default=1000, type=int)
    parser.add_argument("--group_by_which_depth", default="depth", type=str)

    # Dataset flags passed through (mirror safe_eval.Args)
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--keep_only_negative", action="store_true")
    parser.add_argument("--skip_long_examples", action="store_true")
    parser.add_argument("--limit_example_num", default=-1, type=int)
    parser.add_argument("--ignore_fact", action="store_true")
    parser.add_argument("--ignore_both", action="store_true")
    parser.add_argument("--ignore_query", action="store_true")
    parser.add_argument("--shorten_input", action="store_true")
    parser.add_argument("--shrink_ratio", default=1, type=int)
    parser.add_argument("--further_split", action="store_true")
    parser.add_argument("--further_further_split", action="store_true")
    parser.add_argument("--max_depth_during_train", default=1000, type=int)

    # Sweep config
    parser.add_argument("--layer_list", default="2,4,6,8,12", type=str,
                        help="Comma-separated layer counts to evaluate.")
    parser.add_argument("--save_csv", default="layer_sweep_results.csv", type=str)
    parser.add_argument("--cache_dir", default=None, type=str)

# OR (one-liner right after args = parser.parse_args())


    args = parser.parse_args()
    args.device = torch.device(args.device)
    layer_counts = parse_layer_list(args.layer_list)
    if not hasattr(args, "cache_dir"):
        args.cache_dir = None

    # Load tokenizer once; make it the single source of truth
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case)

    # Ensure logic markers exist in THIS tokenizer (covers both cases if your dataset lowercases)
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[AND]", "[THEN]", "[and]", "[then]"]}
    )
    if added:
        print(f"Added {added} special tokens to tokenizer.")

    print("Loading datasets (by depth)...")
    datasets = LogicDataset.initialize_from_file_by_depth(args.val_file_path, args)

    # Force datasets to use the same tokenizer instance
    for d in datasets.values():
        d.tokenizer = tokenizer
        d.max_length = args.max_length

    depths = sorted(list(datasets.keys()))
    print("Depths found:", depths)

    sweep_summary = []  # rows for CSV
    overall_acc_by_layers = {}

    for n_layers in layer_counts:
        print("\n" + "=" * 80)
        print(f"Evaluating with {n_layers} encoder layers")
        print("=" * 80)

        # Fresh model load for each layer count
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path, config=config
        )

        # Resize embeddings if tokenizer grew
        if tokenizer.vocab_size > model.get_input_embeddings().weight.shape[0]:
            print(f"Resizing embeddings: {model.get_input_embeddings().weight.shape[0]} -> {tokenizer.vocab_size}")
            model.resize_token_embeddings(len(tokenizer))

        truncate_layers_for_eval(model, n_layers)

        model.to(args.device)
        model.eval()

        per_depth_acc = {}
        for depth in depths:
            if depth > args.limit_report_max_depth:
                continue

            print(f"\nDepth {depth}")
            result = safe_evaluate(args, model, tokenizer, datasets[depth])
            per_depth_acc[depth] = result["acc"]
            print(f"Accuracy: {result['acc']:.4f}")

            # Append CSV row
            sweep_summary.append({
                "layers": n_layers,
                "depth": depth,
                "num_examples": len(datasets[depth]),
                "acc": result["acc"],
                "loss": result.get("loss", float("nan")),
            })

        # Average over the requested reporting window
        selected = [per_depth_acc[d] for d in per_depth_acc
                    if args.limit_report_depth <= d <= args.limit_report_max_depth]
        overall = sum(selected) / len(selected) if selected else 0.0
        overall_acc_by_layers[n_layers] = overall
        print(f"\n==> Overall (depth {args.limit_report_depth}-{args.limit_report_max_depth}) "
              f"with {n_layers} layers: {overall:.4f}")

    print("\n\n===== Summary (Overall Accuracy by Layers) =====")
    for L in sorted(overall_acc_by_layers.keys()):
        print(f"Layers={L}: Overall Acc={overall_acc_by_layers[L]:.4f}")

    # Save CSV
    if args.save_csv:
        fieldnames = ["layers", "depth", "num_examples", "acc", "loss"]
        with open(args.save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sweep_summary)
        print(f"\nSaved per-depth results to {args.save_csv}")


if __name__ == "__main__":
    main()
