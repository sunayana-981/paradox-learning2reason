#!/usr/bin/env python3
# coding=utf-8
"""
Evaluate a Qwen + LoRA classifier trained with your script.

Features
- Loads base CausalLM + LoRA adapters (PEFT) or a fully saved model dir.
- Dynamic right-trim padding for speed.
- Metrics: accuracy (default). Optional: precision/recall/F1, confusion matrix.
- Multi-file eval (comma-separated) + optional per-depth breakdown.
- 4-bit / 8-bit and bf16/fp16 friendly.

Usage
------
python eval_qwen_lora.py \
  --model_path OUTPUT/my_run \
  --val_file_path data/val.jsonl \
  --per_gpu_eval_batch_size 8 \
  --bf16 \
  --report_prf1 \
  --save_csv metrics.csv

# Multiple datasets:
python eval_qwen_lora.py \
  --model_path OUTPUT/my_run \
  --val_file_path data/val_a.jsonl,data/val_b.jsonl \
  --per_depth
"""

import argparse
import json
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

# Optional PEFT import(s)
try:
    from peft import AutoPeftModelForCausalLM, PeftModel
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False

# Project-local
from dataset_new import LogicDataset  # must provide: initialize/initialize_by_depth + .collate_fn
from helpers import *  # optional; only if your dataset expects it

# Optional: for safetensors head loading
try:
    from safetensors.torch import load_file as safe_load_file
except Exception:
    safe_load_file = None

logger = logging.getLogger("eval_qwen_lora")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_dataloader(dataset, batch_size: int) -> DataLoader:
    """Uses dataset.collate_fn; also trims right padding inside collate for speed if supported."""
    def dynamic_right_trim(batch):
        # dataset.collate_fn returns: (input_ids, attention_mask, token_type_ids, labels, examples)
        input_ids, attention_mask, token_type_ids, labels, examples = dataset.collate_fn(batch)
        max_len = int(attention_mask.sum(dim=1).max().item())

        ids = input_ids[:, :max_len].contiguous()
        attn = attention_mask[:, :max_len].contiguous()

        # Normalize token_type_ids to 2D
        if token_type_ids is None:
            tti = torch.zeros_like(ids)
        else:
            if token_type_ids.dim() == 1:
                # broadcast single row across batch
                tti = token_type_ids[:max_len].unsqueeze(0).expand(ids.size(0), -1).contiguous()
            elif token_type_ids.dim() == 2:
                tti = token_type_ids[:, :max_len].contiguous()
            else:
                tti = torch.zeros_like(ids)

        return (ids, attn, tti, labels, examples)

    return DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=batch_size,
        collate_fn=dynamic_right_trim,
        pin_memory=True,
    )


@torch.no_grad()
def _run_eval_loop(model, dataloader, device) -> Tuple[np.ndarray, np.ndarray]:
    """Core eval loop returning (preds, labels). Assumes model has .classification_head."""
    model.eval()
    all_preds, all_labels = [], []

    for batch in tqdm(dataloader, desc="Evaluating"):
        # Unpack directly; move only needed tensors
        input_ids, attention_mask, _token_type_ids, labels, _examples = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        if outputs.hidden_states is None:
            raise RuntimeError("Hidden states not returned. Ensure model.config.output_hidden_states=True.")

        last_hidden = outputs.hidden_states[-1]
        seq_lens = attention_mask.long().sum(dim=1) - 1  # last non-pad position
        pooled = last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), seq_lens]
        # NEW: cast pooled to head weight dtype if mismatch
        head_dtype = model.classification_head.weight.dtype
        if pooled.dtype != head_dtype:
            pooled = pooled.to(head_dtype)
        logits = model.classification_head(pooled)

        batch_preds = torch.argmax(logits, dim=-1)
        all_preds.append(batch_preds.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    preds = np.concatenate(all_preds) if all_preds else np.array([])
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    return preds, labels


def _accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    if preds.size == 0:
        return 0.0
    return float((preds == labels).mean())


def _prf1(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Binary metrics with label set {0,1}. Safe to compute without sklearn."""
    metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if preds.size == 0:
        return metrics

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    metrics.update({"precision": precision, "recall": recall, "f1": f1})
    return metrics


def _confusion_matrix(preds: np.ndarray, labels: np.ndarray) -> List[List[int]]:
    """2x2 confusion matrix [[tn, fp],[fn, tp]] for binary labels 0/1."""
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tp = int(((preds == 1) & (labels == 1)).sum())
    return [[tn, fp], [fn, tp]]


def _try_load_model(model_path: str,
                    dtype,
                    cache_dir: str = "",
                    use_4bit: bool = False,
                    use_8bit: bool = False,
                    base_model_path: str = "",
                    adapter_path: str = ""):
    """
    Load either:
      A) full saved model at model_path, or
      B) base model at base_model_path + attach adapters from adapter_path/model_path.
    Returns (model, tokenizer).
    """
    def _has_full_weights(p: str) -> bool:
        if not os.path.isdir(p):
            return False
        for fn in os.listdir(p):
            lfn = fn.lower()
            if "adapter" in lfn:               # <-- ignore adapter files
                continue
            if lfn.endswith(".safetensors") and ("model" in lfn or "pytorch_model" in lfn):
                return True
            if lfn.endswith(".bin") and ("pytorch_model" in lfn or lfn.startswith("model")):
                return True
        return False

    adapter_path = adapter_path or model_path

    # If user provides base_model_path, force base+adapter path
    if base_model_path:
        is_full = False
    else:
        is_full = _has_full_weights(model_path)

    # Prefer tokenizer from adapter dir (it contains added tokens) else fallback
    tok_src = adapter_path if os.path.isdir(adapter_path) else (base_model_path or model_path)
    tok_kwargs = dict(trust_remote_code=True)
    if cache_dir:
        tok_kwargs["cache_dir"] = cache_dir
    tokenizer = AutoTokenizer.from_pretrained(tok_src, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Common model kwargs
    model_kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
    if cache_dir:
        model_kwargs["cache_dir"] = cache_dir
    if use_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif use_8bit:
        model_kwargs["load_in_8bit"] = True

    # Case A: load full model directory (merged weights or full Trainer save)
    if is_full and not base_model_path:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        logger.info("Loaded FULL model from %s", model_path)

    # Case B: load base then attach adapters
    else:
        if not base_model_path:
            raise ValueError(
                "Adapter-only load detected. Pass --base_model_path (e.g., qwen3-4B-dense)."
            )
        model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)
        logger.info("Loaded BASE model from %s", base_model_path)

        # force vocab length from adapter tokenizer BEFORE attaching adapters
        vocab_len = len(tokenizer)  # should be 151,671
        model.resize_token_embeddings(vocab_len)

        # now attach adapters
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        logger.info("Attached LoRA adapters from %s", adapter_path)

        model.config.output_hidden_states = True


    # Ensure hidden states and align embeddings AFTER adapters are on the model
    model.config.output_hidden_states = True
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


# ---------- NEW: helper to ensure classification head exists ----------
def ensure_classification_head(model, adapter_dir: str, device, num_labels: int = 2):
    """
    Ensure model has a `.classification_head` (Linear hidden_size -> num_labels).
    Tries to load from common filenames inside adapter_dir; otherwise initializes fresh.
    """
    if hasattr(model, "classification_head"):
        return  # already present

    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        # Fallbacks some Qwen variants may use
        hidden_size = getattr(model.config, "n_embd", None)
    if hidden_size is None:
        raise ValueError("Cannot determine hidden_size to build classification head.")

    head = torch.nn.Linear(hidden_size, num_labels)
    loaded = False

    candidate_files = [
        "classification_head.safetensors",
        "classification_head.pt",
        "classification_head.bin",
        "cls_head.safetensors",
        "cls_head.pt",
        "cls_head.bin",
        "head.pt",
        "head.bin",
    ]
    for fname in candidate_files:
        fpath = os.path.join(adapter_dir, fname)
        if os.path.isfile(fpath):
            try:
                if fpath.endswith(".safetensors") and safe_load_file is not None:
                    state = safe_load_file(fpath, device="cpu")
                else:
                    state = torch.load(fpath, map_location="cpu")
                head.load_state_dict(state)
                loaded = True
                print(f"Loaded classification head weights from {fpath}")
                break
            except Exception as e:
                print(f"Failed loading head from {fpath}: {e}")

    if not loaded:
        # Fresh init (Xavier)
        torch.nn.init.xavier_uniform_(head.weight)
        if head.bias is not None:
            torch.nn.init.zeros_(head.bias)
        print("Initialized new classification head (no saved head found).")

    model.classification_head = head.to(device)
    # NEW: enforce dtype match with backbone (handles bf16/fp16 cases)
    target_dtype = next(model.parameters()).dtype
    model.classification_head.to(dtype=target_dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to saved training output (full model or adapter dir).")
    # NEW: base + adapter overrides
    parser.add_argument("--base_model_path", type=str, default="",
                        help="Base Qwen model path/repo (required if model_path is adapter-only).")
    parser.add_argument("--adapter_path", type=str, default="",
                        help="Explicit adapter dir (defaults to --model_path).")
    parser.add_argument("--val_file_path", type=str, required=True,
                        help="Validation file path or comma-separated list.")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--per_gpu_eval_batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    # dtype / quantization
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--use_8bit", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")

    # reporting
    parser.add_argument("--report_prf1", action="store_true", help="Also compute precision/recall/F1.")
    parser.add_argument("--confmat", action="store_true", help="Also print 2x2 confusion matrix.")
    parser.add_argument("--save_json", type=str, default="", help="Save metrics dict to JSON file.")
    parser.add_argument("--save_csv", type=str, default="", help="Append per-file metrics to CSV.")
    parser.add_argument("--per_depth", action="store_true", help="Breakdown by logical depth if supported.")
    parser.add_argument("--num_labels", type=int, default=2, help="Number of labels for classification head.")

    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Device
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"Device: {device}")

    # Dtype
    dtype = torch.float32
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16

    # Seed
    set_seed(args.seed)

    # Load model + tokenizer
    model, tokenizer = _try_load_model(
        args.model_path, dtype, cache_dir=args.cache_dir,
        use_4bit=args.use_4bit, use_8bit=args.use_8bit,
        base_model_path=args.base_model_path, adapter_path=args.adapter_path
    )
    model.to(device)

    # Ensure classification head (pass adapter_path or model_path as source for possible saved head)
    ensure_classification_head(
        model,
        adapter_dir=(args.adapter_path or args.model_path),
        device=device,
        num_labels=args.num_labels
    )

    # CUDA niceties
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True
    try:
        torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
    except Exception:
        pass

    # Build dataset args stub for LogicDataset
    dataset_args = argparse.Namespace()
    dataset_args.tokenizer_name = args.model_path
    dataset_args.do_lower_case = False
    dataset_args.model_name_or_path = args.model_path
    dataset_args.group_by_which_depth = getattr(dataset_args, "group_by_which_depth", "depth")
    dataset_args.limit_report_depth = getattr(dataset_args, "limit_report_depth", -1)
    dataset_args.limit_report_max_depth = getattr(dataset_args, "limit_report_max_depth", 100)
    dataset_args.shorten_input = getattr(dataset_args, "shorten_input", False)
    dataset_args.ignore_fact = getattr(dataset_args, "ignore_fact", False)
    dataset_args.ignore_both = getattr(dataset_args, "ignore_both", False)
    dataset_args.ignore_query = getattr(dataset_args, "ignore_query", False)
    dataset_args.keep_only_negative = getattr(dataset_args, "keep_only_negative", False)
    dataset_args.skip_long_examples = getattr(dataset_args, "skip_long_examples", False)
    dataset_args.limit_example_num = getattr(dataset_args, "limit_example_num", -1)
    dataset_args.report_example_length = getattr(dataset_args, "report_example_length", False)
    dataset_args.max_length = getattr(dataset_args, "max_length", 512)
    dataset_args.cache_dir = getattr(dataset_args, "cache_dir", None)

    val_files = args.val_file_path.split(",") if "," in args.val_file_path else [args.val_file_path]
    all_metrics = {}

    for vf in val_files:
        vf = vf.strip()
        if not vf:
            continue
        logger.info(f"Evaluating file: {vf}")

        if args.per_depth and hasattr(LogicDataset, "initialize_from_file_by_depth"):
            datasets_by_depth = LogicDataset.initialize_from_file_by_depth(vf, dataset_args)
            for d in datasets_by_depth:
                datasets_by_depth[d].tokenizer = tokenizer

            depths = sorted(datasets_by_depth.keys())
            total = sum(len(datasets_by_depth[d]) for d in depths)

            per_depth_metrics = {}
            for d in depths:
                ds = datasets_by_depth[d]
                dl = _build_dataloader(ds, args.per_gpu_eval_batch_size)
                preds, labels = _run_eval_loop(model, dl, device)
                acc = _accuracy(preds, labels)
                entry = {"acc": acc, "count": int(len(ds)), "fraction": float(len(ds) / max(1, total))}
                if args.report_prf1:
                    entry.update(_prf1(preds, labels))
                if args.confmat:
                    entry["confusion_matrix"] = _confusion_matrix(preds, labels)
                per_depth_metrics[int(d)] = entry

            # Weighted overall (by count)
            total_correct = 0
            total_count = 0
            for d, m in per_depth_metrics.items():
                total_correct += int(m["acc"] * m["count"])
                total_count += m["count"]
            overall_acc = (total_correct / total_count) if total_count > 0 else 0.0
            all_metrics[vf] = {"overall_acc": overall_acc, "per_depth": per_depth_metrics}

        else:
            # Single dataset path
            ds = LogicDataset.initialze_from_file(vf, dataset_args)
            ds.tokenizer = tokenizer
            dl = _build_dataloader(ds, args.per_gpu_eval_batch_size)
            preds, labels = _run_eval_loop(model, dl, device)

            metrics = {"acc": _accuracy(preds, labels)}
            if args.report_prf1:
                metrics.update(_prf1(preds, labels))
            if args.confmat:
                metrics["confusion_matrix"] = _confusion_matrix(preds, labels)

            all_metrics[vf] = metrics

    # Print nicely
    print("\n=== Evaluation Results ===")
    print(json.dumps(all_metrics, indent=2))

    # Optional saves
    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"Saved JSON -> {args.save_json}")

    if args.save_csv:
        import csv
        write_header = not os.path.exists(args.save_csv)
        with open(args.save_csv, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["file", "mode", "acc", "precision", "recall", "f1", "count"])
            for vf, m in all_metrics.items():
                if "per_depth" in m:
                    # write overall first
                    w.writerow([vf, "overall", f"{m['overall_acc']:.6f}", "", "", "", ""])
                    for d, md in sorted(m["per_depth"].items()):
                        w.writerow([
                            vf, f"depth_{d}",
                            f"{md['acc']:.6f}",
                            f"{md.get('precision','') if isinstance(md.get('precision'), float) else ''}",
                            f"{md.get('recall','') if isinstance(md.get('recall'), float) else ''}",
                            f"{md.get('f1','') if isinstance(md.get('f1'), float) else ''}",
                            f"{md.get('count','')}",
                        ])
                else:
                    w.writerow([
                        vf, "all",
                        f"{m['acc']:.6f}",
                        f"{m.get('precision','') if isinstance(m.get('precision'), float) else ''}",
                        f"{m.get('recall','') if isinstance(m.get('recall'), float) else ''}",
                        f"{m.get('f1','') if isinstance(m.get('f1'), float) else ''}",
                        "",
                    ])
        print(f"Appended CSV -> {args.save_csv}")


if __name__ == "__main__":
    main()
