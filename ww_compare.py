#!/usr/bin/env python3
"""

Usage examples:
---------------
python ww_compare.py \
  --model1 OUTPUT/RP/BERT_vanilla_batch16/checkpoint-19 \
  --model2 OUTPUT/RP/BERT_singleGPU/checkpoint-19 \
  --name1 vanilla --name2 modified

python ww_compare.py \
  --model1 bert-base-uncased --model2 /path/to/finetuned --name1 base --name2 finetuned \
  --out-prefix myrun
"""

import argparse
import os
import sys
import warnings
from typing import Optional, Tuple

import numpy as np # type: ignore
import pandas as pd # type: ignore
import torch # type: ignore
import weightwatcher as ww # type: ignore
from transformers import ( # type: ignore
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

warnings.filterwarnings("ignore")


# ---------------------------
# HF model loading (no custom computations)
# ---------------------------

def _resolve_tokenizer_source(model_path: str, tokenizer_path: Optional[str]) -> str:
    """Prefer explicit tokenizer_path; else if model_path is a checkpoint dir, use its parent."""
    if tokenizer_path:
        return tokenizer_path
    base = os.path.basename(model_path.rstrip("/"))
    if base.startswith("checkpoint-"):
        return os.path.dirname(model_path.rstrip("/"))
    return model_path


def load_hf_model(model_path: str, tokenizer_path: Optional[str] = None):
    """Load a HF model purely to pass into WeightWatcher."""
    cfg = AutoConfig.from_pretrained(model_path)
    tok_src = _resolve_tokenizer_source(model_path, tokenizer_path)
    _ = AutoTokenizer.from_pretrained(tok_src, use_fast=True)  # not strictly needed by WW, but good to validate

    if getattr(cfg, "is_encoder_decoder", False):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    elif getattr(cfg, "is_decoder", False) and not getattr(cfg, "is_encoder_decoder", False):
        model = AutoModelForCausalLM.from_pretrained(model_path)  # decoder-only
    else:
        model = AutoModel.from_pretrained(model_path)  # encoder-only
    model.eval()
    return model


# ---------------------------
# WeightWatcher runner (WW-only)
# ---------------------------

def run_weightwatcher_all(model, try_plots: bool = False) -> pd.DataFrame:
    """
    Run WW with multiple compatible signatures to maximize returned columns across versions.
    We DO NOT compute any of our own metrics here.
    """
    watcher = ww.WeightWatcher(model=model)

    # Try modern signature first
    try:
        df = watcher.analyze(mp_fit=True, randomize=True, plot=try_plots)
        if isinstance(df, pd.DataFrame):
            return df
    except TypeError:
        # Older/newer versions may not accept some kwargs
        pass
    except Exception:
        # Fall through to next attempts
        pass

    # Fallback 1: mp_fit only
    try:
        df = watcher.analyze(mp_fit=True)
        if isinstance(df, pd.DataFrame):
            return df
    except Exception:
        pass

    # Fallback 2: plain analyze()
    df = watcher.analyze()
    if isinstance(df, pd.DataFrame):
        return df

    # Final fallback: some versions store the table behind a getter
    if hasattr(watcher, "get_details"):
        out = watcher.get_details()
        if isinstance(out, pd.DataFrame):
            return out

    raise RuntimeError("WeightWatcher failed to return a per-layer DataFrame. "
                       "Try `pip install -U weightwatcher`.")


# ---------------------------
# Light summary printer (WW columns only)
# ---------------------------

AGG_MEAN_COLS = [
    "alpha", "alpha_weighted", "log_alpha_norm",
    "sv_max", "lognorm", "mp_softrank", "rand_distance", "log_spectral_norm",
]
AGG_SUM_COLS = ["num_spikes"]

def _safe_mean(x):
    x = pd.to_numeric(x, errors="coerce")
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float("nan")

def _safe_sum(x):
    x = pd.to_numeric(x, errors="coerce")
    x = x[np.isfinite(x)]
    return float(x.sum()) if len(x) else 0.0

def summarize_ww_df(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Build a tiny summary over ONLY columns produced by WW.
    No new metrics are computed; just means (or sums) of existing WW columns.
    """
    rows = []
    for col in AGG_MEAN_COLS:
        if col in df.columns:
            rows.append((label, col + "_mean", _safe_mean(df[col])))
    for col in AGG_SUM_COLS:
        if col in df.columns:
            rows.append((label, col + "_sum", _safe_sum(df[col])))
    return pd.DataFrame(rows, columns=["which", "metric", "value"])


# ---------------------------
# CLI driver
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Run WeightWatcher (only WW metrics) on two models and save per-layer DataFrames.")
    ap.add_argument("--model1", required=True, help="HF id or local path for model 1 (e.g., bert-base-uncased or checkpoint dir)")
    ap.add_argument("--model2", required=True, help="HF id or local path for model 2")
    ap.add_argument("--name1", default="ModelA", help="Readable name for model 1")
    ap.add_argument("--name2", default="ModelB", help="Readable name for model 2")
    ap.add_argument("--tokenizer1", default=None, help="Optional tokenizer path/source for model 1")
    ap.add_argument("--tokenizer2", default=None, help="Optional tokenizer path/source for model 2")
    ap.add_argument("--out-prefix", default="ww_only", help="Prefix for CSV outputs")
    ap.add_argument("--plot", action="store_true", help="Ask WW to plot during analyze (supported in some versions)")
    args = ap.parse_args()

    print("\n=== WeightWatcher (WW-only) ===")
    print(f"Model 1: {args.name1}: {args.model1}")
    print(f"Model 2: {args.name2}: {args.model2}")

    # Load models
    with torch.no_grad():
        model1 = load_hf_model(args.model1, args.tokenizer1)
        model2 = load_hf_model(args.model2, args.tokenizer2)

    # Run WW
    print(f"\n[WW] Analyzing {args.name1} ...")
    df1 = run_weightwatcher_all(model1, try_plots=args.plot)
    print(f"[WW] {args.name1} → rows: {len(df1)}, columns: {list(df1.columns)}")

    print(f"\n[WW] Analyzing {args.name2} ...")
    df2 = run_weightwatcher_all(model2, try_plots=args.plot)
    print(f"[WW] {args.name2} → rows: {len(df2)}, columns: {list(df2.columns)}")

    # Save per-layer DataFrames exactly as provided by WW
    out1 = f"{args.out_prefix}_{args.name1}_ww_details.csv"
    out2 = f"{args.out_prefix}_{args.name2}_ww_details.csv"
    df1.to_csv(out1, index=False)
    df2.to_csv(out2, index=False)
    print(f"\nSaved:\n- {out1}\n- {out2}")

    # Light summaries of WW columns 
    s1 = summarize_ww_df(df1, args.name1)
    s2 = summarize_ww_df(df2, args.name2)
    summary = pd.concat([s1, s2], ignore_index=True)
    outsum = f"{args.out_prefix}_summaries.csv"
    summary.to_csv(outsum, index=False)
    print(f"- {outsum}")

    # Print to console
    if not summary.empty:
        print("\n=== Summary (WW columns ) ===")
        print(summary.pivot(index="metric", columns="which", values="value").to_string())


if __name__ == "__main__":
    main()
