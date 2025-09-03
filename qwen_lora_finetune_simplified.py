# coding=utf-8
"""
LoRA fine-tuning for Qwen (causal LM backbone) on LogicDataset classification.

Highlights:
- Robust last-layer features from the backbone (no 'last_hidden_state' crash).
- Dynamic right-trim padding in collator + eval (big FLOPs savings).
- Optional --bf16 with clear dtype selection (bf16 > fp16 > fp32).
- Optional val file; cleaner argparse checks.
- Fused AdamW + SDPA/Flash attention preferred when available.
- Gradient checkpointing warning fixed via use_reentrant=False (when supported).
"""

import argparse
import inspect
import logging
import os
import random
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# Project helpers + dataset
from helpers import *
from dataset_new import LogicDataset

logger = logging.getLogger(__name__)


# ---------------------------
# Utilities\
    
# ---------------------------

def set_seed(seed: int, n_gpu: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def prepare_qwen_for_classification(model, num_labels=2):
    """Add a classification head to Qwen model."""
    hidden_size = model.config.hidden_size
    classification_head = torch.nn.Linear(hidden_size, num_labels)
    model.classification_head = classification_head.to(next(model.parameters()).device)
    return model


def _backbone(model):
    """Return the transformer backbone module of a CausalLM model."""
    return getattr(model, "base_model", None) or getattr(model, "model", None) or model


def last_hidden(model, input_ids, attention_mask):
    """
    Robustly obtain last hidden states:
    - Prefer backbone(..., return_dict=True).last_hidden_state
    - Fallback to hidden_states[-1] (retry enabling output_hidden_states if needed)
    """
    backbone = _backbone(model)
    out = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return out.last_hidden_state
    if hasattr(out, "hidden_states") and out.hidden_states is not None:
        return out.hidden_states[-1]
    # One retry with hidden states enabled
    out = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    return out.hidden_states[-1]


# ---------------------------
# Custom Trainer (last token pooling + CE loss)
# ---------------------------

class QwenLoRATrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Custom loss for sequence classification with Qwen + LoRA.
        Accepts **kwargs to stay forward-compatible with Trainer
        (e.g., num_items_in_batch).
        """
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

        if outputs.hidden_states is None:
            raise RuntimeError("Hidden states not returned. Ensure model.config.output_hidden_states = True")

        last_hidden = outputs.hidden_states[-1]
        seq_lens = attention_mask.long().sum(dim=1) - 1
        pooled = last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), seq_lens]

        logits = model.classification_head(pooled)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        return (loss, {"logits": logits}) if return_outputs else loss


# ---------------------------
# TrainingArguments builder
# ---------------------------

def build_training_args(args, has_eval: bool) -> TrainingArguments:
    sig = inspect.signature(TrainingArguments.__init__)
    kw = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_gpu_train_batch_size,
        per_device_eval_batch_size=args.per_gpu_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=int(args.warmup_steps),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        report_to=[],
        gradient_checkpointing=args.gradient_checkpointing,
    )
    if "fp16" in sig.parameters:
        kw["fp16"] = args.fp16
    if "bf16" in sig.parameters:
        kw["bf16"] = args.bf16
    if "optim" in sig.parameters:
        kw["optim"] = "adamw_torch_fused"  # nice speedup when available
    return TrainingArguments(**kw)


# ---------------------------
# LoRA Training
# ---------------------------

def train_with_lora(args, train_dataset, model, tokenizer, eval_dataset=None):
    # LoRA target modules (allow override)
    if args.lora_targets:
        target_modules = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
    else:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )

    # 4/8-bit prep
    if args.use_4bit or args.use_8bit:
        model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Classification head
    model = prepare_qwen_for_classification(model, num_labels=2)
    model.classification_head.to(args.device)

    # Memory-friendly defaults
    model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        # Silence the 2.5 warning by favoring the non-reentrant path when supported
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # Training args
    training_args = build_training_args(args, has_eval=(eval_dataset is not None))

    # Collator with dynamic right-trim
    def data_collator(batch):
        input_ids, attention_mask, token_type_ids, labels, examples = train_dataset.collate_fn(batch)
        max_len = int(attention_mask.sum(dim=1).max().item())
        input_ids = input_ids[:, :max_len].contiguous()
        attention_mask = attention_mask[:, :max_len].contiguous()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    trainer = QwenLoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,  # OK (HF warns; fine until v5)
        data_collator=data_collator,
    )

    trainer.train()
    return trainer


# ---------------------------
# Manual Evaluation
# ---------------------------

@torch.no_grad()
def evaluate_lora(args, model, tokenizer, eval_dataset):
    """Evaluate the LoRA model."""
    results = {}
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=eval_dataset.collate_fn,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size
    )

    model.eval()
    preds, labels = [], []

    print("***** Running evaluation *****")
    print(f"  Num examples = {len(eval_dataset)}")
    print(f"  Batch size = {args.eval_batch_size}")

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch_data, examples = batch[:-1], batch[-1]
        batch_data = tuple(t.to(args.device) for t in batch_data)
        input_ids, attention_mask, token_type_ids, batch_labels = batch_data

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True
            )
            if outputs.hidden_states is None:
                raise RuntimeError("Hidden states not returned in evaluation.")
            last_hidden = outputs.hidden_states[-1]
            seq_lens = attention_mask.long().sum(dim=1) - 1
            pooled = last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), seq_lens]
            logits = model.classification_head(pooled)
            batch_preds = torch.argmax(logits, dim=-1)
            preds.extend(batch_preds.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())

    accuracy = np.mean(np.array(preds) == np.array(labels))
    results["acc"] = accuracy
    return results


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()

    # Model / output
    parser.add_argument("--model_name_or_path", type=str, default="./qwen3-4B-dense",
                        help="Local path or HF repo id to the base Qwen model.")
    parser.add_argument("--output_dir", type=str, required=True)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_targets", type=str, default="",
                        help='Comma-separated target modules (e.g., "q_proj,v_proj"). Empty=default broad set.')

    # Quantization
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--use_8bit", action="store_true")

    # Train/eval
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--per_gpu_train_batch_size", type=int, default=1)
    parser.add_argument("--per_gpu_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--gradient_checkpointing", action="store_true")

    # System
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")   # added
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--no_cuda", action="store_true")

    # Dataset knobs used by LogicDataset
    parser.add_argument("--train_file_path", type=str, default="",
                        help="Training file path. Required if --do_train.")
    parser.add_argument("--val_file_path", type=str, default="",
                        help="Validation file path. Required if --do_eval. Comma-separated for multiple.")
    parser.add_argument("--group_by_which_depth", type=str, default="depth")
    parser.add_argument("--limit_example_num", type=int, default=-1)
    parser.add_argument("--skip_long_examples", action="store_true")
    parser.add_argument("--keep_only_negative", action="store_true")
    parser.add_argument("--ignore_fact", action="store_true")
    parser.add_argument("--ignore_both", action="store_true")
    parser.add_argument("--ignore_query", action="store_true")
    parser.add_argument("--shorten_input", action="store_true")
    parser.add_argument("--limit_report_depth", type=int, default=-1)
    parser.add_argument("--limit_report_max_depth", type=int, default=100)
    parser.add_argument("--report_example_length", action="store_true")
    parser.add_argument("--max_length", type=int, default=512)

    args = parser.parse_args()

    # Device / distributed
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        n_gpu = 1
    args.device = device
    args.n_gpu = n_gpu

    # Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, fp16: %s, bf16: %s",
        args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16, args.bf16,
    )

    # Sanity: required file args
    if args.do_train and not args.train_file_path:
        raise ValueError("`--do_train` requires `--train_file_path`.")
    if args.do_eval and not args.val_file_path:
        raise ValueError("`--do_eval` requires `--val_file_path`.")

    # Seed
    set_seed(args.seed, args.n_gpu)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
        padding_side="right",
        trust_remote_code=True,
    )

    # Add special tokens
    tokenizer.add_special_tokens({"additional_special_tokens": ["[AND]", "[THEN]"]})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dtype preference: bf16 > fp16 > fp32
    preferred_dtype = torch.float32
    if args.bf16:
        preferred_dtype = torch.bfloat16
    elif args.fp16:
        preferred_dtype = torch.float16

    # Model load kwargs (avoid duplicate cache_dir / torch_dtype)
    model_kwargs = {
        "torch_dtype": preferred_dtype,
        "trust_remote_code": True,
    }
    if args.cache_dir:
        model_kwargs["cache_dir"] = args.cache_dir

    if args.use_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=preferred_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif args.use_8bit:
        model_kwargs["load_in_8bit"] = True

    # Try primary load (no duplicated kwargs)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            **model_kwargs
        )
    except TypeError as e:
        logger.warning(f"Primary load failed ({e}); retrying after pruning unsupported kwargs.")
        # Remove keys that might not be accepted by older versions
        fallback_kwargs = dict(model_kwargs)
        for k in ["trust_remote_code"]:
            fallback_kwargs.pop(k, None)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            **fallback_kwargs
        )
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        raise

    # Ensure hidden states are produced
    model.config.output_hidden_states = True

    # Resize embeddings for new tokens
    model.resize_token_embeddings(len(tokenizer))

    # Nice-to-haves (OOM + speed)
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
    except Exception:
        pass

    print("Training/evaluation parameters:", args)

    # -------- Train --------
    if args.do_train:
        dataset_args = argparse.Namespace(**vars(args))
        dataset_args.tokenizer_name = args.model_name_or_path
        dataset_args.do_lower_case = False
        dataset_args.model_name_or_path = args.model_name_or_path

        train_dataset = LogicDataset.initialze_from_file(args.train_file_path, dataset_args)
        train_dataset.tokenizer = tokenizer
        train_dataset.report_length()

        val_dataset = None
        if args.val_file_path:
            val_dataset = LogicDataset.initialze_from_file(args.val_file_path, dataset_args)
            val_dataset.tokenizer = tokenizer

        # Prepare LoRA AFTER ensuring output_hidden_states flag
        trainer = train_with_lora(args, train_dataset, model, tokenizer, val_dataset)

        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)

    # -------- Eval --------
    if args.do_eval and args.val_file_path:
        print("Entering evaluation")
        dataset_args = argparse.Namespace(**vars(args))
        dataset_args.tokenizer_name = args.model_name_or_path
        dataset_args.do_lower_case = False
        dataset_args.model_name_or_path = args.model_name_or_path

        val_files = args.val_file_path.split(",") if "," in args.val_file_path else [args.val_file_path]
        all_results = {}

        for val_file in val_files:
            if not val_file:
                continue
            print("\n\nEvaluating:", val_file)
            val_dataset = LogicDataset.initialze_from_file(val_file, dataset_args)
            val_dataset.tokenizer = tokenizer
            if hasattr(val_dataset, "report_allkinds_of_stats"):
                val_dataset.report_allkinds_of_stats()

            # Optional per-depth breakdown
            if hasattr(LogicDataset, "initialize_from_file_by_depth"):
                datasets_by_depth = LogicDataset.initialize_from_file_by_depth(val_file, dataset_args)
                for depth in datasets_by_depth:
                    datasets_by_depth[depth].tokenizer = tokenizer

                depths = sorted(datasets_by_depth.keys())
                total_examples = sum(len(datasets_by_depth[d]) for d in depths)

                results = []
                results_string = {}
                for depth in depths:
                    print(f"\nEvaluating depth {depth}")
                    r = evaluate_lora(args, model, tokenizer, eval_dataset=datasets_by_depth[depth])
                    results_string[depth] = f"Acc: {r['acc']:.4f} ; Percentage {len(datasets_by_depth[depth]) / total_examples:.4f}"
                    if args.limit_report_depth <= depth <= args.limit_report_max_depth:
                        results.append(r["acc"])

                import pprint
                pprint.pprint(results_string)
                all_results[val_file] = f"{(sum(results) / len(results)) * 100:.3f}" if results else "0.000"
            else:
                r = evaluate_lora(args, model, tokenizer, eval_dataset=val_dataset)
                all_results[val_file] = f"{r['acc'] * 100:.3f}"

        print("Final Results:")
        import pprint
        pprint.pprint(all_results)


if __name__ == "__main__":
    main()
