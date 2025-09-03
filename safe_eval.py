# safe_eval.py
import torch
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from collections import defaultdict
import pprint
import sys
sys.path.append('.')

from dataset_new import LogicDataset

def safe_evaluate(args, model, tokenizer, eval_dataset):
    """Evaluation with safety checks"""
    results = {}
    
    # Get actual embedding size
    actual_vocab_size = model.get_input_embeddings().weight.shape[0]
    print(f"Actual model embedding size: {actual_vocab_size}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, 
        collate_fn=eval_dataset.collate_fn, 
        sampler=eval_sampler, 
        batch_size=args.eval_batch_size
    )
    
    print(f"***** Running evaluation *****")
    print(f"  Num examples = {len(eval_dataset)}")
    print(f"  Batch size = {args.eval_batch_size}")
    
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    
    for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
        model.eval()
        batch_data, examples = batch[:-1], batch[-1]
        
        # Move to device
        batch_data = tuple(t.to(args.device) for t in batch_data)
        
        with torch.no_grad():
            inputs = {
                "input_ids": batch_data[0], 
                "attention_mask": batch_data[1], 
                "labels": batch_data[3]
            }
            
            # --- FIXED SECTION ---
            # VALIDATION: Check for out-of-bounds token IDs
            max_token = inputs["input_ids"].max().item()
            if max_token >= actual_vocab_size:
                print(f"\nWARNING: Batch {batch_idx} contains a token ID ({max_token}) that is out of the model's vocabulary size ({actual_vocab_size}).")
                print(f"This indicates a mismatch between the tokenizer and the model.")
                print("SKIPPING this batch to avoid data corruption and inaccurate results.")
                continue # Skip the rest of the loop for this batch
            
            # Add token type ids if needed
            if args.model_type in ["bert", "xlnet", "albert"]:
                inputs["token_type_ids"] = batch_data[2]
            
            # The rest of the logic proceeds only if the batch is valid
            try:
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                
                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1
                
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs["labels"].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
    
    # Handle case where all batches were skipped
    if nb_eval_steps == 0:
        print("WARNING: All evaluation batches were skipped due to vocabulary mismatch. Cannot compute metrics.")
        return {"acc": 0, "loss": float('inf')}

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    accuracy = (out_label_ids == preds).mean()
    
    results = {"acc": accuracy, "loss": eval_loss}
    return results

def main():
    class Args:
        model_name_or_path = "OUTPUT/RP/BERT/checkpoint-19"
        # Use the same dir as the model checkpoint so tokenizer & model stay in lockstep
        tokenizer_path = "OUTPUT/RP/BERT"
        val_file_path = "DATA/RP/prop_examples.balanced_by_backward.max_6.json_train"  # <- use test/val file
        model_type = "bert"
        do_lower_case = True
        eval_batch_size = 16
        max_length = 512
        device = torch.device("cpu")
        n_gpu = 0
        local_rank = -1
        cache_dir = None
        group_by_which_depth = "depth"
        limit_report_depth = -1
        limit_report_max_depth = 6
        # dataset args...
        keep_only_negative = False
        skip_long_examples = False
        limit_example_num = -1
        ignore_fact = False
        ignore_both = False
        ignore_query = False
        shorten_input = False
        shrink_ratio = 1
        further_split = False
        further_further_split = False
        max_depth_during_train = 1000

    args = Args()

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    # Optional: enforce specials if missing, then resize model later.
    specials = ["[AND]", "[THEN]"]
    if not all(t in tokenizer.get_vocab() for t in specials):
        print("Special tokens missing in tokenizer; adding them now.")
        tokenizer.add_special_tokens({"additional_special_tokens": specials})

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)

    # Ensure embeddings cover tokenizer size
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print(f"Resizing model embeddings from {model.get_input_embeddings().weight.shape[0]} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    model.to(args.device)
    model.eval()

    print("\nLoading datasets...")
    # IMPORTANT: pass tokenizer so dataset doesn't create its own
    datasets = LogicDataset.initialize_from_file_by_depth(args.val_file_path, args, tokenizer=tokenizer)
    depths = sorted(list(datasets.keys()))

    all_results = {}
    for depth in depths:
        if depth > args.limit_report_max_depth:
            continue
        print(f"\n\nEvaluating examples of depth {depth}")
        result = safe_evaluate(args, model, tokenizer, datasets[depth])
        all_results[depth] = result
        print(f"Depth {depth}: Accuracy = {result['acc']:.4f}")

    print("\n\nFinal Results:")
    pprint.pprint(all_results)

    reported_accs = [all_results[d]['acc'] for d in all_results
                     if args.limit_report_depth <= d <= args.limit_report_max_depth]
    if reported_accs:
        avg_acc = sum(reported_accs) / len(reported_accs)
        print(f"\nAverage accuracy (depth {args.limit_report_depth}-{args.limit_report_max_depth}): {avg_acc:.4f}")

if __name__ == "__main__":
    main()