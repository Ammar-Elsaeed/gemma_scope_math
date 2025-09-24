#!/usr/bin/env python3
import argparse
import os
import torch as t
import pandas as pd
import json
import re
import logging
from transformers import AutoTokenizer
from sae_lens import HookedSAETransformer, SAE
from datetime import datetime
import timeit
t.set_grad_enabled(False)

# Compile regex patterns once at module level for efficiency
CLEAN_PROBLEM_PATTERN = re.compile(r'[^\d\s\+\-\*/]')
MATH_PATTERN = re.compile(r'(\d+\s*[\+\-\*/]\s*\d+)')

# Global logger instance
logger = None

def setup_logger(log_path):
    """Setup efficient logger with buffered I/O."""
    global logger
    logger = logging.getLogger('parallel_ablation')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler with buffering (8KB buffer for efficiency)
    handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    handler.setLevel(logging.INFO)
    
    # Set custom formatter with timestamp
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', 
                                datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    return logger

def write_log(msg, level="INFO"):
    """Write log message using efficient logging system.

    Args:
        msg (str): Message to write.
        level (str): Log level string, e.g. 'INFO' or 'ERROR'.
    """
    if logger is None:
        return  # Silently ignore if logger not initialized
    
    try:
        if level.upper() == "ERROR":
            logger.error(msg)
        elif level.upper() == "WARNING":
            logger.warning(msg)
        else:
            logger.info(msg)
    except Exception:
        # Best-effort: if logging fails, silently ignore to avoid crashing experiments
        pass

def load_model_and_sae(layer, device):
    # Cache model name to avoid string recreation
    model_name = "google/gemma-2-2b-it"
    device_str = str(device)

    model: HookedSAETransformer = HookedSAETransformer.from_pretrained(model_name, device=device)
    sae, _, _ = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res-canonical",
        sae_id=f"layer_{layer}/width_16k/canonical",
        device=device_str,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, sae, tokenizer

@t.no_grad()
def generate_with_scaled_recon(
    model,
    sae,
    tokenizer,
    tokens,
    lambda_scale,
    ablation_features,
    control_features,
    start_position,
    end_position,
    max_new_tokens=25,
    stop_token=None,
    ablate_operation_features=True,
):
    # Reduce logging frequency - only log key parameters
    # write_log(f"Generating: scale={lambda_scale}, start={start_position}, end={end_position}, max_tokens={max_new_tokens}")

    cached = {}
    finished = [False] * tokens["input_ids"].shape[0]  # batch size

    # Pre-compute hook names to avoid string operations in inner loop
    hook_base = sae.cfg.hook_name
    hook_input = f"{hook_base}.hook_sae_input"
    hook_acts = f"{hook_base}.hook_sae_acts_post"
    hook_output = f"{hook_base}.hook_sae_output"

    def cache_input_hook(x, hook):
        cached['acts_in'] = x.clone()
        return x

    def ablation_hook(feats, hook):
        mask = t.zeros_like(feats)
        mask[:, start_position:end_position, ablation_features] = 1.0
        # print("ABLATION hook op:", feats * mask)
        # print("indices where ABLATION hook op is not zero: ", (feats * mask).nonzero())
        # print("ABLATION hook op shape:", (feats * mask).shape)
        return feats * mask

    def alive_feats_ablation_hook(feats, hook):
        mask = t.zeros_like(feats)
        mask[:, start_position:end_position, control_features] = 1.0
        op = feats * mask
        # print("CONTROL hook op:", op)
        # print("indices where CONTROL hook op is not zero: ", op.nonzero())
        # print("CONTROL hook op shape:", op.shape)
        return op

    def recon_hook(recon, hook):
        acts_out = cached['acts_in'].clone()
        acts_out[:, start_position:end_position, :] = cached['acts_in'][:, start_position:end_position, :] - lambda_scale * recon[:, start_position:end_position, :]
        # print("start and end positions:", start_position, end_position)
        # print("RECON hook INPUT SHAPE:", acts_out.shape)
        # print("RECON hook op shape:", recon.shape)
        return acts_out

    # Pre-build hooks list once
    hooks = [
        (hook_input, cache_input_hook),
        (hook_acts, ablation_hook if ablate_operation_features else alive_feats_ablation_hook),
        (hook_output, recon_hook),
    ]

    # Attach SAE explicitly
    model.add_sae(sae)

    # Attach hooks explicitly
    for name, fn in hooks:
        model.add_hook(name, fn)

    # OLD: Use model.generate
    # output_tokens = model.generate(
    #     tokens,
    #     max_new_tokens=max_new_tokens,
    #     temperature=0,
        
    # )
    # Use model.generate
    with t.no_grad():
        output_tokens = model.generate(
            tokens.input_ids,
            # attention_mask=tokens.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            freq_penalty=0,
            # num_beams=1,  
            # pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=0,
        )
    model.reset_hooks(including_permanent=True)

    return [model.to_string(seq) for seq in output_tokens]

def calc_correct_answer(problem):
    """
    Calculate the correct answer for a given problem.
    
    Args:
        problem: Problem string (e.g., "2 + 2")
    
    Returns:
        str: Correct answer as a string
    """
    # Use pre-compiled regex pattern
    clean_problem = CLEAN_PROBLEM_PATTERN.sub('', problem)
    
    try:
        # Evaluate the expression safely
        answer = eval(clean_problem)
        return str(answer)
    except Exception as e:
        write_log(f"Error evaluating problem '{problem}': {e}", level="ERROR")
        return "ERROR"

def extract_model_answer(full_output, original_prompt):
    """
    Extract just the model's answer from the full output by removing the original prompt.
    
    Args:
        full_output: The complete output from the model including the prompt
        original_prompt: The original prompt that was fed to the model
    
    Returns:
        str: Just the model's response (answer part)
    """
    if full_output.startswith(original_prompt):
        return full_output[len(original_prompt):].strip()
    else:
        # Fallback: try to find where the answer starts
        return full_output.strip()

def process_batch_compatible(model, sae, tokenizer, batch_prompts, feats_indices, control_feats_indices, run_no_ablation, ablate_topk_features,
                  layer, device, output_dir, batch_idx, max_new_tokens, start=None, end=None, scale=1.0):
    """
    Process batch and save results in a format compatible with notebook functions.
    
    Returns:
        dict: Dictionary containing results for each condition in the expected format
    """
    # Reduce logging verbosity - only log essential info
    write_log(f"Processing batch {batch_idx}: {len(batch_prompts)} prompts, layer={layer}")

    # Tokenize once and reuse
    # tokens = model.to_tokens(batch_prompts, prepend_bos=False).to(device)
    # Tokenize batch
    tokens = tokenizer(
        batch_prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        add_special_tokens=True
    ).to(device)
        
    # write_log(f"Input tokens: {tokens}")
    # write_log(f"tokens keys: {tokens.keys()}")
    if start is None or end is None:
        # start, end = tokens["input_ids"].shape[1] - 1, tokens["input_ids"].shape[1] # Ablate only the last token, in the first generation step
        start, end = -1, None  # THIS LINE IS THE FIX: Ablate only the last token, in every generation step

    # Dictionary to store results in compatible format
    batch_results = {}
    
    if run_no_ablation:
        no_ablation_outputs = generate_with_scaled_recon(model = model, sae = sae, tokenizer = tokenizer, tokens = tokens, lambda_scale = 0,
                                                 ablation_features = feats_indices, control_features = control_feats_indices,
                                                   start_position = start,  end_position = end, max_new_tokens = max_new_tokens, ablate_operation_features = True)

        # Convert to compatible format: List of tuples (prompt, model_answer)
        no_ablation_results = []
        for i, prompt in enumerate(batch_prompts):
            model_answer = extract_model_answer(no_ablation_outputs[i], prompt)
            no_ablation_results.append((prompt, model_answer))
        
        batch_results["no_ablation"] = no_ablation_results

    # Operation-specific ablation
    print("running operation ablation")
    op_abl_outputs = generate_with_scaled_recon(model=model, sae=sae, tokenizer=tokenizer, tokens=tokens, lambda_scale=scale,
                                        ablation_features=feats_indices, control_features=control_feats_indices,
                                        start_position=start, end_position=end, max_new_tokens=max_new_tokens,
                                        ablate_operation_features=True)

    op_abl_results = []
    for i, prompt in enumerate(batch_prompts):
        model_answer = extract_model_answer(op_abl_outputs[i], prompt)
        op_abl_results.append((prompt, model_answer))
    
    batch_results["operation_ablation"] = op_abl_results

    if ablate_topk_features:
        print("running topk ablation")
        topk_abl_outputs = generate_with_scaled_recon(model=model, sae=sae, tokenizer=tokenizer, tokens=tokens, lambda_scale=scale,
                                              ablation_features=feats_indices, control_features=control_feats_indices,
                                              start_position=start, end_position=end, max_new_tokens=max_new_tokens,
                                              ablate_operation_features=False)

        topk_abl_results = []
        for i, prompt in enumerate(batch_prompts):
            model_answer = extract_model_answer(topk_abl_outputs[i], prompt)
            topk_abl_results.append((prompt, model_answer))
        
        batch_results["topk_ablation"] = topk_abl_results
        batch_results["control_features"] = control_feats_indices
        batch_results["ablated_features"] = feats_indices
    return batch_results

def save_batch_results(batch_results, output_dir, layer, num_feats, dataset_name, batch_idx, descending):
    """
    Save batch results in multiple formats:
    1. JSON format for human readability
    2. Original text format for backwards compatibility
    
    Args:
        batch_results: Dictionary with condition -> list of (prompt, answer) tuples for this batch
        output_dir: Output directory
        layer: Layer number
        num_feats: Number of features
        dataset_name: Name of the dataset
        batch_idx: Batch index
    """
    base_filename = f"{dataset_name}_layer-{layer}_batch-{batch_idx}_numfeats-{num_feats}_descending-{descending}"
    
    # Save as JSON (human readable)
    json_path = os.path.join(output_dir, f"{base_filename}.json")
    with open(json_path, "w") as f:
        json.dump(batch_results, f, indent=2)
    write_log(f"Saved JSON results to: {json_path}")
    
    # Save in original text format for backwards compatibility
    txt_path = os.path.join(output_dir, f"{base_filename}.txt")
    # with open(txt_path, "w", encoding="utf-8") as f:
    #     for condition, results in batch_results.items():
    #         f.write(f"=== {condition.upper()} ===\n")
    #         for prompt, answer in results:
    #             f.write(f"PROMPT: {prompt}\n")
    #             f.write(f"ANSWER: {answer}\n")
    #             f.write("--" * 30 + "\n")
    #         f.write("\n" + "=" * 60 + "\n\n")
    # write_log(f"Saved text results to: {txt_path}")
    
    return json_path, txt_path

def generate_correct_answers_compatible(prompts, dataset_name=None):
    """
    Generate correct answers in the format expected by notebook functions.
    
    Args:
        prompts: List of prompts
        dataset_name: Optional dataset name for context
    
    Returns:
        List of tuples (prompt, correct_answer)
    """
    correct_answers = []
    for prompt in prompts:
        # Extract the mathematical problem from the prompt
        clean_prompt = prompt.strip()
        
        # Use pre-compiled regex pattern
        match = MATH_PATTERN.search(clean_prompt)
        
        if match:
            math_problem = match.group(1)
            correct_answer = calc_correct_answer(math_problem)
        else:
            # Fallback: try to evaluate the whole prompt
            correct_answer = calc_correct_answer(clean_prompt)
        
        correct_answers.append((prompt, correct_answer))
    
    return correct_answers


def main():
    start_time = timeit.default_timer()

    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ablation_features", type=str, required=True)
    parser.add_argument("--run_no_ablation", action="store_true")
    parser.add_argument("--ablate_topk_features", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, default="ablation_results")
    parser.add_argument("--max_new_tokens", type=int, default=25)
    parser.add_argument("--start", type=int, default=None, help="Start position for ablation")
    parser.add_argument("--end", type=int, default=None, help="End position for ablation")
    parser.add_argument("--scale", type=float, default=1.0, help="Lambda for scaling")
    parser.add_argument("--num_feats", type=int, default=1, help="Number of features to ablate")
    parser.add_argument("--save_correct_answers", action="store_true", help="Also save correct answers file")
    parser.add_argument("--descending", action="store_true", help="ablate top-k features in descending order")
    args = parser.parse_args()
    

    # Get dataset name for file naming
    dataset_name = os.path.basename(args.dataset).replace('.txt', '')
    
    # Setup logging with proper buffered file I/O
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{dataset_name}_parallel_ablation_layer_{args.layer}.log" if not args.descending else f"{dataset_name}_parallel_ablation_layer_{args.layer}_descending.log")
    setup_logger(log_path)
    
    write_log("Starting ablation experiment...")

    # log arguments used (reduced verbosity):
    write_log(f"Args: layer={args.layer}, batch_size={args.batch_size}, max_tokens={args.max_new_tokens}, num_feats={args.num_feats}, descending={args.descending}")
    os.makedirs(args.output_dir, exist_ok=True)

    # GPU-awareness
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    write_log(f"Using device: {device}")

    model, sae, tokenizer = load_model_and_sae(args.layer, device)
    sae.use_error_term = False
    write_log(f"Loaded model and SAE for layer {args.layer}")
    model_loading_time = timeit.default_timer() - start_time
    write_log(f"Model loading time: {model_loading_time:.2f} seconds")

    # Load and preprocess data once
    task_feats = pd.read_csv(args.ablation_features)

    # Get the four relevant columns
    cols = [
        "median_addition",
        "median_subtraction",
        "median_random_addition",
        "median_random_subtraction"
    ]

    # Calculate sum, mean, std, CV
    sum_vals = task_feats[cols].sum(axis=1)
    mean_vals = task_feats[cols].mean(axis=1)
    std_vals = task_feats[cols].std(axis=1)

    cv = std_vals / mean_vals  # coefficient of variation

    # Calculate control_metric
    task_feats["control_metric"] = sum_vals / (1 + cv)
    
    # Create a column that sums the last 4 columns
    task_feats['total_acts'] = task_feats[cols].sum(axis=1)

    # Identify features to avoid using as controls (those with non-negative metric)
    feats_to_avoid = task_feats.loc[task_feats["metric"] >= 0, "feature_idx"].tolist()

    # Sort the features by their total activation
    total_acts_feats = task_feats.sort_values('control_metric', ascending=False)

    # If descending flag is set, reverse the order of top features
    if args.descending:
        task_feats = task_feats.head(args.num_feats).iloc[::-1].reset_index(drop=True)

    # Load prompts once and preprocess
    with open(args.dataset, "r") as f:
        prompts = [p.strip() + " " for p in f.readlines()]

    # Pre-compute batch count
    total_batches = (len(prompts) + args.batch_size - 1) // args.batch_size

    for num_feats in range(1, args.num_feats + 1, 2):
        write_log("--" * 30)
        write_log(f"Processing with top {num_feats} features...")

        # Top N features based on whatever your original ranking is
        feats_indices = task_feats.feature_idx[:num_feats].tolist()
        sum_activation_values = task_feats.loc[
            task_feats.feature_idx.isin(feats_indices), "total_acts"
        ].sum()

        # --- Control feature selection ---
        target_sum = sum_activation_values
        per_feat_cap = (target_sum / num_feats) * 1.5   # 50% tolerance per feature
        cumulative_cap = target_sum * 1.2              # 20% tolerance on total sum

        control_feats = []
        control_sum = 0

        for _, row in total_acts_feats.iterrows():
            if row.feature_idx in feats_to_avoid:
                continue
            if len(control_feats) >= num_feats:
                break

            # Skip if this control is too large compared to average task feature
            if row.total_acts > per_feat_cap:
                continue

            # Add if it keeps cumulative sum within tolerance
            if control_sum + row.total_acts <= cumulative_cap:
                control_feats.append(row.feature_idx)
                control_sum += row.total_acts

        control_feats_indices = control_feats


        write_log(f"Selected features: op={feats_indices}, control={control_feats_indices}")
        # write the dataframe rows for feats_indices to log
        write_log("Ablation features activations: " + task_feats[task_feats.feature_idx.isin(feats_indices)].to_string()+"\n")
        write_log("Control features activations: " + task_feats[task_feats.feature_idx.isin(control_feats_indices)].to_string()+"\n")

        avg_batch_processing_time = 0

        for i in range(0, len(prompts), args.batch_size):
            batch_idx = i // args.batch_size
            batch_start_time = timeit.default_timer()

            batch = prompts[i:i + args.batch_size]
            batch_results = process_batch_compatible(
                model, sae, tokenizer, batch, feats_indices, control_feats_indices,
                args.run_no_ablation, args.ablate_topk_features,
                args.layer, device, args.output_dir, batch_idx=batch_idx,
                max_new_tokens=args.max_new_tokens, start=args.start, end=args.end, scale=args.scale
            )

            # Save batch results immediately
            json_path, txt_path = save_batch_results(
                batch_results, args.output_dir, args.layer, num_feats, dataset_name, batch_idx, descending=args.descending
            )

            batch_processing_time = timeit.default_timer() - batch_start_time
            avg_batch_processing_time += batch_processing_time
            
            # Only log every 10 batches to reduce I/O
            if (batch_idx + 1) % 10 == 0 or batch_idx + 1 == total_batches:
                write_log(f"Processed batch {batch_idx + 1}/{total_batches} in {batch_processing_time:.2f}s")

        # Optionally save correct answers in compatible format (once per feature count)
        if args.save_correct_answers:
            correct_answers = generate_correct_answers_compatible(prompts, dataset_name)
            correct_answers_path = os.path.join(args.output_dir, f"correct_answers_{dataset_name}_{num_feats}_feats.json")
            with open(correct_answers_path, "w") as f:
                json.dump(correct_answers, f, indent=2)
            write_log(f"Saved correct answers to: {correct_answers_path}")

        avg_batch_processing_time /= total_batches
        write_log(f"Average batch time: {avg_batch_processing_time:.2f}s, Total time: {timeit.default_timer() - start_time:.2f}s")

    # Ensure all logs are flushed and handlers are closed
    if logger:
        for handler in logger.handlers[:]:
            handler.flush()
            handler.close()
            logger.removeHandler(handler)

if __name__ == "__main__":
    main()
