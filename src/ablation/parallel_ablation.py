#!/usr/bin/env python3
import argparse
import os
import torch as t
import pandas as pd
from transformers import AutoTokenizer
from sae_lens import HookedSAETransformer, SAE
from datetime import datetime
import timeit
t.set_grad_enabled(False)

def write_log(msg, level="INFO"):
    """Append a timestamped log line to the log file.

    Args:
        msg (str): Message to write.
        level (str): Log level string, e.g. 'INFO' or 'ERROR'.
    """
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"{ts} {level}: {msg}\n")
    except Exception:
        # Best-effort: if logging fails, silently ignore to avoid crashing experiments
        pass

def load_model_and_sae(layer, device):
    model = HookedSAETransformer.from_pretrained("google/gemma-2-2b-it", device=device)
    sae, _, _ = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res-canonical",
        sae_id=f"layer_{layer}/width_16k/canonical",
        device=str(device),
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    return model, sae, tokenizer

@t.no_grad()
def generate_with_scaled_recon(
    model,
    sae,
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
    # write logs for all arguments passed:
    write_log("Generating with scaled reconstruction...")
    write_log("Tokens: {}".format(tokens))
    write_log("Lambda scale: {}".format(lambda_scale))
    write_log("Ablation features: {}".format(ablation_features))
    write_log("Control features: {}".format(control_features))
    write_log("Start position: {}".format(start_position))
    write_log("End position: {}".format(end_position))
    write_log("Max new tokens: {}".format(max_new_tokens))
    write_log("Stop token: {}".format(stop_token))
    write_log("Ablate operation features: {}".format(ablate_operation_features))

    cached = {}
    finished = [False] * tokens.shape[0]  # batch size

    def cache_input_hook(x, hook):
        cached['acts_in'] = x.clone()
        return x

    def ablation_hook(feats, hook):
        mask = t.zeros_like(feats)
        mask[:, start_position:end_position, ablation_features] = 1.0
        return feats * mask

    def alive_feats_ablation_hook(feats, hook):
        # # Old logic: ablate top 2 most activating tokenss other than ablation_features
        # temp_feats = feats.clone()
        # temp_feats[:, start_position:end_position, ablation_features] = 0
        # top_k = temp_feats[:, start_position:end_position, :].topk(len(ablation_features), dim=-1).indices
        # mask = t.zeros_like(feats)
        # mask.scatter_(-1, top_k, 1.0)
        # return feats * mask
        mask = t.zeros_like(feats)
        mask[:, start_position:end_position, control_features] = 1.0
        return feats * mask

    def recon_hook(recon, hook):
        acts_out = cached['acts_in'].clone()
        acts_out[:, start_position:end_position, :] = cached['acts_in'][:, start_position:end_position, :] - lambda_scale * recon[:, start_position:end_position, :]
        return acts_out

    hp = sae.cfg.hook_name
    hooks = [
        (f"{hp}.hook_sae_input", cache_input_hook),
        (f"{hp}.hook_sae_acts_post", ablation_hook if ablate_operation_features else alive_feats_ablation_hook),
        (f"{hp}.hook_sae_output", recon_hook),
    ]

    write_log("Using hooks: {}".format(hooks))

    for _ in range(max_new_tokens):
        logits = model.run_with_hooks_with_saes(tokens, saes=[sae],
                                                fwd_hooks=hooks,
                                                reset_saes_end=True,
                                                reset_hooks_end=True)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens = t.cat([tokens, next_token], dim=-1)

        if stop_token is not None:
            for i, tok in enumerate(next_token):
                if not finished[i] and model.to_string(tok) == stop_token:
                    finished[i] = True
            if all(finished):
                break

    return [model.to_string(seq) for seq in tokens]

def process_batch(model, sae, batch_prompts, feats_indices, control_feats_indices, run_no_ablation, ablate_topk_features,
                  layer, device, output_dir, batch_idx, max_new_tokens, start=None, end=None, scale=1.0):
    # print all args to log
    write_log(f"Processing batch {batch_idx} with {len(batch_prompts)} prompts")
    write_log(f"Layer: {layer}, Device: {device}, Output Dir: {output_dir}, Max New Tokens: {max_new_tokens}, Start: {start}, End: {end}, Scale: {scale}")
    write_log(f"Selected operation features: {feats_indices}")
    write_log(f"Selected control features: {control_feats_indices}")
    write_log(f"Batch prompts: {batch_prompts}")
    write_log(f"Run no ablation: {run_no_ablation}")
    write_log(f"Ablate top-k features: {ablate_topk_features}")
    write_log(f"Output directory: {output_dir}")

    out_path = os.path.join(output_dir, f"layer{layer}_batch{batch_idx}_num_feats_{len(feats_indices)}.txt")
    if os.path.exists(out_path):
        print(f"Skipping batch {batch_idx}, output already exists: {out_path}")
        return

    # Tokenize once
    tokens = model.to_tokens(batch_prompts, prepend_bos=False).to(device)
    if start is None or end is None:
        start, end = tokens.shape[1] - 1, tokens.shape[1]

    write_log(f"Start: {start}, End: {end}")

    results = []
    if run_no_ablation:
        no_ablation = generate_with_scaled_recon(model=model, sae=sae, tokens=tokens.clone(), lambda_scale=0,
                                                 ablation_features=feats_indices, control_features=control_feats_indices,
                                                 start_position=start, end_position=end, max_new_tokens=max_new_tokens,
                                                 ablate_operation_features=True)
        results.append(("No ablation", no_ablation))

    write_log("Generating operation-specific ablation outputs...")
    op_abl = generate_with_scaled_recon(model=model, sae=sae, tokens=tokens.clone(), lambda_scale=scale,
                                        ablation_features=feats_indices, control_features=control_feats_indices,
                                        start_position=start, end_position=end, max_new_tokens=max_new_tokens,
                                        ablate_operation_features=True)
    results.append(("Ablating operation-specific features", op_abl))

    if ablate_topk_features:
        write_log("Generating top-k ablation outputs...")
        topk_abl = generate_with_scaled_recon(model=model, sae=sae, tokens=tokens.clone(), lambda_scale=scale,
                                              ablation_features=feats_indices, control_features=control_feats_indices,
                                              start_position=start, end_position=end, max_new_tokens=max_new_tokens,
                                              ablate_operation_features=False)
        results.append(("Ablating highest activating features", topk_abl))

    with open(out_path, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(batch_prompts):
            f.write(f"PROMPT: {prompt}\n")
            for label, outs in results:
                f.write(f"{label}: {outs[i]}\n")
            f.write("--" * 30 + "\n")

    print(f"Saved batch {batch_idx} results to {out_path}")

def main():
    start_time = timeit.default_timer()
    write_log("Starting ablation experiment...")

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
    args = parser.parse_args()
    
    # Simple file-based logging: append timestamped lines to ./logs/parallel_ablation.txt
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    global log_path
    log_path = os.path.join(log_dir, "parallel_ablation_layer_{}.txt".format(args.layer))

    # log arguments used:
    write_log("Arguments used:")
    for arg in vars(args):
        write_log(f"  {arg}: {getattr(args, arg)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # GPU-awareness
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    write_log(f"Using device: {device}")

    model, sae, _ = load_model_and_sae(args.layer, device)
    sae.use_error_term = False
    write_log(f"Loaded model and SAE for layer {args.layer}")
    model_loading_time = timeit.default_timer() - start_time
    write_log(f"Model loading time: {model_loading_time:.2f} seconds")


    task_feats = pd.read_csv(args.ablation_features)
    # Create a column that sums the last 4 columns
    task_feats['total_acts'] = task_feats.iloc[:, -4:].sum(axis=1)

    # Sort the features by their total activation
    total_acts_feats = task_feats.sort_values('total_acts', ascending=False)

    with open(args.dataset, "r") as f:
        prompts = [p.strip() + " " for p in f.readlines()]

    for num_feats in range(1, args.num_feats + 1):
        write_log("--" * 30)
        write_log(f"Processing with top {num_feats} features...")
        
        # Top N features based on whatever your original ranking is
        feats_indices = task_feats.feature_idx[:num_feats].tolist()

        # Filter out features already in feats_indices before selecting control features
        control_feats_indices = (
            total_acts_feats[~total_acts_feats.feature_idx.isin(feats_indices)].feature_idx[:num_feats].tolist()
        )

        write_log(f"Selected operation features: {feats_indices}")
        write_log(f"Selected control features: {control_feats_indices}")

        avg_batch_processing_time = 0
        total_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
        
        for i in range(0, len(prompts), args.batch_size):
            write_log("*" * 30)
            batch_idx = i // args.batch_size
            write_log(f"Processing batch {batch_idx + 1}/{total_batches}...")
            batch_start_time = timeit.default_timer()
            batch = prompts[i:i + args.batch_size]
            process_batch(model, sae, batch, feats_indices, control_feats_indices, args.run_no_ablation, args.ablate_topk_features,
                        args.layer, device, args.output_dir, batch_idx=batch_idx, 
                        max_new_tokens=args.max_new_tokens, start=args.start, end=args.end, scale=args.scale)
            batch_processing_time = timeit.default_timer() - batch_start_time
            write_log(f"Processed batch {batch_idx + 1} in {batch_processing_time:.2f} seconds")
            avg_batch_processing_time += batch_processing_time
        avg_batch_processing_time /= total_batches
        write_log(f"Average batch processing time: {avg_batch_processing_time:.2f} seconds")
        write_log(f"Total processing time for {num_feats} features: {timeit.default_timer() - start_time:.2f} seconds")
if __name__ == "__main__":
    main()
