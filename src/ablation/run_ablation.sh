#!/bin/bash
#SBATCH --job-name=addition
#SBATCH --partition=gpu-2h
#SBATCH --array=20
#SBATCH --gpus-per-task=1   # Give each array job 1 GPU
#SBATCH --ntasks=1          # Each job runs a single task
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --constraint="80gb|40gb"
#SBATCH --output=logs/addition_ablation-%A_%a.out

apptainer exec --nv --bind /home/ammar/gemma_scope_math:/workspace torch-gpu.sif python /workspace/src/ablation/parallel_ablation_compatible.py \
--layer $SLURM_ARRAY_TASK_ID \
--dataset "./data/addition.txt" \
--ablation_features "./feature_metrics/layer_${SLURM_ARRAY_TASK_ID}_feature_metrics.csv" \
--batch_size 16 \
--max_new_tokens 25 \
--num_feats 12 \
--output_dir "./ablation_results" \
--descending \
# --ablate_topk 5 \
# --run_no_ablation \
# --save_correct_answers \

# layer_${SLURM_ARRAY_TASK_ID}_feature_metrics.csv layer_${SLURM_ARRAY_TASK_ID}_subtraction_features.csv