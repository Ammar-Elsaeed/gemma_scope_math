#!/bin/bash
#SBATCH --job-name=run_feedforward
#SBATCH --partition=gpu-2h
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/run_feedforward-%j.out

apptainer exec --nv --bind /home/ammar/gemma_scope_math:/workspace torch-gpu.sif python /workspace/src/feed_forward/feedforward_module.py