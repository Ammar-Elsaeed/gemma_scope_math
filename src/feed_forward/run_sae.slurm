#!/bin/bash
#SBATCH --job-name=run_sae
#SBATCH --partition=cpu-2h
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/run_sae-%j.out

apptainer exec --bind /home/ammar/gemma_scope_math:/workspace torch-gpu.sif python /workspace/src/feed_forward/sae_module.py