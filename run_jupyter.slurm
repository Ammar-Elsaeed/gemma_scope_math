#!/bin/bash
#SBATCH --job-name=gpu_jupyter
#SBATCH --partition=gpu-5h
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/jupyter-gpu-%j.out

cd /home/ammar/gemma_scope_math

# Run Jupyter on fixed port 8888, no browser, accessible on all IPs
# Redirect stdout and stderr to tee so it appears in the log file AND console
apptainer run --nv ./torch-gpu.sif jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser 2>&1 | tee -a jupyter.log &
sleep 5  # Give it a few seconds to start

# Wait for the notebook process to finish (keep job alive)
wait
