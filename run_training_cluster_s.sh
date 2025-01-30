#!/bin/bash
#SBATCH --job-name=train_fastpitch
#SBATCH --account=nn9866k
#SBATCH --time=11:50:00
#SBATCH --mem=16G
#SBATCH --partition=accel
#SBATCH --gres=gpu:1

# == Logging

#SBATCH --error="log_err" # Save the error messages
#SBATCH --output="log_out" # Save the stdout

## Set up job environment:
# set -o errexit  # Exit the script on any error
# set -o nounset  # Treat any unset variables as an error

## Activate environment
# source ~/.bashrc

eval "$(conda shell.bash hook)"
conda activate fastpitch

# Setup monitoring
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory \
        --format=csv --loop=1 > "gpu_util-$SLURM_JOB_ID.csv" &
NVIDIA_MONITOR_PID=$!  # Capture PID of monitoring process

# Run our computation
bash scripts/train_2.sh

# After computation stop monitoring
kill -SIGINT "$NVIDIA_MONITOR_PID"
