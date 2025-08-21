#!/bin/bash

#SBATCH --partition=a100
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --job-name=wm-training
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_output_%j.log

# Ensure the script fails on any error
set -e

# Print job information
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"

# Your commands go here.
# For example, to run a python script:
# python your_script.py --arguments

source ~/miniconda3/bin/activate
conda activate owl

nvidia-smi

export OMP_NUM_THREADS=12 
torchrun --nproc_per_node=2 train.py --config_path configs/tekken_dit_v2.yml