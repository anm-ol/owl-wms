#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=50G
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_output_%j.log
#SBATCH --error=logs/slurm_error_%j.log

# Print SLURM environment variables for debugging
echo "=== SLURM Environment ==="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_LOCALID: $SLURM_LOCALID"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "========================="

source /home/venky/ankitd/miniconda3/bin/activate
conda activate owl
export OMP_NUM_THREADS=24
export WANDB_USER_NAME=d-fusion
# Change to the working directory
cd /mnt/venky/ankitd/anmol/WM/owl-vaes
wandb offline

python -m train --config_path configs/t3/t3_gan.yml
# Run the pipeline - each task will process a subset of videos

