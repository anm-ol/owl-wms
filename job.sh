#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=50G
#SBATCH --partition=a100
#SBATCH --gres=gpu:A100:2
#SBATCH --time=24:00:00
#SBATCH --output=slurm_output_%j.log
#SBATCH --error=slurm_error_%j.log

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
export OMP_NUM_THREADS=12
# Change to the working directory
cd /owl-wms/owl-vaes
torchrun --nproc_per_node=2 -m train --config_path configs/t3/tekken_H200_optimized.yml
# Run the pipeline - each task will process a subset of videos

