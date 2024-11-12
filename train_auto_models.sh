#!/bin/bash
#SBATCH -N 1                   # Number of nodes
#SBATCH -n 1                   # Number of tasks
#SBATCH --mem=16G              # Memory per node
#SBATCH -t 5:00:00             # Time required
#SBATCH -p short               # Partition
#SBATCH -J adversarial_job     # Job name
#SBATCH -o adversarial_out.txt # Standard output
#SBATCH -e adversarial_err.txt # Standard error
# SLURM --gres=gpu:A100:1      # Request 1 A100 GPU

# Load Python environment
module load python/3.8.13/slu6jvw

# activate environment
source /home/okcava/projects/autoadvex/.venv/bin/activate

python train_auto.py --num_iters=4 --sum_losses=False
