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

# Define variables
NOTES="4_iters_sumloss_fashion"
NUM_ADV=3
DATASET="fashion"

# First set: With true class included
python main.py --num_confused=2 --includes_true=True --num_adversarial_examples=${NUM_ADV} --notes="${NOTES}" --dataset="${DATASET}"
python main.py --num_confused=3 --includes_true=True --num_adversarial_examples=${NUM_ADV} --notes="${NOTES}" --dataset="${DATASET}"
python main.py --num_confused=4 --includes_true=True --num_adversarial_examples=${NUM_ADV} --notes="${NOTES}" --dataset="${DATASET}"
python main.py --num_confused=5 --includes_true=True --num_adversarial_examples=${NUM_ADV} --notes="${NOTES}" --dataset="${DATASET}"
python main.py --num_confused=6 --includes_true=True --num_adversarial_examples=${NUM_ADV} --notes="${NOTES}" --dataset="${DATASET}"
python main.py --num_confused=7 --includes_true=True --num_adversarial_examples=${NUM_ADV} --notes="${NOTES}" --dataset="${DATASET}"
python main.py --num_confused=8 --includes_true=True --num_adversarial_examples=${NUM_ADV} --notes="${NOTES}" --dataset="${DATASET}"
python main.py --num_confused=9 --includes_true=True --num_adversarial_examples=${NUM_ADV} --notes="${NOTES}" --dataset="${DATASET}"
python main.py --num_confused=10 --includes_true=True --num_adversarial_examples=${NUM_ADV} --notes="${NOTES}" --dataset="${DATASET}"

# Second set: Without true class
python main.py --num_confused=1 --includes_true=False --num_adversarial_examples=${NUM_ADV} --notes="${NOTES}" --dataset="${DATASET}"
python main.py --num_confused=2 --includes_true=False --num_adversarial_examples=${NUM_ADV} --notes="${NOTES}" --dataset="${DATASET}"
python main.py --num_confused=3 --includes_true=False --num_adversarial_examples=${NUM_ADV} --notes="${NOTES}" --dataset="${DATASET}"
python main.py --num_confused=4 --includes_true=False --num_adversarial_examples=${NUM_ADV} --notes="${NOTES}" --dataset="${DATASET}"
python main.py --num_confused=5 --includes_true=False --num_adversarial_examples=${NUM_ADV} --notes="${NOTES}" --dataset="${DATASET}"
python main.py --num_confused=6 --includes_true=False --num_adversarial_examples=${NUM_ADV} --notes="${NOTES}" --dataset="${DATASET}"
python main.py --num_confused=7 --includes_true=False --num_adversarial_examples=${NUM_ADV} --notes="${NOTES}" --dataset="${DATASET}"
python main.py --num_confused=8 --includes_true=False --num_adversarial_examples=${NUM_ADV} --notes="${NOTES}" --dataset="${DATASET}"
python main.py --num_confused=9 --includes_true=False --num_adversarial_examples=${NUM_ADV} --notes="${NOTES}" --dataset="${DATASET}"
