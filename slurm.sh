#!/bin/bash
#SBATCH -N 1                   # Number of nodes
#SBATCH -n 1                   # Number of tasks
#SBATCH --mem=16G              # Memory per node
#SBATCH -t 24:00:00             # Time required
#SBATCH -p short               # Partition
#SBATCH -J adversarial_job     # Job name
#SBATCH -o adversarial_out.txt # Standard output
#SBATCH -e adversarial_err.txt # Standard error
# SLURM --gres=gpu:A100:1      # Request 1 A100 GPU

# Load Python environment
module load python/3.8.13/slu6jvw

# activate environment
source /home/okcava/projects/autoadvex/.venv/bin/activate

# Define constant variables
MLP_PATH="models/mlp.pth"
NUM_ADV=10
DATASET="digit"

# Loop through each model number (1-6)
for model_num in {1..6}; do
    echo "Processing model ${model_num}"

    # Set model-specific variables
    NOTES="${model_num}_True"
    ENCODER_PATH="models/encoder_${model_num}_True_digit.pth"
    DECODER_PATH="models/decoder_${model_num}_True_digit.pth"

    # Loop through each digit (0-9)
    for digit in {0..9}; do
        echo "Processing digit ${digit} with model ${model_num}"

        # First set: With true class included
        for num_confused in {2..10}; do
            echo "Running with true class, model ${model_num}, digit ${digit}, num_confused ${num_confused}"
            python main.py \
                --encoder_path=${ENCODER_PATH} \
                --decoder_path=${DECODER_PATH} \
                --mlp_path=${MLP_PATH} \
                --num_confused=${num_confused} \
                --includes_true=True \
                --num_adversarial_examples=${NUM_ADV} \
                --notes="${NOTES}" \
                --dataset="${DATASET}" \
                --digit_number=${digit}
        done

        # Second set: Without true class
        for num_confused in {1..9}; do
            echo "Running without true class, model ${model_num}, digit ${digit}, num_confused ${num_confused}"
            python main.py \
                --encoder_path=${ENCODER_PATH} \
                --decoder_path=${DECODER_PATH} \
                --mlp_path=${MLP_PATH} \
                --num_confused=${num_confused} \
                --includes_true=False \
                --num_adversarial_examples=${NUM_ADV} \
                --notes="${NOTES}" \
                --dataset="${DATASET}" \
                --digit_number=${digit}
        done
    done
done
