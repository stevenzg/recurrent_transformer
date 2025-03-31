#!/bin/bash

# Initialize conda for shell interaction
source $(conda info --base)/etc/profile.d/conda.sh

# Activate the conda environment
echo "Activating conda environment 'rt'..."
conda activate rt

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "Failed to activate the 'rt' environment. Please run setup.sh first."
    exit 1
fi

# Run Experiments
echo "=== Running experiments (CPU only) ==="

# Run a simple textual Sudoku experiment with CPU
echo "Running Sudoku experiment..."
cd sudoku
python main.py --all_layers --n_layer 1 --n_recur 32 --n_head 4 --epochs 10 --eval_interval 1 --lr 0.001 --dataset satnet

# Run a simple nonogram experiment with CPU
echo "Running Nonogram experiment..."
cd ../nonogram
python main.py --game_size 7

# Run a simple shortest path experiment with CPU
echo "Running Shortest Path experiment..."
cd ../shortest_path
python main.py --grid_size 4

echo "=== Experiments completed ===" 