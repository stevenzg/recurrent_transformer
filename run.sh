#!/bin/bash

# Ensure conda environment is activated
conda activate rt || source ~/anaconda3/etc/profile.d/conda.sh && conda activate rt

# Run Experiments
echo "=== Running experiments (CPU only) ==="

# Run a simple textual Sudoku experiment with CPU
cd sudoku
python main.py --all_layers --n_layer 1 --n_recur 32 --n_head 4 --epochs 10 --eval_interval 1 --lr 0.001 --dataset satnet

# Run a simple nonogram experiment with CPU
cd ../nonogram
python main.py --game_size 7

# Run a simple shortest path experiment with CPU
cd ../shortest_path
python main.py --grid_size 4

echo "=== Experiments completed ===" 