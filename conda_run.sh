#!/bin/bash

# Check if conda command is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda command not found. Please ensure Miniconda/Anaconda is installed and added to PATH."
    exit 1
fi

# Check if rt environment exists
if ! conda env list | grep -q "rt "; then
    echo "Error: 'rt' environment not found. Please run ./conda_install.sh first."
    exit 1
fi

# Activate environment
echo "Activating rt environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rt

# Check if PyTorch can be imported
echo "Checking PyTorch..."
python -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Cannot import PyTorch. Please run ./conda_install.sh again."
    exit 1
fi

# Check if NumPy can be imported
echo "Checking NumPy..."
python -c "import numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Cannot import NumPy. Please run ./conda_install.sh again."
    exit 1
fi

# Check data files
SATNET_DATA_DIR="data/satnet"
if [ ! -f "$SATNET_DATA_DIR/features.pt" ] || [ ! -f "$SATNET_DATA_DIR/labels.pt" ] || [ ! -f "$SATNET_DATA_DIR/perm.pt" ]; then
    echo "Error: Required SATNet data files are missing."
    echo "Generating simulation data..."
    python create_dummy_data.py
    if [ $? -ne 0 ]; then
        echo "Failed to generate simulation data. Please check the error message."
        exit 1
    fi
fi

# Display available experiment options
echo -e "\nAvailable experiments:"
echo "1. Basic Sudoku (SATNet dataset)"
echo "2. Sudoku with Constraints"
echo "3. Visual Sudoku (requires features_img.pt)"
echo "4. 16x16 Sudoku"
echo "5. Shortest Path"
echo "6. MNIST Mapping"
echo "7. Nonogram"
echo "8. Debug Mode (only 2 epochs)"
echo "9. Super Debug Mode (1 epoch, smaller model)"

# Get user selection
read -p "Enter experiment number (1-9): " experiment

case $experiment in
    1)
        echo "Running Basic Sudoku experiment..."
        cd sudoku
        python main.py --all_layers --n_layer 1 --n_recur 32 --n_head 4 --epochs 200 --eval_interval 1 --lr 0.001 --dataset satnet --gpu -1
        ;;
    2)
        echo "Running Sudoku with Constraints experiment..."
        cd sudoku
        python main.py --all_layers --n_layer 1 --n_recur 32 --n_head 4 --epochs 200 --eval_interval 1 --lr 0.001 --dataset satnet --gpu -1 --loss c1 att_c1 --hyper 1 0.1
        ;;
    3)
        if [ ! -f "$SATNET_DATA_DIR/features_img.pt" ]; then
            echo "Error: Visual Sudoku requires features_img.pt file, but it was not found."
            echo "Attempting to generate simulation data..."
            python create_dummy_data.py
            if [ ! -f "$SATNET_DATA_DIR/features_img.pt" ]; then
                echo "Failed to generate features_img.pt. Exiting."
                exit 1
            fi
        fi
        echo "Running Visual Sudoku experiment..."
        cd visual_sudoku
        python main.py --all_layers --n_layer 1 --n_recur 32 --n_head 4 --epochs 500 --eval_interval 1 --lr 0.001 --dataset satnet --gpu -1
        ;;
    4)
        echo "Running 16x16 Sudoku experiment..."
        cd sudoku_16
        python main.py --dataset easy --gpu -1
        ;;
    5)
        echo "Running Shortest Path experiment..."
        cd shortest_path
        python main.py --grid_size 4 --gpu -1
        ;;
    6)
        echo "Running MNIST Mapping experiment..."
        cd MNIST_mapping
        python main.py --gpu -1
        ;;
    7)
        echo "Running Nonogram experiment..."
        cd nonogram
        python main.py --game_size 7 --gpu -1
        ;;
    8)
        echo "Running Debug Mode (only 2 epochs)..."
        cd sudoku
        python main.py --all_layers --n_layer 1 --n_recur 32 --n_head 4 --epochs 2 --eval_interval 1 --lr 0.001 --dataset satnet --gpu -1
        ;;
    9)
        echo "Running Super Debug Mode (1 epoch, smaller model)..."
        cd sudoku
        python main.py --all_layers --n_layer 1 --n_recur 4 --n_head 2 --n_embd 64 --epochs 1 --eval_interval 1 --lr 0.001 --dataset satnet --gpu -1
        ;;
    *)
        echo "Invalid selection"
        exit 1
        ;;
esac

echo -e "\nExperiment complete. To exit the conda environment, type 'conda deactivate'" 