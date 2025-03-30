#!/bin/bash

# Check if conda command is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda command not found. Please ensure Miniconda/Anaconda is installed and added to PATH."
    exit 1
fi

# Check if rt environment exists, if so ask whether to delete it
if conda env list | grep -q "rt "; then
    echo "Found existing conda environment named 'rt'."
    read -p "Delete and recreate it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing rt environment..."
        conda env remove -n rt
    else
        echo "Will use existing rt environment. Please ensure it's based on Python 3.7."
    fi
fi

# Create new environment (if it doesn't exist or user chose to delete)
if ! conda env list | grep -q "rt "; then
    echo "Creating new conda environment 'rt' with Python 3.7..."
    conda create -n rt python=3.7 -y
fi

# Activate environment and install dependencies
echo "Installing dependencies..."
conda activate rt || source $(conda info --base)/etc/profile.d/conda.sh && conda activate rt

# Basic dependencies
echo "Installing basic dependencies..."
conda install -y -c anaconda tqdm numpy pandas
conda install -y -c conda-forge matplotlib

# PyTorch (CPU version)
echo "Installing PyTorch (CPU version)..."
conda install -y pytorch torchvision torchaudio cpuonly -c pytorch

# Other dependencies
echo "Installing other Python packages..."
pip install wandb jupyter scikit-learn pillow

# Create data directories
echo "Creating data directories..."
mkdir -p data/satnet data/sudoku-hard data/visual_sudoku

# Download SATNet dataset
echo "Preparing to download SATNet dataset..."
cd data/satnet

# Generate simulation data
echo "Generating simulated Sudoku data..."
cat > ../../create_dummy_data.py << 'EOL'
import os
import torch
import numpy as np

def create_dummy_data():
    """Create simulated data for Sudoku training and testing"""
    print("Creating dummy SATNet data in data/satnet...")
    
    # Ensure directory exists
    os.makedirs("data/satnet", exist_ok=True)
    
    # Define file paths
    features_path = "data/satnet/features.pt"
    labels_path = "data/satnet/labels.pt"
    perm_path = "data/satnet/perm.pt"
    features_img_path = "data/satnet/features_img.pt"
    
    # Generate example data
    num_samples = 10000
    board_size = 9
    
    # Features: [num_samples, 81, 10] - one-hot encoding for each position (0-9, 0 means blank)
    features = torch.zeros(num_samples, board_size*board_size, 10)
    # Labels: [num_samples, 81, 10] - one-hot encoding of the correct solution
    labels = torch.zeros(num_samples, board_size*board_size, 10)
    # Cell permutation
    perm = torch.arange(board_size*board_size)
    # Image features: [num_samples, 81, 28, 28] - 28x28 MNIST-style image for each digit
    features_img = torch.zeros(num_samples, board_size*board_size, 28, 28)
    
    # Fill in some random initial digits for each sample
    for i in range(num_samples):
        # Create a solution (1-9 randomly arranged in each row, column and 3x3 block)
        solution = torch.zeros(board_size, board_size, dtype=torch.long)
        for j in range(board_size):
            solution[j] = torch.randperm(board_size) + 1
        
        # Flatten the solution
        solution_flat = solution.reshape(-1)
        
        # Convert solution to one-hot labels
        for j in range(len(solution_flat)):
            val = solution_flat[j]
            labels[i, j, val] = 1
        
        # Randomly select 20-30 cells to show as initial values
        num_givens = np.random.randint(20, 31)
        given_cells = np.random.choice(board_size*board_size, num_givens, replace=False)
        
        # Fill in the features matrix
        for j in given_cells:
            val = solution_flat[j]
            features[i, j, val] = 1
        
        # Set empty cells to blank (index 0)
        for j in range(board_size*board_size):
            if j not in given_cells:
                features[i, j, 0] = 1
        
        # Create simple digit representations for image features
        for j in range(board_size*board_size):
            if j in given_cells:
                val = solution_flat[j]
                # Create a simple digit image (center lit up)
                img = torch.zeros(28, 28)
                img[10:18, 10:18] = val / 9.0  # Simplified digit representation
                features_img[i, j] = img
    
    # Save the data
    print(f"Saving features.pt: shape={features.shape}")
    torch.save(features, features_path)
    
    print(f"Saving labels.pt: shape={labels.shape}")
    torch.save(labels, labels_path)
    
    print(f"Saving perm.pt: shape={perm.shape}")
    torch.save(perm, perm_path)
    
    print(f"Saving features_img.pt: shape={features_img.shape}")
    torch.save(features_img, features_img_path)
    
    print("Dummy data creation complete!")

if __name__ == "__main__":
    create_dummy_data()
EOL

cd ../..
conda run -n rt python create_dummy_data.py

# Download RRN dataset
echo "Downloading RRN (Sudoku-hard) dataset..."
cd data
curl -L "https://www.dropbox.com/s/rp3hbjs91xiqdgc/sudoku-hard.zip?dl=1" -o sudoku-hard.zip
unzip -q -o sudoku-hard.zip -d sudoku-hard
rm sudoku-hard.zip
cd ..

echo "========================================================================"
echo "Installation complete! You can now run experiments with these commands:"
echo ""
echo "1. Activate conda environment:"
echo "   conda activate rt"
echo ""
echo "2. Enter an experiment directory and run:"
echo "   cd sudoku"
echo "   python main.py --all_layers --n_layer 1 --n_recur 4 --n_head 2 --n_embd 64 --epochs 1 --eval_interval 1 --lr 0.001 --dataset satnet --gpu -1"
echo ""
echo "Or use our quick run script:"
echo "   ./conda_run.sh"
echo "========================================================================" 