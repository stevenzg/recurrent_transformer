#!/bin/bash

# Environment Setup
echo "=== Setting up environment ==="

# Initialize conda for shell interaction
source $(conda info --base)/etc/profile.d/conda.sh

# Check if environment already exists, create only if it doesn't
if conda info --envs | grep -q "^rt "; then
    echo "Conda environment 'rt' already exists, skipping creation."
else
    echo "Creating new conda environment 'rt'."
    conda create -n rt python=3.7 -y
fi

# Activate the conda environment
conda activate rt

# Install required packages (will skip if already installed)
echo "Installing required packages..."
conda install -c anaconda tqdm numpy pandas -y
conda install -c conda-forge matplotlib -y
python3 -m pip install wandb

# Note about wandb login - only needed once
echo "Note: If you need wandb, use 'wandb login' command manually (only needed once)."

# Install PyTorch for CPU only
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y


# Note about pre-downloaded data files
echo "=== Data files ==="
echo "Note: Using pre-downloaded data files."
echo "Skipping download of all data files (palm_i2t_train.csv, features.pt, features_img.pt, labels.pt, perm.pt, train.csv, valid.csv, test.csv)."

echo "=== Environment setup completed ===" 