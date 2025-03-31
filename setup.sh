#!/bin/bash

# Environment Setup
echo "=== Setting up environment ==="

# Create and activate conda environment
conda create -n rt python=3.7 -y
conda activate rt || source ~/anaconda3/etc/profile.d/conda.sh && conda activate rt

# Install required packages
conda install -c anaconda tqdm numpy pandas -y
conda install -c conda-forge matplotlib -y
python3 -m pip install wandb
# Install PyTorch for CPU only
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Create necessary directories
mkdir -p data/visual_sudoku
mkdir -p data/satnet
mkdir -p data/sudoku-hard

# Download necessary files from Google Drive and Dropbox
echo "=== Downloading necessary files ==="

# Function to download from Google Drive
download_gdrive() {
    FILEID=$1
    FILENAME=$2
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=${FILEID}" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O ${FILENAME} && rm -rf /tmp/cookies.txt
}

# Download palm_i2t_train.csv (Google Drive)
download_gdrive "1SCBkX_c2Xaxjvkx0P481G3-SnUGMZX_L" "data/visual_sudoku/palm_i2t_train.csv"

# Create temp directory for SATNet files
mkdir -p temp_satnet
cd temp_satnet

# Clone SATNet repository to get dataset
git clone https://github.com/locuslab/SATNet.git
mv SATNet/data/sudoku/features_img.pt ../data/satnet/
mv SATNet/data/sudoku/features.pt ../data/satnet/
mv SATNet/data/sudoku/labels.pt ../data/satnet/
mv SATNet/data/sudoku/perm.pt ../data/satnet/
cd ..
rm -rf temp_satnet

# Download RRN dataset (Dropbox)
wget -O sudoku-hard.zip "https://www.dropbox.com/s/rp3hbjs91xiqdgc/sudoku-hard.zip?dl=1"
unzip sudoku-hard.zip -d temp_sudoku
mv temp_sudoku/train.csv data/sudoku-hard/
mv temp_sudoku/valid.csv data/sudoku-hard/
mv temp_sudoku/test.csv data/sudoku-hard/
rm -rf temp_sudoku sudoku-hard.zip

echo "=== Environment setup completed ===" 