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

# Setup wandb login
echo "Setting up wandb..."
read -p "Do you want to log in to wandb? (y/n) [default: n]: " do_wandb_login
do_wandb_login=${do_wandb_login:-n}

if [[ "$do_wandb_login" == "y" ]]; then
    echo "Please enter your wandb API key (or press Enter to open browser login):"
    read -p "API Key: " wandb_api_key
    
    if [[ -z "$wandb_api_key" ]]; then
        # No API key provided, use browser login
        wandb login
    else
        # Use provided API key
        wandb login "$wandb_api_key"
    fi
    
    echo "wandb login completed."
else
    echo "Skipping wandb login. You can manually run 'wandb login' later if needed."
fi

# Install PyTorch for CPU only
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Create necessary directories (mkdir -p will not fail if dir exists)
mkdir -p data/visual_sudoku
mkdir -p data/satnet
mkdir -p data/sudoku-hard

# Check if files already exist before downloading
echo "=== Checking and downloading necessary files ==="

# Check for wget or curl
if command -v wget &> /dev/null; then
    DOWNLOAD_CMD="wget"
    echo "Using wget for downloads"
elif command -v curl &> /dev/null; then
    DOWNLOAD_CMD="curl"
    echo "Using curl for downloads"
else
    echo "Neither wget nor curl is installed. Installing wget with conda..."
    conda install -c conda-forge wget -y
    DOWNLOAD_CMD="wget"
fi

# Function to download file using wget or curl
download_file() {
    URL=$1
    OUTPUT=$2
    
    if [ -f "$OUTPUT" ]; then
        echo "File $OUTPUT already exists, skipping download."
    else
        echo "Downloading $OUTPUT from $URL..."
        if [ "$DOWNLOAD_CMD" = "wget" ]; then
            wget -O "$OUTPUT" "$URL"
        else
            curl -L "$URL" -o "$OUTPUT"
        fi
    fi
}

# Function to download from Google Drive (using either wget or curl)
download_gdrive() {
    FILEID=$1
    FILENAME=$2
    
    if [ -f "$FILENAME" ]; then
        echo "File $FILENAME already exists, skipping download."
        return
    fi
    
    echo "Downloading $FILENAME from Google Drive..."
    
    # First attempt: direct download with curl
    if [ "$DOWNLOAD_CMD" = "curl" ]; then
        curl -L "https://drive.google.com/uc?export=download&id=${FILEID}" -o "$FILENAME"
    else
        # Traditional wget method
        wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=${FILEID}" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O "${FILENAME}" && rm -rf /tmp/cookies.txt
    fi
    
    # Check if downloaded file is valid (not an error page)
    if grep -q "Google Drive - Quota exceeded" "$FILENAME" 2>/dev/null; then
        echo "Error: Google Drive quota exceeded. Please try again later or download manually."
        rm "$FILENAME"
        return 1
    fi
}

# Download palm_i2t_train.csv (Google Drive)
download_gdrive "1SCBkX_c2Xaxjvkx0P481G3-SnUGMZX_L" "data/visual_sudoku/palm_i2t_train.csv"

# Check if SATNet files already exist
if [ -f "data/satnet/features_img.pt" ] && [ -f "data/satnet/features.pt" ] && [ -f "data/satnet/labels.pt" ] && [ -f "data/satnet/perm.pt" ]; then
    echo "SATNet files already exist, skipping download."
else
    echo "Downloading SATNet files..."
    # Create temp directory for SATNet files
    mkdir -p temp_satnet
    cd temp_satnet

    # Clone SATNet repository to get dataset
    git clone https://github.com/locuslab/SATNet.git
    
    # Check the actual file paths in the repository
    echo "Locating data files in SATNet repository..."
    DATADIR=""
    
    # Check possible path structures
    if [ -d "SATNet/data/sudoku" ]; then
        DATADIR="SATNet/data/sudoku"
    elif [ -d "SATNet/sudoku" ]; then
        DATADIR="SATNet/sudoku"
    else
        # Find the data directory recursively
        DATADIR=$(find SATNet -name "*.pt" -type f | head -n 1 | xargs dirname 2>/dev/null)
    fi
    
    if [ -z "$DATADIR" ]; then
        echo "Could not find data files in SATNet repository. Check repository structure."
        cd ..
        rm -rf temp_satnet
        
        # Alternative: directly download from author's URLs
        echo "Attempting to download SATNet files directly..."
        mkdir -p temp_direct
        cd temp_direct
        
        # Direct download links (adjust if you have direct links)
        for FILE in features_img.pt features.pt labels.pt perm.pt; do
            if [ "$DOWNLOAD_CMD" = "wget" ]; then
                wget -O "../data/satnet/$FILE" "https://github.com/locuslab/SATNet/raw/master/data/sudoku/$FILE" || echo "Failed to download $FILE"
            else
                curl -L "https://github.com/locuslab/SATNet/raw/master/data/sudoku/$FILE" -o "../data/satnet/$FILE" || echo "Failed to download $FILE"
            fi
        done
        
        cd ..
        rm -rf temp_direct
    else
        echo "Found data in: $DATADIR"
        # Copy files with error checking
        for FILE in features_img.pt features.pt labels.pt perm.pt; do
            if [ -f "$DATADIR/$FILE" ]; then
                cp "$DATADIR/$FILE" "../data/satnet/"
                echo "Copied $FILE successfully"
            else
                echo "Warning: $FILE not found in $DATADIR"
            fi
        done
        
        cd ..
        rm -rf temp_satnet
    fi
fi

# Check if RRN dataset files already exist
if [ -f "data/sudoku-hard/train.csv" ] && [ -f "data/sudoku-hard/valid.csv" ] && [ -f "data/sudoku-hard/test.csv" ]; then
    echo "RRN dataset files already exist, skipping download."
else
    echo "Downloading RRN dataset..."
    # Download RRN dataset (Dropbox)
    download_file "https://www.dropbox.com/s/rp3hbjs91xiqdgc/sudoku-hard.zip?dl=1" "sudoku-hard.zip"
    
    if [ -f "sudoku-hard.zip" ]; then
        echo "Extracting sudoku-hard.zip..."
        mkdir -p temp_sudoku
        unzip -q sudoku-hard.zip -d temp_sudoku
        
        # Check if files were extracted properly
        if [ -f "temp_sudoku/train.csv" ]; then
            mv temp_sudoku/train.csv data/sudoku-hard/
            mv temp_sudoku/valid.csv data/sudoku-hard/
            mv temp_sudoku/test.csv data/sudoku-hard/
            echo "RRN dataset files moved successfully"
        else
            # Look for csv files in subdirectories
            find temp_sudoku -name "*.csv" -type f | while read file; do
                filename=$(basename "$file")
                case "$filename" in
                    train.csv|valid.csv|test.csv)
                        cp "$file" "data/sudoku-hard/"
                        echo "Found and copied $filename"
                        ;;
                esac
            done
        fi
        
        rm -rf temp_sudoku sudoku-hard.zip
    else
        echo "Failed to download sudoku-hard.zip"
    fi
fi

echo "=== Environment setup completed ===" 