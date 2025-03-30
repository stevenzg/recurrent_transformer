# Local Environment Setup Guide

This guide helps you set up and run the Recurrent Transformer project in your local environment.

## Problem Resolution

The original project may have the following issues:

1. **Dependency Version Conflicts**: Modern Python versions (such as 3.12) are incompatible with some dependencies
2. **Missing Data Files**: External datasets need to be downloaded to run experiments
3. **Complex Environment Setup**: Virtual environment needs to be configured correctly

Our scripts have resolved these issues.

## Recommended Method: Using Conda Environment (Python 3.7)

Using a Conda environment is recommended as it ensures the correct Python version and dependency compatibility.

### 1. Install Conda Environment

Ensure Miniconda or Anaconda is installed, then run:

```bash
./conda_install.sh
```

This script will:
- Create a Python 3.7 Conda environment named `rt`
- Install all necessary dependencies (PyTorch, NumPy, Pandas, etc.)
- Generate simulated Sudoku data for testing
- Download the RRN dataset (hard Sudoku)

### 2. Run Experiments

After installation, run the following command to select and run experiments:

```bash
./conda_run.sh
```

This will display a menu of available experiments:
1. Basic Sudoku (SATNet dataset)
2. Sudoku with Constraints
3. Visual Sudoku
4. 16x16 Sudoku
5. Shortest Path
6. MNIST Mapping
7. Nonogram
8. Debug Mode (only 2 epochs)
9. Super Debug Mode (1 epoch, smaller model)

It's recommended to first select option 9 (Super Debug Mode) for a quick test to confirm the environment is set up correctly.

## Common Issues and Solutions

### Python Version Issues

- This project works best with Python 3.7, which is guaranteed when using the Conda environment
- If you use Python 3.12 or higher, you may encounter various compatibility issues

### Data File Issues

If automatic download or generation fails, you can manually download files from:
- SATNet dataset: https://github.com/locuslab/SATNet#getting-the-datasets
- Hard Sudoku dataset: https://www.dropbox.com/s/rp3hbjs91xiqdgc/sudoku-hard.zip?dl=1

And place them in the correct directories (data/satnet/, data/sudoku-hard/).

### NumPy/PyTorch Compatibility

- The Conda environment uses compatible NumPy and PyTorch version combinations
- For non-Conda environments, we have fixed NumPy to a compatible version

### GPU Issues

By default, we run experiments on CPU (using the `--gpu -1` parameter). If you have a GPU and want to use it:
- Make sure the correct CUDA version is installed
- Change `--gpu -1` to `--gpu 0` (or the appropriate GPU number) in the command

## Directory Structure

Main components of the project:

- `mingpt/`: Modified minGPT implementation that supports recursion
- `sudoku/`, `sudoku_16/`, `visual_sudoku/`: Different types of Sudoku tasks
- `shortest_path/`, `nonogram/`, `MNIST_mapping/`: Other constraint satisfaction problems
- `data/`: Dataset directory
- `ste.py`: Straight-Through Estimator implementation

## Parameter Descriptions

Some common command-line parameters:

- `--n_layer`: Number of Transformer layers (default: 1)
- `--n_recur`: Number of recursions (default: 32)
- `--n_head`: Number of attention heads (default: 4)
- `--epochs`: Training epochs (default: 200)
- `--lr`: Learning rate (default: 0.001)
- `--loss`: Constraint loss types (e.g., c1, att_c1)
- `--gpu`: GPU index (-1 means CPU) 