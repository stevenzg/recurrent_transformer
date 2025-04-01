import sys
import os
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb

from mingpt.model import GPT, GPTConfig
from network import inference_trick

def load_model_from_wandb():
    """
    Load the trained model from wandb
    
    Returns:
        model: The loaded GPT model
    """
    # Initialize wandb
    run = wandb.init()
    
    # Use the artifact
    artifact = run.use_artifact('agir/transformer-ste-sudoku/model-satnet_9k_att_c1-c1_0.5-0.5_L1R32H4:v0', type='model')
    artifact_dir = artifact.download()
    
    # Create model structure (same as training)
    mconf = GPTConfig(vocab_size=10, block_size=81, n_layer=1, n_head=4, n_embd=128, 
                    num_classes=9, causal_mask=False, losses=[], n_recur=32, all_layers=True,
                    hyper=[1, 0.1])
    model = GPT(mconf)
    
    # Find model file
    model_files = [f for f in os.listdir(artifact_dir) if f.endswith('.pt')]
    if not model_files:
        raise FileNotFoundError(f"No model file (.pt) found in {artifact_dir}")
    
    model_path = os.path.join(artifact_dir, model_files[0])
    print(f"Using model file: {model_path}")
    
    # Load model weights
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    
    # Set model to evaluation mode
    model.eval()
    
    return model

def solve_sudoku(model, sudoku_input):
    """
    Solve a Sudoku puzzle using the trained model
    
    Args:
        model: The loaded GPT model
        sudoku_input: Sudoku input (9x9 array, 0 for empty cells, 1-9 for filled cells)
    
    Returns:
        solution: Solved Sudoku (9x9 array)
    """
    # Convert input to the correct format
    device = next(model.parameters()).device
    X = torch.tensor(sudoku_input).view(1, 81).to(device)
    
    # Use inference_trick to solve the Sudoku
    with torch.no_grad():
        pred = inference_trick(model, X)
    
    # Convert 0-8 indices to 1-9 numbers
    solution = pred.cpu().numpy()[0].reshape(9, 9) + 1
    
    return solution

def visualize_sudoku(input_sudoku, solved_sudoku):
    """
    Visualize the input Sudoku and its solution
    
    Args:
        input_sudoku: Original Sudoku puzzle (9x9 array)
        solved_sudoku: Solution to the Sudoku puzzle (9x9 array)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Visualize input
    axes[0].set_title("Input Sudoku")
    for i in range(9):
        for j in range(9):
            if input_sudoku[i][j] != 0:
                axes[0].text(j, i, str(input_sudoku[i][j]), ha='center', va='center', 
                             color='black', fontsize=12)
    
    # Draw grid
    for axis in axes:
        axis.set_xlim(-0.5, 8.5)
        axis.set_ylim(8.5, -0.5)  # Flip y-axis to have (0,0) at top-left
        
        # Draw grid lines
        for i in range(10):
            lw = 2 if i % 3 == 0 else 0.5
            axis.axhline(i-0.5, color='black', linewidth=lw)
            axis.axvline(i-0.5, color='black', linewidth=lw)
            
        axis.set_xticks([])
        axis.set_yticks([])
    
    # Visualize solution
    axes[1].set_title("Sudoku Solution")
    for i in range(9):
        for j in range(9):
            color = 'blue' if input_sudoku[i][j] == 0 else 'black'
            axes[1].text(j, i, str(int(solved_sudoku[i][j])), ha='center', va='center', 
                         color=color, fontsize=12)
    
    plt.tight_layout()
    plt.savefig('sudoku_solution.png')
    print(f"Solution saved as 'sudoku_solution.png'")
    plt.show()

def main():
    # Example Sudoku puzzle (0 for empty cells, 1-9 for filled cells)
    # This is a medium difficulty Sudoku puzzle
    sample_sudoku = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]
    
    # Load model
    print("Loading model from wandb...")
    model = load_model_from_wandb()
    
    # Solve Sudoku
    print("Solving Sudoku...")
    solution = solve_sudoku(model, sample_sudoku)
    
    # Print solution
    print("\nSudoku Solution:")
    for row in solution:
        print(" ".join(map(str, [int(x) for x in row])))
    
    # Visualize
    visualize_sudoku(sample_sudoku, solution)

if __name__ == "__main__":
    main() 