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
    artifact = run.use_artifact('agir/transformer-ste-sudoku/model-satnet_9k_att_c1-c1_0.5-0.5_L4R16H4:v7', type='model')
    artifact_dir = artifact.download()
    
    # Create model structure (same as training)
    mconf = GPTConfig(vocab_size=10, block_size=81, n_layer=4, n_head=4, n_embd=128, 
                    num_classes=9, causal_mask=False, losses=[], n_recur=16, all_layers=True,
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

def solve_sudoku_with_process(model, sudoku_input):
    """
    Solve a Sudoku puzzle using the trained model and show the solving process
    
    Args:
        model: The loaded GPT model
        sudoku_input: Sudoku input (9x9 array, 0 for empty cells, 1-9 for filled cells)
    
    Returns:
        solution: Solved Sudoku (9x9 array)
        probabilities: Probabilities for each cell (9x9x9 array)
    """
    # Convert input to the correct format
    device = next(model.parameters()).device
    X = torch.tensor(sudoku_input).view(1, 81).to(device)
    
    # Forward pass through the model to get logits
    with torch.no_grad():
        logits, _, _ = model(X)
    
    # Get probabilities using softmax
    probs = torch.nn.functional.softmax(logits, dim=-1)
    probs = probs.cpu().numpy()[0].reshape(9, 9, 9)  # shape: (9, 9, 9) for row, col, digit
    
    # Use inference_trick to solve the Sudoku (same as before)
    with torch.no_grad():
        pred = inference_trick(model, X)
    
    # Convert 0-8 indices to 1-9 numbers
    solution = pred.cpu().numpy()[0].reshape(9, 9) + 1
    
    # Print solving process
    print("\n=== Solving Process ===")
    empty_cells = []
    for r in range(9):
        for c in range(9):
            if sudoku_input[r][c] == 0:
                empty_cells.append((r, c))
    
    print(f"Found {len(empty_cells)} empty cells to solve")
    
    for r, c in empty_cells:
        print(f"\nPosition: Row {r+1}, Column {c+1}")
        print("Probabilities for each digit (1-9):")
        
        # Find the digit with highest probability
        max_prob_digit = np.argmax(probs[r, c]) + 1
        
        for digit in range(9):
            prob_percent = probs[r, c, digit] * 100
            bar_length = int(prob_percent / 5)  # Scale for display
            bar = '█' * bar_length + '░' * (20 - bar_length)
            confidence = "HIGH" if prob_percent > 90 else "MEDIUM" if prob_percent > 50 else "LOW"
            
            # Mark the highest probability digit
            highlight = "← HIGHEST" if (digit + 1) == max_prob_digit else ""
            print(f"  {digit+1}: {bar} {prob_percent:.2f}% ({confidence}) {highlight}")
        
        # Show the model's final choice
        chosen_digit = int(solution[r, c])
        print(f"Single-cell prediction: {max_prob_digit}")
        print(f"Final model choice after global inference: {chosen_digit}")
        
        # Explain if there's a discrepancy
        if max_prob_digit != chosen_digit:
            print("Note: The final choice differs from the highest probability because inference_trick")
            print("      considers global sudoku constraints across the entire puzzle, not just")
            print("      individual cell probabilities.")
    
    return solution, probs

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

def visualize_cell_probabilities(input_sudoku, probs, row, col):
    """
    Visualize probabilities for a specific cell
    
    Args:
        input_sudoku: Original Sudoku puzzle
        probs: Probabilities from the model (9x9x9)
        row, col: Cell position to visualize
    """
    plt.figure(figsize=(8, 5))
    
    # Plot probabilities as a bar chart
    digits = range(1, 10)
    cell_probs = probs[row, col] * 100
    
    bars = plt.bar(digits, cell_probs)
    
    # Color the highest probability bar differently
    max_idx = np.argmax(cell_probs)
    bars[max_idx].set_color('red')
    
    plt.title(f'Probabilities for Row {row+1}, Column {col+1}')
    plt.xlabel('Digit')
    plt.ylabel('Probability (%)')
    plt.xticks(digits)
    plt.ylim(0, 100)
    
    # Add probability values on top of bars
    for i, v in enumerate(cell_probs):
        plt.text(i+1, v+1, f'{v:.1f}%', ha='center')
    
    plt.tight_layout()
    plt.savefig(f'cell_probs_r{row+1}c{col+1}.png')
    print(f"Cell probabilities saved as 'cell_probs_r{row+1}c{col+1}.png'")

def main():
    # Example Sudoku puzzle (0 for empty cells, 1-9 for filled cells)
    # This is a medium difficulty Sudoku puzzle
    medium_sudoku = [
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
    
    # Simpler Sudoku with fewer empty cells
    simple_sudoku = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 0, 0]  # Simplified last row with more empty cells
    ]
    
    # Even simpler Sudoku with most cells filled
    very_simple_sudoku = [
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]  # Only last row has empty cells
    ]
    
    # Choose which puzzle to use (uncomment the one you want)
    # sample_sudoku = medium_sudoku
    # sample_sudoku = simple_sudoku
    sample_sudoku = very_simple_sudoku
    
    # Load model
    print("Loading model from wandb...")
    model = load_model_from_wandb()
    
    # Solve Sudoku with detailed proc
    # ess
    print("Solving Sudoku...")
    solution, probs = solve_sudoku_with_process(model, sample_sudoku)

    
    # Print solution
    print("\nFinal Sudoku Solution:")
    for row in solution:
        print(" ".join(map(str, [int(x) for x in row])))
    
    # Visualize the full solution
    visualize_sudoku(sample_sudoku, solution)
    
    # Visualize probabilities for a few selected cells
    # Find a few interesting empty cells (with different confidence levels)
    empty_cells = []
    for r in range(9):
        for c in range(9):
            if sample_sudoku[r][c] == 0:
                empty_cells.append((r, c, np.max(probs[r, c])))
    
    # Sort by confidence and pick low, medium and high confidence examples
    empty_cells.sort(key=lambda x: x[2])
    
    if empty_cells:
        # Low confidence example
        r, c, _ = empty_cells[0]
        visualize_cell_probabilities(sample_sudoku, probs, r, c)
        
        # Medium confidence example (if available)
        if len(empty_cells) > 10:
            idx = len(empty_cells) // 2
            r, c, _ = empty_cells[idx]
            visualize_cell_probabilities(sample_sudoku, probs, r, c)
        
        # High confidence example
        r, c, _ = empty_cells[-1]
        visualize_cell_probabilities(sample_sudoku, probs, r, c)

if __name__ == "__main__":
    main() 