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
