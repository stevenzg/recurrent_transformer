#!/bin/bash

# Initialize conda for shell interaction
source $(conda info --base)/etc/profile.d/conda.sh

# Activate the conda environment
echo "Activating conda environment 'rt'..."
conda activate rt

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "Failed to activate the 'rt' environment. Please run setup.sh first."
    exit 1
fi

# Display available experiments
echo "=== Available Experiments ==="
echo "1) Textual Sudoku (9x9)"
echo "   - Dataset options: satnet, palm"
echo "   - Constraint loss options: none, c1 (sudoku constraints), att_c1 (attention constraints)"
echo ""
echo "2) Visual Sudoku (9x9)"
echo "   - Dataset options: satnet, palm"
echo "   - Constraint loss options: none, c1 (sudoku constraints), att_c1 (attention constraints)"
echo ""
echo "3) Sudoku 16x16"
echo "   - Dataset options: easy, medium"
echo ""
echo "4) Shortest Path"
echo "   - Grid size options: 4, 12"
echo "   - Loss options: none, path"
echo ""
echo "5) Nonogram"
echo "   - Game size options: 7, 15"
echo ""
echo "6) MNIST Mapping"
echo ""
echo "0) Quit"
echo ""

# Ask for global wandb option
read -p "Enable wandb logging for this run? (y/n) [default: n]: " use_wandb
use_wandb=${use_wandb:-n}

wandb_arg=""
if [[ "$use_wandb" == "y" ]]; then
    wandb_arg="--wandb"
    echo "Wandb logging enabled."
fi

# Ask for user choice
read -p "Enter your choice (0-6): " choice

case $choice in
    1)
        echo "=== Running Textual Sudoku Experiment ==="
        
        # Check if previous run parameters exist
        param_file="./sudoku/last_run_params.txt"
        continue_previous="n"
        wandb_path=""
        
        if [ -f "$param_file" ]; then
            echo "Found previous run parameters."
            cat "$param_file"
            read -p "Continue from previous run? (y/n) [default: n]: " continue_previous
            continue_previous=${continue_previous:-n}
            
            if [[ "$continue_previous" == "y" ]]; then
                # Source the parameters from the file
                source "$param_file"
                
                # Ask for wandb artifact path
                read -p "Enter wandb artifact path (e.g. 'agir/transformer-ste-sudoku/model-satnet_9k_att_c1-c1_0.5-0.5_L1R32H4:v1'): " wandb_path
                
                # Create continue training python command
                cd sudoku
                echo "Running: python main.py --all_layers --n_layer $n_layer --n_recur $n_recur --n_head $n_head --epochs $epochs --eval_interval $eval_interval --lr $lr --dataset $dataset $loss_args $wandb_arg --continue_from \"$wandb_path\""
                python main.py --all_layers --n_layer $n_layer --n_recur $n_recur --n_head $n_head --epochs $epochs --eval_interval $eval_interval --lr $lr --dataset $dataset $loss_args $wandb_arg --continue_from "$wandb_path"
                exit 0
            fi
        fi
        
        # If not continuing, proceed with new run
        read -p "Enter dataset (satnet/palm) [default: satnet]: " dataset
        dataset=${dataset:-satnet}
        
        read -p "Use constraint losses? (y/n) [default: n]: " use_constraints
        use_constraints=${use_constraints:-n}
        
        loss_args=""
        if [[ "$use_constraints" == "y" ]]; then
            read -p "Use sudoku constraints (c1)? (y/n) [default: y]: " use_c1
            use_c1=${use_c1:-y}
            
            read -p "Use attention constraints (att_c1)? (y/n) [default: n]: " use_att_c1
            use_att_c1=${use_att_c1:-n}
            
            if [[ "$use_c1" == "y" && "$use_att_c1" == "y" ]]; then
                loss_args="--loss att_c1 c1 --hyper 0.5 0.5"
            elif [[ "$use_c1" == "y" ]]; then
                loss_args="--loss c1 --hyper 1"
            elif [[ "$use_att_c1" == "y" ]]; then
                loss_args="--loss att_c1 --hyper 0.5"
            fi
        fi
        
        read -p "Number of epochs [default: 10]: " epochs
        epochs=${epochs:-10}
        
        # Save parameters for future runs
        mkdir -p ./sudoku
        n_layer=1
        n_recur=32
        n_head=4
        eval_interval=1
        lr=0.001
        
        # Save parameters to file
        cat > "$param_file" << EOF
# Last run parameters ($(date))
dataset=$dataset
epochs=$epochs
n_layer=$n_layer
n_recur=$n_recur
n_head=$n_head
eval_interval=$eval_interval
lr=$lr
loss_args="$loss_args"
EOF
        
        cd sudoku
        echo "Running: python main.py --all_layers --n_layer $n_layer --n_recur $n_recur --n_head $n_head --epochs $epochs --eval_interval $eval_interval --lr $lr --dataset $dataset $loss_args $wandb_arg"
        python main.py --all_layers --n_layer $n_layer --n_recur $n_recur --n_head $n_head --epochs $epochs --eval_interval $eval_interval --lr $lr --dataset $dataset $loss_args $wandb_arg
        ;;
        
    2)
        echo "=== Running Visual Sudoku Experiment ==="
        read -p "Enter dataset (satnet/palm) [default: satnet]: " dataset
        dataset=${dataset:-satnet}
        
        read -p "Use constraint losses? (y/n) [default: n]: " use_constraints
        use_constraints=${use_constraints:-n}
        
        loss_args=""
        if [[ "$use_constraints" == "y" ]]; then
            read -p "Use sudoku constraints (c1)? (y/n) [default: y]: " use_c1
            use_c1=${use_c1:-y}
            
            read -p "Use attention constraints (att_c1)? (y/n) [default: n]: " use_att_c1
            use_att_c1=${use_att_c1:-n}
            
            if [[ "$use_c1" == "y" && "$use_att_c1" == "y" ]]; then
                loss_args="--loss att_c1 c1 --hyper 0.5 0.5"
            elif [[ "$use_c1" == "y" ]]; then
                loss_args="--loss c1 --hyper 1"
            elif [[ "$use_att_c1" == "y" ]]; then
                loss_args="--loss att_c1 --hyper 0.5"
            fi
        fi
        
        read -p "Number of epochs [default: 10]: " epochs
        epochs=${epochs:-10}
        
        cd visual_sudoku
        echo "Running: python main.py --all_layers --n_layer 1 --n_recur 32 --n_head 4 --epochs $epochs --eval_interval 1 --lr 0.001 --dataset $dataset $loss_args $wandb_arg"
        python main.py --all_layers --n_layer 1 --n_recur 32 --n_head 4 --epochs $epochs --eval_interval 1 --lr 0.001 --dataset $dataset $loss_args $wandb_arg
        ;;
        
    3)
        echo "=== Running 16x16 Sudoku Experiment ==="
        read -p "Enter dataset (easy/medium) [default: easy]: " dataset
        dataset=${dataset:-easy}
        
        cd sudoku_16
        echo "Running: python main.py --dataset $dataset $wandb_arg"
        python main.py --dataset $dataset $wandb_arg
        ;;
        
    4)
        echo "=== Running Shortest Path Experiment ==="
        read -p "Enter grid size (4/12) [default: 4]: " grid_size
        grid_size=${grid_size:-4}
        
        read -p "Use path constraints? (y/n) [default: n]: " use_path
        use_path=${use_path:-n}
        
        loss_args=""
        if [[ "$use_path" == "y" ]]; then
            loss_args="--loss path"
        fi
        
        cd shortest_path
        echo "Running: python main.py --grid_size $grid_size $loss_args $wandb_arg"
        python main.py --grid_size $grid_size $loss_args $wandb_arg
        ;;
        
    5)
        echo "=== Running Nonogram Experiment ==="
        read -p "Enter game size (7/15) [default: 7]: " game_size
        game_size=${game_size:-7}
        
        cd nonogram
        echo "Running: python main.py --game_size $game_size $wandb_arg"
        python main.py --game_size $game_size $wandb_arg
        ;;
        
    6)
        echo "=== Running MNIST Mapping Experiment ==="
        cd MNIST_mapping
        echo "Running: python main.py $wandb_arg"
        python main.py $wandb_arg
        ;;
        
    0)
        echo "Exiting..."
        exit 0
        ;;
        
    *)
        echo "Invalid choice. Exiting..."
        exit 1
        ;;
esac

echo "=== Experiment completed ===" 