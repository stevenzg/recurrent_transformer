import sys 
sys.path.append('..')

import argparse
import torch
import re
import os

from dataset import Sudoku_Dataset, Sudoku_Dataset_Palm, Sudoku_Dataset_SATNet
from network import testNN
from helper import print_result, visualize_adjacency
from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig
from mingpt.utils import set_seed

def main(args):
    print('Hyperparameters: ', args.hyper)

    # generate the name of this experiment
    prefix = f'[{args.dataset},{args.n_train//1000}k]'
    if args.label_size < args.batch_size:
        prefix = prefix[:-1] + f',{args.label_size}-{args.batch_size-args.label_size}]'
    if args.loss:
        prefix += '[' + '-'.join(args.loss) + ';' + '-'.join([str(v) for v in args.hyper]) + ']'
    prefix += f'L{args.n_layer}R{args.n_recur}H{args.n_head}_'

    if args.wandb:
        import wandb
        wandb.init(project='transformer-ste-sudoku')
        wandb.run.name = prefix[:-1]
    else:
        wandb = None

    #############
    # Seed everything for reproductivity
    #############
    set_seed(args.seed)

    #############
    # Load data
    #############
    train_dataset_ulb = None
    if args.dataset == '70k':
        input_path = '../data/easy_130k_given.p'
        label_path = '../data/easy_130k_solved.p'
        dataset = Sudoku_Dataset(input_path, label_path, args.n_train + args.n_test, args.seed)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [args.n_train, args.n_test])
    elif args.dataset == 'satnet':
        dataset = Sudoku_Dataset_SATNet()
        indices = list(range(len(dataset)))
        # we use the same test set in the SATNet repository for comparison
        test_dataset = torch.utils.data.Subset(dataset, indices[-1000:])
        train_dataset = torch.utils.data.Subset(dataset, indices[:min(9000, args.n_train)])
    elif args.dataset == 'palm':
        train_dataset = Sudoku_Dataset_Palm(segment='train', limit=args.n_train, seed=args.seed)
        test_dataset = Sudoku_Dataset_Palm(segment='test', limit=args.n_test, seed=args.seed)
    # check if we have unlabeled training data
    n_train_lb = int(args.n_train * (args.label_size / args.batch_size))
    n_train_ulb = args.n_train - n_train_lb
    if n_train_ulb:
        indices = list(range(len(train_dataset)))
        train_dataset_ulb = torch.utils.data.Subset(train_dataset, indices[n_train_lb:])
        train_dataset = torch.utils.data.Subset(train_dataset, indices[:n_train_lb])
    if train_dataset_ulb:
        print(f'[{args.dataset}] use {len(train_dataset) + len(train_dataset_ulb)} ({len(train_dataset)}lb + {len(train_dataset_ulb)}ulb) for training and {len(test_dataset)} for testing')
    else:
        print(f'[{args.dataset}] use {len(train_dataset)} for training and {len(test_dataset)} for testing')

    #############
    # Construct a GPT model and a trainer
    #############
    # vocab_size is the number of different digits in the input
    mconf = GPTConfig(vocab_size=10, block_size=81, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, 
        num_classes=9, causal_mask=False, losses=args.loss, n_recur=args.n_recur, all_layers=args.all_layers,
        hyper=args.hyper)
    model = GPT(mconf)

    # Load model from wandb artifact if specified
    if args.continue_from and args.wandb:
        print(f"Loading model from wandb artifact: {args.continue_from}")
        artifact = wandb.run.use_artifact(args.continue_from, type='model')
        artifact_dir = artifact.download()
        
        # Find model file
        model_files = [f for f in os.listdir(artifact_dir) if f.endswith('.pt')]
        if not model_files:
            raise FileNotFoundError(f"No model file (.pt) found in {artifact_dir}")
        
        model_path = os.path.join(artifact_dir, model_files[0])
        print(f"Using model file: {model_path}")
        
        # Load model weights
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print("Model loaded successfully. Continuing training...")

    if args.wandb: wandb.watch(model, log_freq=100)
    if args.heatmap: visualize_adjacency()

    tconf = TrainerConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        label_size=args.label_size,
        learning_rate=args.lr,
        lr_decay=args.lr_decay,
        warmup_tokens=1024, # until which point we increase lr from 0 to lr; lr decays after this point
        final_tokens=100 * len(train_dataset), # at what point we reach 10% of lr
        eval_funcs=[testNN], # test without inference trick
        eval_interval=args.eval_interval, # test for every eval_interval number of epochs
        gpu=args.gpu,
        heatmap=args.heatmap,
        prefix=prefix,
        wandb=wandb
    )

    trainer = Trainer(model, train_dataset, train_dataset_ulb, test_dataset, tconf)

    #############
    # Start training
    #############
    trainer.train()
    result = trainer.result
    print('Total and single accuracy are the board and cell accuracy respectively.')
    print_result(result)

    # Save model to wandb if enabled
    if args.wandb:
        # Create a valid artifact name by replacing invalid characters
        # Only allow alphanumeric characters, dashes, underscores, and dots
        artifact_name = prefix[:-1]
        # Replace special characters with underscore
        artifact_name = re.sub(r'[^\w\-\.]', '_', artifact_name)
        # Remove consecutive underscores
        artifact_name = re.sub(r'_+', '_', artifact_name)
        # Remove leading/trailing underscores
        artifact_name = artifact_name.strip('_')
        print(f"Using artifact name: model-{artifact_name}")
        
        # Save model state dict
        model_path = f'model_{artifact_name}.pt'
        torch.save(model.state_dict(), model_path)
        
        # Log model as artifact
        artifact = wandb.Artifact(
            name=f'model-{artifact_name}',
            type='model',
            description=f'Model trained on {args.dataset} dataset with {args.n_train} samples'
        )
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
        
        # Clean up local file
        os.remove(model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Training
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs.')
    parser.add_argument('--eval_interval', type=int, default=1, help='Compute accuracy for how many number of epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--label_size', type=int, default=16, help='The number of labeled training data in a batch')
    parser.add_argument('--lr', type=float, default=6e-4, help='Learning rate')
    parser.add_argument('--lr_decay', default=False, action='store_true', help='use lr_decay defined in minGPT')
    # Model and loss
    parser.add_argument('--n_layer', type=int, default=1, help='Number of sequential self-attention blocks.')
    parser.add_argument('--n_recur', type=int, default=32, help='Number of recurrency of all self-attention blocks.')
    parser.add_argument('--n_head', type=int, default=4, help='Number of heads in each self-attention block.')
    parser.add_argument('--n_embd', type=int, default=128, help='Vector embedding size.')
    parser.add_argument('--loss', default=[], nargs='+', help='specify regularizers in \{c1, att_c1\}')
    parser.add_argument('--all_layers', default=False, action='store_true', help='apply losses to all self-attention layers')    
    parser.add_argument('--hyper', default=[1, 0.1], nargs='+', type=float, help='Hyper parameters: Weights of [L_sudoku, L_attention]')
    # Data
    parser.add_argument('--dataset', type=str, default='palm', help='Name of dataset in \{satnet, palm, 70k\}')
    parser.add_argument('--n_train', type=int, default=9000, help='The number of data for training')
    parser.add_argument('--n_test', type=int, default=1000, help='The number of data for testing')
    # Other
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproductivity.')
    parser.add_argument('--gpu', type=int, default=-1,
        help='gpu index; -1 means using all GPUs or using CPU if no GPU is available')
    parser.add_argument('--debug', default=False, action='store_true', help='debug mode')
    parser.add_argument('--wandb', default=False, action='store_true', help='save all logs on wandb')
    parser.add_argument('--heatmap', default=False, action='store_true', help='save all heatmaps in trainer.result')
    parser.add_argument('--comment', type=str, default='', help='Comment of the experiment')
    parser.add_argument('--continue_from', type=str, default='', help='WandB artifact path to continue training from')
    args = parser.parse_args()

    # we do not log onto wandb in debug mode
    if args.debug: args.wandb = False
    main(args)
