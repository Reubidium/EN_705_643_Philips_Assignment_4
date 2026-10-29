"""
Module: run.py

This module is the main entry point for training or evaluating a video classification model.
It uses command-line arguments to configure the experiment, including dataset paths, model parameters,
and training hyperparameters. Depending on the selected mode ('train' or 'eval'), it performs
the following:

- Train mode:
    - Loads the dataset from a directory structure.
    - Splits the dataset into training, validation, and test sets.
    - Creates PyTorch Datasets and DataLoaders for training and validation.
    - Initializes the model (e.g., LRCN) and defines the loss function, optimizer, and learning rate scheduler.
    - Runs the training loop and saves the best model weights.

- Eval mode:
    - Loads the pre-generated dataset splits.
    - Creates a DataLoader for the test set.
    - Loads the trained model checkpoint.
    - Prints the evaluation configuration.
    - Evaluates the model on the test set and prints overall test accuracy.
    
The module also includes a helper function for parsing command-line arguments.
"""

import os
import argparse
import json

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

import numpy as np

from video_datasets import VideoDataset, load_dataset, dataset_split
from utils import transform_stats, compose_data_transforms, train_val_dloaders, test_dloaders
from models import LRCN
from train import create_run_directory, train
from test import test, get_test_report, get_confusion_matrix
from tracking import start_tracking, finish_tracking

def args_parser(argv=None):
    """
    Parse command-line arguments for configuring the video classification training or evaluation.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
        
    Arguments include:
        -fd/--frame_dir: Directory for storing video frames.
        -trs/--train_size: Proportion of data to use for training (default 0.7).
        -tss/--test_size: Proportion of data to use for testing (default 0.1).
        -fpv/--fr_per_vid: Number of frames per video to consider (default 16).
        --random_temporal_sampling: Resample training frames on each visit (default False).
        -nc/--n_classes: Number of classes for the classification task (required).
        -c/--ckpt: Path for loading a trained model checkpoint.
        -mt/--model_type: Model type, either '3dcnn' or 'lrcn' (default 'lrcn').
        -cnn/--cnn_backbone: Backbone CNN for 2D feature extraction (default 'resnet34').
        -p/--pretrained: Whether to use a pretrained CNN backbone (default True).
        -rhs/--rnn_hidden_size: LSTM hidden features per direction (default 100).
        -rnl/--rnn_n_layers: Number of RNN/LSTM layers (default 1).
        --bidirectional: Enable whole-sequence BiLSTM processing (default off).
        --rnn_layer_norm: Normalize recurrent outputs before temporal pooling (default off).
        --temporal_attention: Learn temporal pooling weights instead of a mean (default off).
        -m/--mode: Mode of operation: 'train' or 'eval' (required).
        -bs/--batch_size: Mini-batch size (required).
        -d/--dropout: Dropout rate for regularization (default 0.1).
        -lr/--learning_rate: Learning rate for the recurrent head (default 3e-5).
        --cnn_learning_rate: CNN learning rate (defaults to --learning_rate).
        --wandb: Enable optional Weights & Biases logging.
        --wandb_project: Destination project (default assignment-4-ucf50).
        --wandb_entity: Destination team/entity, or the SDK's configured default.
        --wandb_mode: online, offline, or disabled (default online).
        --wandb_name: Optional display name for the W&B run.
        --wandb_run_id: Existing W&B run ID to resume for evaluation only.
        --weight_decay: AdamW weight decay; zero disables decay (default 0.0).
        --label_smoothing: Training target smoothing in [0, 1] (default 0.0).
        -ne/--n_epochs: Number of training epochs (default 30).
    """
    parser = argparse.ArgumentParser(description='Video Classification Training')

    parser.add_argument('-fd', '--frame_dir', help='Directory for storing video frames')
    parser.add_argument('-trs', '--train_size', type=float, default=0.7, help='Train set size')
    parser.add_argument('-tss', '--test_size', type=float, default=0.1, help='Test set size')
    parser.add_argument('-fpv', '--fr_per_vid', type=int, default=16, help='Number of frames per video')
    parser.add_argument('--random_temporal_sampling', action='store_true', help='Randomly select training frames from time segments on each visit (default: off)')
    parser.add_argument('-nc', '--n_classes', type=int, help='Number of classes for the classification task', required=True)

    parser.add_argument('-c', '--ckpt', help='Path for loading trained model checkpoints')
    parser.add_argument('-mt', '--model_type', help='3D CNN or LRCN', default='lrcn')
    parser.add_argument('-cnn', '--cnn_backbone', default='resnet34', help='CNN backbone: resnet18, resnet34, resnet50, resnet101, resnet152, convnext_tiny')
    parser.add_argument('-p', '--pretrained', help='Use pretrained 2D CNN backbone', default=True)
    parser.add_argument('-rhs', '--rnn_hidden_size', type=int, default=100, help='Number of LSTM hidden features per direction')
    parser.add_argument('-rnl', '--rnn_n_layers', type=int, default=1, help='Number of RNN/LSTM layers')
    parser.add_argument('--bidirectional', action='store_true', help='Use a whole-sequence bidirectional LSTM (default: off)')
    parser.add_argument('--rnn_layer_norm', action='store_true', help='Apply LayerNorm to recurrent outputs before temporal pooling (default: off)')
    parser.add_argument('--temporal_attention', action='store_true', help='Use learned, padding-aware temporal attention pooling (default: off)')
    parser.add_argument('--fine_tune_last_block', action='store_true', default=False, help='Freeze early CNN stages; train the final stage, LSTM, and classifier')

    parser.add_argument('-m', '--mode', type=str, default='train', help="Either 'train' or 'eval'", required=True)
    parser.add_argument('-bs', '--batch_size', type=int, help='Mini-batch size', required=True)
    parser.add_argument('-d', '--dropout', type=float, default=0.1, help='Dropout rate for regularization')
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-5, help='Learning rate for the recurrent head (default: 3e-5)')
    parser.add_argument('--cnn_learning_rate', type=float, default=None, help='CNN learning rate; defaults to --learning_rate when omitted')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='AdamW weight decay; use 0 to disable (default: 0.0)')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Training label smoothing in [0, 1]; use 0 to disable (default: 0.0)')
    parser.add_argument('-ne', '--n_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading (default: 0)')
    parser.add_argument('--wandb', action='store_true', help='Log metrics and configuration to Weights & Biases')
    parser.add_argument('--wandb_project', default='assignment-4-ucf50', help='W&B project name')
    parser.add_argument('--wandb_entity', default=None, help='W&B team/entity; otherwise use the configured SDK default')
    parser.add_argument('--wandb_mode', choices=('online', 'offline', 'disabled'), default='online', help='W&B logging mode')
    parser.add_argument('--wandb_name', default=None, help='Optional W&B run display name')
    parser.add_argument('--wandb_run_id', default=None, help='Existing W&B run ID to resume when evaluating its checkpoint')

    args = parser.parse_args(argv)
    for option in ('learning_rate', 'cnn_learning_rate'):
        value = getattr(args, option)
        if value is not None and not 0.0 < value < float('inf'):
            parser.error(f'--{option} must be finite and positive')
    if not 0.0 <= args.weight_decay < float('inf'):
        parser.error('--weight_decay must be finite and nonnegative')
    if not 0.0 <= args.label_smoothing <= 1.0:
        parser.error('--label_smoothing must be finite and between 0 and 1')
    if args.wandb_run_id and (not args.wandb or args.mode != 'eval'
                              or args.wandb_mode != 'online'):
        parser.error('--wandb_run_id requires --wandb, --mode eval, and online mode')
    return args


def _evaluation_tracking_args(args, run_dir):
    """Reuse the checkpoint's recorded online run unless an ID is explicitly given.

    Only pointers in the requested project/team are eligible. Old checkpoints
    without tracking metadata get a separate evaluation run. An explicit
    ``--wandb_run_id`` can attach evaluation to an imported historical run.
    """
    if (not args.wandb or args.wandb_mode != 'online' or args.wandb_run_id):
        return args
    metadata_path = os.path.join(run_dir, 'tracking.json')
    if not os.path.isfile(metadata_path):
        return args
    with open(metadata_path, encoding='utf-8') as source:
        metadata = json.load(source)
    if not isinstance(metadata, dict):
        raise ValueError('tracking.json must contain a W&B run metadata object')
    if metadata.get('mode') != 'online' or metadata.get('job_type') not in ('train', 'import'):
        return args
    if (metadata.get('project') != args.wandb_project
            or (args.wandb_entity and metadata.get('entity') != args.wandb_entity)):
        raise ValueError(
            'The checkpoint W&B destination differs from --wandb_project/--wandb_entity. '
            'Use the original destination or an explicit --wandb_run_id.'
        )
    if not metadata.get('run_id'):
        raise ValueError('tracking.json is missing its W&B run_id')
    values = vars(args).copy()
    values.update(wandb_run_id=metadata['run_id'], wandb_entity=metadata.get('entity'))
    print(f"Appending test metrics to W&B run: {metadata['run_id']}")
    return argparse.Namespace(**values)

def main(args):
    """
    Main function to execute training or evaluation based on the parsed command-line arguments.

    For training:
        - Loads the dataset from the specified frame directory.
        - Splits the dataset into training, validation, and test sets.
        - Saves the splits for later use.
        - Creates the training and validation DataLoaders.
        - Initializes the model (LRCN) with specified hyperparameters.
        - Defines the loss function, optimizer, and learning rate scheduler.
        - Runs the training loop and saves the best model weights.

    For evaluation:
        - Loads the saved dataset splits.
        - Creates the test DataLoader.
        - Loads the trained model checkpoint.
        - Evaluates the model on the test set and prints the overall test accuracy.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # Dataset parameters
    frame_dir = args.frame_dir
    tr_size = args.train_size
    ts_size = args.test_size
    fr_per_vid = args.fr_per_vid
    n_classes = args.n_classes

    # Model parameters
    model_type = args.model_type
    rnn_hidden_size = args.rnn_hidden_size
    rnn_n_layers = args.rnn_n_layers
    bidirectional = args.bidirectional
    rnn_layer_norm = args.rnn_layer_norm
    temporal_attention = args.temporal_attention
    dropout = args.dropout
    pretrained = args.pretrained
    cnn_backbone = args.cnn_backbone
    ckpt = args.ckpt
    fine_tune_last_block = args.fine_tune_last_block

    # Training parameters
    mode = args.mode
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    learning_rate = args.learning_rate
    cnn_learning_rate = (learning_rate if args.cnn_learning_rate is None else args.cnn_learning_rate)
    weight_decay = args.weight_decay
    label_smoothing = args.label_smoothing
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    # Data loading parameters
    num_workers = args.num_workers

    # Load transformation statistics and create data augmentation transforms
    h, w, mean, std = transform_stats(model_type)
    tr_transforms, val_ts_transforms = compose_data_transforms(h, w, mean, std)

    # Initialize the model (LRCN)
    model = LRCN(
        hidden_size=rnn_hidden_size, n_layers=rnn_n_layers, dropout_rate=dropout,
        n_classes=n_classes, pretrained=pretrained, cnn_model=cnn_backbone,
        fine_tune_last_block=fine_tune_last_block, bidirectional=bidirectional,
        rnn_layer_norm=rnn_layer_norm,
        temporal_attention=temporal_attention,
    )

    if mode == 'train':
        # Print the configuration
        print(
            "\n"
            "========== Training Configuration ==========\n"
            "Dataset parameters\n"
            f"  Frame directory:            {frame_dir}\n"
            f"  Training split:             {tr_size}\n"
            f"  Test split:                 {ts_size}\n"
            f"  Frames per video:           {fr_per_vid}\n"
            f"  Random temporal sampling:   {args.random_temporal_sampling}\n"
            f"  Number of classes:          {n_classes}\n"
            "\n"
            "Model parameters\n"
            f"  Model type:                 {model_type}\n"
            f"  RNN hidden size/direction:  {rnn_hidden_size}\n"
            f"  RNN layers:                 {rnn_n_layers}\n"
            f"  Bidirectional LSTM:         {bidirectional}\n"
            f"  RNN output LayerNorm:       {rnn_layer_norm}\n"
            f"  Temporal attention:         {temporal_attention}\n"
            f"  Dropout:                    {dropout}\n"
            f"  Pretrained backbone:        {pretrained}\n"
            f"  CNN backbone:               {cnn_backbone}\n"
            f"  Fine-tune final CNN stage:  {fine_tune_last_block}\n"
            f"  Checkpoint:                 {ckpt}\n"
            "\n"
            "Training parameters\n"
            f"  Mode:                       {mode}\n"
            f"  Batch size:                 {batch_size}\n"
            f"  Epochs:                     {n_epochs}\n"
            f"  CNN learning rate:          {cnn_learning_rate}\n"
            f"  Recurrent head rate:        {learning_rate}\n"
            f"  Optimizer:                  AdamW\n"
            f"  Weight decay:               {weight_decay}\n"
            f"  Training label smoothing:   {label_smoothing}\n"
            f"  Device:                     {device}\n"
            "\n"
            "Data loading parameters\n"
            f"  Training workers:           {num_workers}\n"
            "=======================================\n"
        )
        # Load dataset and split into train/validation/test
        run_dir = create_run_directory()
        optim_model_dir = os.path.join(run_dir, 'models')
        print(f"Saving run outputs to: {run_dir}")

        vid_dataset, label_dict = load_dataset(frame_dir)
        tr_split, val_split, ts_split = dataset_split(vid_dataset, tr_size, ts_size)

        # Save the splits for reproducibility and later use in evaluation
        splits = {'train': np.array(tr_split),
                  'val': np.array(val_split),
                  'test': np.array(ts_split)}
        splits_path = os.path.join(run_dir, 'splits.npy')
        np.save(splits_path, splits)

        # Create PyTorch Datasets and DataLoaders for train and validation
        tr_dataset = VideoDataset(
            tr_split, fr_per_vid, tr_transforms, training=args.random_temporal_sampling,
        )
        val_dataset = VideoDataset(val_split, fr_per_vid, val_ts_transforms)
        dataloaders = train_val_dloaders(tr_dataset, val_dataset, batch_size, num_workers, model_type)

        # Define the loss function, optimizer, and learning rate scheduler
        model.to(device)
        loss_func = nn.CrossEntropyLoss(reduction='sum', label_smoothing=label_smoothing)
        # Keep validation loss comparable with earlier unsmoothed runs.
        val_loss_func = nn.CrossEntropyLoss(reduction='sum')
        # Preserve pretrained CNN features with a separately configurable step size.
        # The head includes the LSTM, LayerNorm, attention (when enabled), and classifier.
        opt = optim.AdamW([
            {'name': 'cnn', 'params': model.base_model.parameters(), 'lr': cnn_learning_rate},
            {
                'name': 'head',
                'params': [
                    parameter for name, parameter in model.named_parameters()
                    if not name.startswith('base_model.')
                ],
                'lr': learning_rate,
            },
        ], lr=learning_rate, weight_decay=weight_decay)
        lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)

        # Preserve the exact configuration for later CSV imports and reproduction.
        configuration = {
            **vars(args), 'training_method': 'joint', 'device_type': device.type,
            'training_videos': len(tr_split), 'validation_videos': len(val_split),
            'test_videos': len(ts_split), 'label_mapping': label_dict,
            'effective_cnn_learning_rate': cnn_learning_rate, 'split_seed': 0,
        }
        with open(os.path.join(run_dir, 'training_config.json'), 'w', encoding='utf-8') as output:
            json.dump(configuration, output, indent=2)
        wandb_run = start_tracking(args, run_dir, configuration, job_type='train')
        exit_code = 0
        try:
            model, loss_hist, acc_hist = train(
                dataloaders, model, loss_func, opt, lr_scheduler, device,
                optim_model_dir, n_epochs, metrics_dir=run_dir,
                val_criterion=val_loss_func, wandb_run=wandb_run,
            )
        except BaseException:
            exit_code = 1
            raise
        finally:
            finish_tracking(wandb_run, exit_code=exit_code)

    elif mode == 'eval':
        print(
            "\n"
            "========== Evaluation Configuration ==========\n"
            f"  Checkpoint:                 {ckpt}\n"
            f"  Model type:                 {model_type}\n"
            f"  CNN backbone:               {cnn_backbone}\n"
            f"  RNN hidden size/direction:  {rnn_hidden_size}\n"
            f"  RNN layers:                 {rnn_n_layers}\n"
            f"  Bidirectional LSTM:         {bidirectional}\n"
            f"  RNN output LayerNorm:       {rnn_layer_norm}\n"
            f"  Temporal attention:         {temporal_attention}\n"
            f"  Number of classes:          {n_classes}\n"
            f"  Frames per video:           {fr_per_vid}\n"
            f"  Batch size:                 {batch_size}\n"
            f"  Device:                     {device}\n"
            "==============================================\n"
        )

        # Load saved dataset splits
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(ckpt)))
        splits_path = os.path.join(run_dir, 'splits.npy')
        splits = np.load(splits_path, allow_pickle=True)
        ts_split = splits.item()['test']
        ts_split = [(sample[0], int(sample[1])) for sample in ts_split]

        # Create PyTorch Dataset and DataLoader for the test set
        ts_dataset = VideoDataset(ts_split, fr_per_vid, val_ts_transforms)
        dataloaders = test_dloaders(ts_dataset, batch_size, num_workers, model_type)

        # Load the trained model checkpoint
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        model.to(device)
        tracking_args = _evaluation_tracking_args(args, run_dir)
        wandb_run = start_tracking(
            tracking_args, run_dir, {'evaluation_checkpoint': os.path.abspath(ckpt),
                            'test_videos': len(ts_split), 'device_type': device.type},
            job_type='eval',
        )
        exit_code = 0
        try:
            evaluation_metrics = {}
            targets, outputs, accuracy = test(
                model, dataloaders['test'], device, evaluation_metrics,
            )
            if evaluation_metrics:
                result = {
                    **evaluation_metrics, 'checkpoint': os.path.abspath(ckpt),
                    'splits_file': os.path.abspath(splits_path),
                    'evaluation_config': vars(args),
                }
                stem = os.path.splitext(os.path.basename(ckpt))[0]
                metrics_path = os.path.join(run_dir, f'test_metrics_{stem}.json')
                with open(metrics_path, 'w', encoding='utf-8') as output:
                    json.dump(result, output, indent=2)
                print(f'Saved test metrics to: {metrics_path}')
                if wandb_run is not None:
                    logged = {f'test/{name}': value for name, value in evaluation_metrics.items()}
                    wandb_run.log(logged)
                    for name, value in logged.items():
                        wandb_run.summary[name] = value
                    wandb_run.summary['test_checkpoint'] = os.path.abspath(ckpt)
        except BaseException:
            exit_code = 1
            raise
        finally:
            finish_tracking(wandb_run, exit_code=exit_code)
        if evaluation_metrics:
            print(f"The overall test loss is {evaluation_metrics['loss']:.6f}.")
        print('The overall test accuracy is {:.4f}%.'.format(100 * accuracy))

    else:
        raise ValueError('The mode argument must be either "train" or "eval".')

if __name__ == "__main__":
    args = args_parser()
    main(args)
