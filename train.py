"""
Module: train.py

This module provides functions to train and evaluate a video classification model.
It includes the main training loop that tracks loss and accuracy over epochs,
updates the best performing model based on validation accuracy, and provides helper
functions to compute the learning rate, batch loss, and epoch loss.

Functions:
    - train: Runs the training and validation loops over a specified number of epochs.
    - get_learning_rate: Retrieves the current learning rate for an optimizer group.
    - batch_correct_preds: Computes the number of correct predictions in a mini-batch.
    - get_batch_loss: Computes the loss for a mini-batch and performs backpropagation.
    - get_epoch_loss: Computes the average loss and accuracy over an epoch.
"""

import os
import copy
from tqdm import tqdm
import torch
import csv
from datetime import datetime

def create_run_directory(runs_dir='./runs'):
    """Create a timestamped run directory with a nested models directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = os.path.join(runs_dir, timestamp)
    os.makedirs(os.path.join(run_dir, "models"))
    return run_dir


def train(
    dataloaders,
    model,
    criterion,
    optimizer,
    scheduler,
    device,
    optim_model_wts_dir,
    n_epochs=30,
    metrics_dir=None,
    val_criterion=None,
):
    """
    Train and validate the model over a given number of epochs.
    
    This function performs the training loop for a video classification model.
    It iterates through the specified number of epochs, updates model weights
    using backpropagation, and evaluates model performance on a validation set.
    The model with the best validation accuracy is saved to disk.

    Args:
        dataloaders (dict): Dictionary containing 'train' and 'val' DataLoaders.
        model (torch.nn.Module): The video classification model to be trained.
        criterion (callable): Training loss function.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler, which adjusts the learning rate based on validation loss.
        device (torch.device): Device (CPU or GPU) on which to perform training.
        optim_model_wts_dir (str): Directory to save the best model weights.
        metrics_dir (str, optional): Directory for the metrics CSV. Defaults to the model directory.
        n_epochs (int, optional): Number of training epochs. Default is 30.
        val_criterion (callable, optional): Validation loss function. Defaults to
            the training criterion, preserving existing calls.

    Returns:
        tuple: (model, loss_hist, acc_hist)
            - model (torch.nn.Module): The trained model loaded with the best validation weights.
            - loss_hist (dict): Dictionary containing lists of training and validation losses for each epoch.
            - acc_hist (dict): Dictionary containing lists of training and validation accuracies for each epoch.
    """
    loss_hist = {'train': [], 'val': []}
    acc_hist = {'train': [], 'val': []}
    if val_criterion is None:
        val_criterion = criterion

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    # Initialize training logs and metrics
    metrics_path = os.path.join(metrics_dir or optim_model_wts_dir, "metrics.csv")
    with open(metrics_path, "x", newline="", encoding="utf-8") as metrics_file:
        writer = csv.writer(metrics_file)
        writer.writerow([
            "epoch",
            "learning_rate",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "cnn_learning_rate",
        ])
    print(f"Saving metrics to: {metrics_path}")

    # Begin training loop
    for epoch in range(n_epochs):
        current_lr = get_learning_rate(optimizer)
        cnn_lr = get_learning_rate(optimizer, 'cnn')
        print(
            f'Epoch {epoch + 1}/{n_epochs}; '
            f'CNN learning rate {cnn_lr}; recurrent head learning rate {current_lr}'
        )

        # Training phase
        model.train()
        train_loss, train_accuracy = get_epoch_loss(model, criterion, dataloaders['train'], device, optimizer)
        loss_hist['train'].append(train_loss)
        acc_hist['train'].append(train_accuracy)

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_loss, val_accuracy = get_epoch_loss(
                model, val_criterion, dataloaders['val'], device,
            )
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
            best_model_name = 'best_model_wts.pt'
            best_model_path = os.path.join(optim_model_wts_dir, best_model_name)
            torch.save(best_model_wts, best_model_path)
            print('Best model weights are updated at epoch {}!'.format(epoch+1))
        loss_hist['val'].append(val_loss)
        acc_hist['val'].append(val_accuracy)

        # Log metrics to CSV
        with open(metrics_path, "a", newline="", encoding="utf-8") as metrics_file:
            writer = csv.writer(metrics_file)
            writer.writerow([
                epoch + 1,
                current_lr,
                train_loss,
                train_accuracy,
                val_loss,
                val_accuracy,
                cnn_lr,
            ])

        # Update learning rate based on validation loss
        scheduler.step(val_loss)

        print(
            f"train loss: {train_loss:.6f}, "
            f"train accuracy: {100 * train_accuracy:.2f}%, "
            f"val loss: {val_loss:.6f}, "
            f"val accuracy: {100 * val_accuracy:.2f}%"
        )
        print("-" * 60)
        print()

    # Load the best model weights before returning the model
    model.load_state_dict(best_model_wts)
    return model, loss_hist, acc_hist

def get_learning_rate(optimizer, group_name='head'):
    """
    Retrieve a named group's rate, or the first rate for legacy optimizers.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer from which to get the learning rate.
        group_name (str): Parameter-group name. Defaults to the recurrent head.
    
    Returns:
        float: The named group's rate, falling back to the first group when absent.
    """
    for param_group in optimizer.param_groups:
        if param_group.get('name') == group_name:
            return param_group['lr']
    return optimizer.param_groups[0]['lr']

def batch_correct_preds(output, target):
    """
    Compute the number of correct predictions for a mini-batch.
    
    Args:
        output (torch.Tensor): Model outputs (logits) with shape (batch_size, num_classes).
        target (torch.Tensor): True labels with shape (batch_size).
    
    Returns:
        int: Number of correct predictions in the mini-batch.
    """
    pred = output.argmax(dim=1, keepdim=True)
    correct_preds = pred.eq(target.view_as(pred)).sum().item()
    return correct_preds

def get_batch_loss(criterion, output, target, optimizer=None):
    """
    Compute the loss for a mini-batch and perform backpropagation (if optimizer is provided).
    
    Args:
        criterion (callable): Loss function.
        output (torch.Tensor): Model outputs for the mini-batch.
        target (torch.Tensor): True labels for the mini-batch.
        optimizer (torch.optim.Optimizer, optional): Optimizer to update model weights. If None, no backpropagation is performed.
    
    Returns:
        tuple: (loss_value, n_batch_correct_preds)
            - loss_value (float): Loss value for the mini-batch.
            - n_batch_correct_preds (int): Number of correct predictions in the mini-batch.
    """
    loss = criterion(output, target)
    with torch.no_grad():
        n_batch_correct_preds = batch_correct_preds(output, target)
    if optimizer:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item(), n_batch_correct_preds

def get_epoch_loss(model, criterion, dataloader, device, optimizer=None):
    """
    Compute the average loss and overall accuracy for an epoch.

    Iterates over the entire DataLoader, computes loss and accuracy for each mini-batch,
    and aggregates the results over the epoch.

    Args:
        model (torch.nn.Module): The video classification model.
        criterion (callable): Loss function.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (torch.device): Device (CPU or GPU) on which to perform computations.
        optimizer (torch.optim.Optimizer, optional): If provided, used to update model weights during training.

    Returns:
        tuple: (loss, accuracy)
            - loss (float): Average loss over the epoch.
            - accuracy (float): Overall accuracy over the epoch.
    """
    running_loss, running_total_correct_preds = 0.0, 0.0
    len_dataset = len(dataloader.dataset)

    for x_batch, y_batch, lengths in tqdm(dataloader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        output = model(x_batch, lengths)
        batch_loss, n_batch_correct_preds = get_batch_loss(criterion, output, y_batch, optimizer)

        running_loss += batch_loss
        running_total_correct_preds += n_batch_correct_preds

    loss = running_loss / float(len_dataset)
    accuracy = running_total_correct_preds / float(len_dataset)
    return loss, accuracy
