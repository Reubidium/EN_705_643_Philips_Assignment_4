"""
Module: test.py

This module provides functions for evaluating a video classification model on test data.
It includes functions to compute predictions, cross-entropy loss and accuracy, generate a report,
and compute a multilabel confusion matrix for all classes.

Functions:
    - test: Evaluates the model on a test DataLoader and returns the ground truth labels,
      predicted labels, and overall accuracy.
    - get_test_report: Generates a classification report using scikit-learn's classification_report.
    - get_confusion_matrix: Computes a multilabel confusion matrix for each class.
"""

import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, multilabel_confusion_matrix


def test(model, dataloader, device, metrics=None):
    """
    Evaluate the model and optionally collect mean loss and accuracy for logging.

    This function sets the model to evaluation mode and processes the test data
    from the provided DataLoader. It computes predictions for each batch, counts the number
    of correct predictions, and accumulates the true and predicted labels.
    Unsmoothed cross-entropy is summed over videos, then divided by the number
    actually evaluated, including when a sampler visits only part of a dataset.

    Args:
        model (torch.nn.Module): The trained video classification model.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the test dataset.
        device (torch.device): The device (CPU or GPU) on which to perform evaluation.
        metrics (dict, optional): Output mapping updated with ``loss`` (mean
            video cross-entropy), ``accuracy`` (fraction correct), and ``samples``.
            Omitting it preserves the original three-value return interface.

    Returns:
        tuple: (targets, outputs, accuracy)
            - targets (list): Ground truth labels for all samples.
            - outputs (list): Predicted labels for all samples.
            - accuracy (float): Overall accuracy computed as the ratio of correct predictions
                                to the total number of samples.

    Raises:
        ValueError: If the loader yields no test videos.
    """
    model.eval()
    with torch.no_grad():
        total_correct_preds = 0
        total_loss = 0.0
        num_seen = 0
        targets, outputs = [], []
        for x_batch, y_batch, lengths in tqdm(dataloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch, lengths)
            total_loss += torch.nn.functional.cross_entropy(
                output,
                y_batch,
                reduction="sum",
            ).item()
            pred = output.argmax(dim=1, keepdim=True)
            total_correct_preds += pred.eq(y_batch.view_as(pred)).sum().item()
            num_seen += y_batch.numel()
            outputs.extend(pred.view(-1).detach().cpu().tolist())
            targets.extend(y_batch.detach().cpu().tolist())

        if num_seen == 0:
            raise ValueError("The test loader must contain at least one video")
        accuracy = total_correct_preds / float(num_seen)
        if metrics is not None:
            metrics.update(loss=total_loss / num_seen, accuracy=accuracy, samples=num_seen)

    return targets, outputs, accuracy


def get_test_report(target, output, target_names):
    """
    Generate a detailed classification report based on test results.

    This function uses scikit-learn's classification_report to produce a dictionary
    containing precision, recall, F1-score, and support for each class.

    Args:
        target (list): Ground truth labels.
        output (list): Predicted labels.
        target_names (list): List of class names corresponding to the labels.

    Returns:
        dict: A classification report as a dictionary.
    """
    return classification_report(target, output, output_dict=True, target_names=target_names)


def get_confusion_matrix(targets, outputs, labels_dict, all_cats):
    """
    Compute a multilabel confusion matrix for each class.

    This function converts numeric labels to their corresponding class names using the provided
    labels_dict, then computes a multilabel confusion matrix for each class using scikit-learn.

    Args:
        targets (list): Ground truth numeric labels.
        outputs (list): Predicted numeric labels.
        labels_dict (dict): Dictionary mapping class names to numeric labels.
        all_cats (list): List of all class names.

    Returns:
        dict: Class names mapped to their corresponding confusion matrices.
    """
    # Create an inverse mapping from numeric label to class name
    inv_labels_dict = {label: cat for cat, label in labels_dict.items()}
    target_cats = [inv_labels_dict[target] for target in targets]
    output_cats = [inv_labels_dict[output] for output in outputs]
    confusion_mat = multilabel_confusion_matrix(target_cats, output_cats, labels=all_cats)
    return dict(zip(all_cats, confusion_mat))
