"""
Module: utils.py

This module provides helper functions for video processing and data transformations 
for video classification tasks. It includes functions for:
    - Uniformly sampling frames from videos.
    - Storing extracted frames as JPEG images.
    - Retrieving ImageNet normalization statistics for LRCN frames.
    - Composing data transforms for training and validation/test datasets.
    - Creating DataLoaders for training, validation, and testing, using custom collate functions.
"""

import os
import cv2
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader
from video_datasets import collate_fn_rnn


def get_frames(vid, n_frames=1):
    """
    Uniformly sample frames from a video file.

    Args:
        vid (str): Path to the video file.
        n_frames (int): Number of frames to sample from the video.

    Returns:
        tuple: (frames, v_len)
            - frames (list): List of sampled frames (as numpy arrays in RGB format).
            - v_len (int): Total number of frames in the video.

    Notes:
        - If the video cannot be opened or contains no frames, an empty list and 0 are returned.
        - Frames are sampled at uniformly spaced indices.
    """
    frames = []
    v_cap = cv2.VideoCapture(vid)
    if not v_cap.isOpened():
        print(f"[utils.py] ERROR - Failed to open video: {vid}")
        return frames, 0
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if v_len <= 0:
        print(f"[utils.py] WARNING - No frames found in video: {vid}")
        v_cap.release()
        return frames, 0
    frame_idx = np.linspace(0, v_len - 1, n_frames, dtype=np.int16)
    for idx in range(v_len):
        success, frame = v_cap.read()
        if not success:
            print(f"[utils.py] WARNING - Failed to read frame {idx} from video: {vid}")
            continue
        if idx in frame_idx:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    v_cap.release()
    return frames, v_len


def store_frames(frames, store_path):
    """
    Save a list of frames as JPEG images to the specified directory.

    Each frame is converted from RGB to BGR format (as expected by OpenCV)
    before saving.

    Args:
        frames (list): List of frames (numpy arrays in RGB format) to save.
        store_path (str): Directory path where the frames will be stored.

    Returns:
        None
    """
    for idx, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        path_to_frame = os.path.join(store_path, f"frame{idx}.jpg")
        cv2.imwrite(path_to_frame, frame)


def transform_stats():
    """Return the LRCN frame size and ImageNet normalization statistics.

    Returns:
        tuple: Image height, image width, RGB mean, and RGB standard deviation.
    """
    return 224, 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def compose_data_transforms(height, width, mean, std):
    """
    Compose and return data transforms for training and validation/test datasets.

    Training adds mild crop/scale variation, flipping, translation, and color jitter.
    VideoDataset calls the pipeline once on the whole tensor clip, so sampled
    settings are shared by all frames. Validation/test use only resize/normalize.

    Args:
        height (int): Desired image height.
        width (int): Desired image width.
        mean (list): Mean values for normalization.
        std (list): Standard deviation values for normalization.

    Returns:
        tuple: (train_transforms, val_test_transforms)
            - train_transforms: Composed transforms for the training set.
            - val_test_transforms: Composed transforms for the validation/test set.
    """
    train_transforms = transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.RandomResizedCrop(
                (height, width),
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.Normalize(mean, std),
        ]
    )
    val_test_transforms = transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.Normalize(mean, std),
        ]
    )
    return train_transforms, val_test_transforms


def train_val_dloaders(train_dataset, val_dataset, batch_size, num_workers):
    """Create training and validation loaders for padded LRCN clips.

    Args:
        train_dataset (Dataset): Training clips.
        val_dataset (Dataset): Validation clips.
        batch_size (int): Training batch size; validation uses twice this size.
        num_workers (int): Number of worker processes for data loading.

    Returns:
        dict: Training and validation DataLoaders with genuine sequence lengths.
    """
    train_dl = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_rnn,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size=2 * batch_size,
        shuffle=False,
        collate_fn=collate_fn_rnn,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )
    return {"train": train_dl, "val": val_dl}


def test_dloaders(test_dataset, batch_size, num_workers):
    """Create a deterministic test loader for padded LRCN clips.

    Args:
        test_dataset (Dataset): Held-out test clips.
        batch_size (int): Base batch size; evaluation uses twice this size.
        num_workers (int): Number of worker processes for data loading.

    Returns:
        dict: Test DataLoader with genuine sequence lengths.
    """
    test_dl = DataLoader(
        test_dataset,
        batch_size=2 * batch_size,
        shuffle=False,
        collate_fn=collate_fn_rnn,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )
    return {"test": test_dl}
