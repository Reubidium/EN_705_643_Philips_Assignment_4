"""
Module: video_datasets.py

This module provides classes and functions for loading and processing video datasets,
splitting them into training, validation, and test sets, and preparing data for video
classification models. It includes a custom PyTorch Dataset for videos stored as directories
of frame images, functions to load the dataset from a directory structure, split the dataset
using group-disjoint sampling, and collate functions for variable-length sequences.
"""

import glob
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm


def sample_frame_indices(frame_count, sample_count, random_sampling=False):
    """Select chronological frames spanning a cache without duplicating short clips.

    Training randomly selects one index from each of ``sample_count`` disjoint
    temporal segments. Evaluation uses fixed, uniformly spaced indices including
    the first and last frames when selecting at least two frames. Torch's RNG is
    used so DataLoader worker seeds also control temporal augmentation.
    """
    if frame_count < 0:
        raise ValueError("frame_count must be nonnegative")
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if frame_count <= sample_count:
        return list(range(frame_count))
    if not random_sampling:
        return torch.linspace(0, frame_count - 1, steps=sample_count).long().tolist()

    boundaries = [index * frame_count // sample_count for index in range(sample_count + 1)]
    return [
        torch.randint(start, stop, (1,)).item()
        for start, stop in zip(boundaries[:-1], boundaries[1:])
    ]


class VideoDataset(Dataset):
    """
    PyTorch Dataset class for loading video data from directories of frame images.

    Each video is represented as a directory containing numerically named frame JPEGs.
    Training resamples the timeline on each access. Validation and test sampling are
    deterministic, and all spatial/color transforms receive the entire clip at once.

    Args:
        vid_dataset (sequence): Pairs of video-directory paths and integer labels.
        fr_per_vid (int): Maximum frames to load per video, sampled across its cache.
        transforms (callable, optional): Transform accepting a complete (T, C, H, W) clip.
        training (bool): Enable random segment sampling. Defaults to deterministic sampling.
    """

    def __init__(self, vid_dataset, fr_per_vid, transforms=None, training=False):
        if fr_per_vid < 1:
            raise ValueError("fr_per_vid must be positive")
        self.dataset = vid_dataset
        self.fpv = fr_per_vid
        self.transforms = transforms
        self.training = training

    def __len__(self):
        """Return the number of video samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Load frames from the video directory corresponding to the given index, apply transforms,
        and return the stacked tensor of frames along with its label.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (frames_tensor, label) where frames_tensor is a tensor of shape (T, C, H, W)
                   with T being the number of frames (up to fr_per_vid) and label is an integer.
        """
        video_path, label = self.dataset[idx]
        frame_paths = sorted(
            glob.glob(os.path.join(video_path, "*.jpg")),
            key=lambda frame_path: int(Path(frame_path).stem[len("frame") :]),
        )
        indices = sample_frame_indices(len(frame_paths), self.fpv, self.training)
        if not indices:
            raise ValueError(f"A video has no frames; check preprocessing: {video_path}")

        # Decode only the selected frames, even when the cache contains many more.
        frames = []
        for frame_index in indices:
            with Image.open(frame_paths[frame_index]) as frame:
                frames.append(to_tensor(frame.convert("RGB")))
        clip = torch.stack(frames)

        # Sample augmentation settings once for all timesteps to preserve motion.
        if self.transforms is not None:
            clip = self.transforms(clip)
        return clip, label


def load_dataset(frame_dir):
    """
    Load the full video dataset from the specified directory.

    Each subdirectory in frame_dir is assumed to correspond to a video category.
    The function builds a dictionary where keys are paths to video directories and
    values are integer labels corresponding to each category.

    Args:
        frame_dir (str): Path to the directory containing subdirectories for each video category.

    Returns:
        tuple: (vid_dataset, label_dict)
            - vid_dataset (dict): Dictionary mapping video directory paths to integer labels.
            - label_dict (dict): Dictionary mapping video category names to integer labels.
    """
    label_dict = {vid_cat: idx for idx, vid_cat in enumerate(sorted(os.listdir(frame_dir)))}
    vid_dataset = {}
    print("Loading video dataset....")
    for vid_cat in tqdm(sorted(os.listdir(frame_dir))):
        vid_cat_path = os.path.join(frame_dir, vid_cat)
        for vid in os.listdir(vid_cat_path):
            vid_path = os.path.join(vid_cat_path, vid)
            vid_dataset[vid_path] = label_dict[vid_cat]
    return vid_dataset, label_dict


def dataset_split(vid_dataset, tr_ratio, ts_ratio, seed=0):
    """
    Split the dataset into training, validation, and test sets using group shuffle sampling.

    GroupShuffleSplit keeps each UCF50 group ID in only one split.
    Group sampling does not guarantee exact class proportions.

    Args:
        vid_dataset (dict): Dictionary mapping video paths to labels.
        tr_ratio (float): Proportion of the data to use for training.
        ts_ratio (float): Proportion of the data to use for testing.
        seed (int, optional): Random seed for reproducibility. Default is 0.

    Returns:
        tuple: (tr_dataset, val_dataset, ts_dataset)
            - tr_dataset (list): List of (video_path, label) tuples for the training set.
            - val_dataset (list): List of (video_path, label) tuples for the validation set.
            - ts_dataset (list): List of (video_path, label) tuples for the test set.
    """

    vid_paths = np.array(list(vid_dataset))
    vid_labels = np.array(list(vid_dataset.values()))
    vid_groups = np.array([_group_number(path) for path in vid_paths])

    print("Splitting train/validation/test datasets....")
    tr_val_idx, ts_idx = _group_split_indices(
        vid_paths,
        vid_labels,
        vid_groups,
        ts_ratio,
        seed,
    )
    val_ratio = 1 - tr_ratio - ts_ratio
    val_wt = val_ratio / (tr_ratio + val_ratio)
    tr_idx, val_idx = _group_split_indices(
        vid_paths[tr_val_idx],
        vid_labels[tr_val_idx],
        vid_groups[tr_val_idx],
        val_wt,
        seed,
    )
    return (
        _samples_at_indices(vid_paths, vid_labels, tr_val_idx[tr_idx]),
        _samples_at_indices(vid_paths, vid_labels, tr_val_idx[val_idx]),
        _samples_at_indices(vid_paths, vid_labels, ts_idx),
    )


def _group_number(vid_path):
    """Read the UCF50 group ID that must remain within a single dataset split."""
    video_name = os.path.basename(vid_path)
    group_number = re.search(r"_g(\d+)_c\d+$", video_name)
    if group_number is None:
        raise ValueError(f"Cannot find a UCF50 group ID in: {video_name}")
    return int(group_number.group(1))


def _group_split_indices(paths, labels, groups, test_size, seed):
    """Return the same seeded group shuffle indices for either split stage."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(splitter.split(paths, labels, groups=groups))


def _samples_at_indices(paths, labels, indices):
    """Collect video/label pairs in the original split index order."""
    return list(zip(paths[indices], labels[indices]))


def collate_fn_rnn(batch):
    """
    Collate function for RNN-based models.

    Handles variable-length video sequences by padding them to the length of the longest sequence
    in the batch. Each sample in the batch is expected to be a tuple (video_frames, label),
    where video_frames is a tensor of shape (T, C, H, W). The function returns a padded tensor
    of video frames with shape (batch_size, max_T, C, H, W) and a tensor of labels.

    Args:
        batch (list): List of samples, each as (video_frames, label).

    Returns:
        tuple: (padded_imgs, labels_tensor)
            - padded_imgs (Tensor): Padded tensor of video frames.
            - labels_tensor (Tensor): Tensor of labels.
    """
    # Unzip the batch into image tensors and labels
    imgs_batch, label_batch = list(zip(*batch))

    # Record real lengths BEFORE adding padding.
    lengths = torch.tensor(
        [len(clip) for clip in imgs_batch],
        dtype=torch.long,
    )

    if (lengths == 0).any():
        raise ValueError("A video has no frames; check preprocessing.")

    # Pad the video frame tensors along the time dimension (T)
    # Resulting shape: (batch_size, max_T, C, H, W)
    padded_imgs = pad_sequence(imgs_batch, batch_first=True)

    # Convert labels to a tensor
    labels_tensor = torch.tensor(label_batch)

    return padded_imgs, labels_tensor, lengths
