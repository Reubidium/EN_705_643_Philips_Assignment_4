"""Basic tests for the video classification pipeline."""

from test import test as evaluate_model

import pytest
import torch
from torch import nn

from models import LRCN
from video_datasets import sample_frame_indices, dataset_split, collate_fn_rnn


def test_uniform_frame_sampling():
    """Validation and test frames need to be ordered and span the video."""
    indices = sample_frame_indices(64, 16)

    assert len(indices) == 16
    assert indices == sorted(indices)
    assert indices[0] == 0
    assert indices[-1] == 63
    assert indices == sample_frame_indices(64, 16)
    print("test_uniform_frame_sampling")
    print(
        "Objective: Verify that evaluation frame sampling is ordered, deterministic, "
        "and spans the video."
    )
    print("Sampling 16 frames from a 64-frame video...")
    print(f"sample_frame_indices returned: {indices}")
    print(f"Selected {len(indices)} frames, from frame {indices[0]} to frame {indices[-1]}.")
    print("=" * 100)
    print()


def test_random_frame_sampling():
    """Training should select one frame from each group of four frames."""
    indices = sample_frame_indices(64, 16, random_sampling=True)

    print("test_random_frame_sampling")
    print(
        "Objective: Verify that training frame sampling selects one frame "
        "from each temporal segment."
    )
    print(f"sample_frame_indices returned: {indices}")
    print("=" * 100)
    print()

    assert len(indices) == 16
    for group, index in enumerate(indices):
        assert group * 4 <= index < (group + 1) * 4


def test_dataset_splits_do_not_share_groups():
    """Related videos must stay in the same split to prevent data leakage."""
    videos = {}
    for label in range(2):
        for group in range(1, 26):
            path = f"Action{label}/v_Action{label}_g{group:02d}_c01"
            videos[path] = label

    train, val, test = dataset_split(videos, 0.6, 0.2)
    train_groups = {path.split("_g")[1].split("_c")[0] for path, _ in train}
    val_groups = {path.split("_g")[1].split("_c")[0] for path, _ in val}
    test_groups = {path.split("_g")[1].split("_c")[0] for path, _ in test}

    print("test_dataset_splits_do_not_share_groups")
    print(
        "Objective: Verify that each related-video group appears in only one split "
        "to prevent data leakage."
    )
    print(
        f"dataset_split assigned {len(videos)} videos: {len(train)} train, "
        f"{len(val)} validation, {len(test)} test."
    )
    print(
        "Group overlap checks: "
        f"train/validation={not train_groups.isdisjoint(val_groups)}, "
        f"train/test={not train_groups.isdisjoint(test_groups)}, "
        f"validation/test={not val_groups.isdisjoint(test_groups)}."
    )
    print("=" * 100)
    print()

    assert train and val and test
    assert len(train) + len(val) + len(test) == len(videos)
    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)


def test_padding_keeps_original_lengths():
    """A shorter video should be padded without changing its recorded length."""
    short_clip = torch.ones(2, 3, 4, 4)     # 2 frames
    long_clip = torch.ones(4, 3, 4, 4)      # 4 frames

    frames, labels, lengths = collate_fn_rnn([(short_clip, 0), (long_clip, 1)])

    print("test_padding_keeps_original_lengths")
    print(
        "Objective: Verify that variable-length video clips are padded to the batch maximum "
        "while their original lengths are preserved."
    )
    print(f"Input clip shapes: short={tuple(short_clip.shape)}, long={tuple(long_clip.shape)}.")
    print(
        f"collate_fn_rnn returned frames with shape {tuple(frames.shape)} "
        f"and labels {labels.tolist()}."
    )
    print(
        f"Original clip lengths: {lengths.tolist()}; "
        f"padded frames in the short clip: {frames[0, 2:].shape[0]}."
    )
    print(f"Added frames are all zero: {torch.all(frames[0, 2:] == 0).item()}.")
    print("=" * 100)
    print()

    assert frames.shape == (2, 4, 3, 4, 4)
    assert labels.tolist() == [0, 1]
    assert lengths.tolist() == [2, 4]
    assert torch.all(frames[0, 2:] == 0)


def test_model_output_shape():
    """The model should return one score per class for each video."""
    model = LRCN(
        hidden_size=8,
        n_layers=1,
        dropout_rate=0.3,
        n_classes=50,
        pretrained=False,
        cnn_model="resnet50",
        bidirectional=True,
        rnn_layer_norm=True,
    )
    model.eval()
    videos = torch.randn(2, 2, 3, 32, 32)
    lengths = torch.tensor([2, 1])

    with torch.no_grad():
        output = model(videos, lengths)

    print("test_model_output_shape")
    print("Objective: Verify that the model returns one class score per class for each video.")
    print(f"Input videos have shape {tuple(videos.shape)} and sequence lengths {lengths.tolist()}.")
    print(
        f"Model returned output with shape {tuple(output.shape)}: {output.shape[0]} videos "
        f"and {output.shape[1]} class scores per video."
    )
    print("=" * 100)
    print()

    assert output.shape == (2, 50)


class EmptyModel(nn.Module):
    """Return supplied class scores so evaluation has a known answer."""

    def forward(self, inputs, _lengths):
        """Pass the example scores straight through."""
        return inputs


def test_evaluation_loss_and_accuracy():
    """Equal scores give 50% accuracy and a loss of about 0.693 for these labels."""
    scores = torch.zeros(2, 2)
    labels = torch.tensor([0, 1])
    lengths = torch.tensor([1, 1])
    batches = [(scores, labels, lengths)]
    metrics = {}

    _, predictions, accuracy = evaluate_model(EmptyModel(), batches, "cpu", metrics)

    print("test_evaluation_loss_and_accuracy")
    print(
        "Objective: Verify that evaluation computes the expected predictions, accuracy, "
        "and cross-entropy loss."
    )
    print(
        f"Input scores have shape {tuple(scores.shape)}; labels={labels.tolist()}, "
        f"sequence lengths={lengths.tolist()}."
    )
    print(
        f"Evaluation returned predictions {predictions}, accuracy {accuracy:.3f}, "
        f"and loss {metrics['loss']:.6f}."
    )
    print("Expected: predictions [0, 0], accuracy 0.500, and loss approximately 0.693147.")
    print("=" * 100)
    print()

    assert predictions == [0, 0]
    assert accuracy == 0.5
    assert metrics["loss"] == pytest.approx(0.693147, abs=1e-6)
