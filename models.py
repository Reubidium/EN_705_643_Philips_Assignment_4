"""
Module: models.py

This module defines the LRCN (Long-term Recurrent Convolutional Network) model for video
classification. The LRCN model combines a ResNet or ConvNeXt 2D CNN for spatial feature
extraction from individual frames with an LSTM to capture temporal dynamics across frames.
An additional fully-connected layer is used to output the final class predictions.

Classes:
    Identity: A helper module that returns the input unchanged. It is used to replace the fully-connected
              layer of the ResNet backbone.
    LRCN: The main model that integrates the CNN backbone, an LSTM, dropout regularization, and a final
          fully-connected layer to produce class logits.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import models

class Identity(nn.Module):
    """
    A placeholder identity operator that is argument-insensitive.
    
    This module is used to replace the fully-connected (fc) layer in the ResNet backbone,
    effectively making the backbone output the raw features before classification.
    
    Example:
        >>> identity = Identity()
        >>> output = identity(input_tensor)
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """
        Forward pass that returns the input as is.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: The same tensor x.
        """
        return x

class LRCN(nn.Module):
    """
    LRCN (Long-term Recurrent Convolutional Network) for video classification.
    
    This model uses a ResNet or ConvNeXt backbone to extract features from each video frame.
    An LSTM network is then used to model the temporal dynamics across the sequence of frame features.
    Optional LayerNorm normalizes each recurrent timestep before temporal pooling.
    Optional temporal attention learns weights over genuine recurrent timesteps.
    Dropout is applied before the final fully-connected layer that produces class logits.

    Args:
        hidden_size (int): Number of hidden features per LSTM direction.
        n_layers (int): Number of recurrent layers in the LSTM.
        dropout_rate (float): Dropout rate applied before the final classification layer.
        n_classes (int): Number of output classes.
        pretrained (bool, optional): Load ImageNet CNN weights. Default is True.
        cnn_model (str, optional): ResNet variant or 'convnext_tiny'. Default is 'resnet34'.
        fine_tune_last_block (bool): Train only the final CNN stage and, for ConvNeXt, its final feature normalization. The LSTM and video classifier stay trainable.
        bidirectional (bool): Process the complete frame-feature sequence in both
            directions. Default False preserves the unidirectional architecture.
        rnn_layer_norm (bool): Normalize the recurrent feature vector at each
            timestep before pooling. Default False preserves older checkpoints.
        temporal_attention (bool): Use learned temporal pooling instead of the
            genuine-frame mean. Default False preserves older checkpoints.
    
    Raises:
        ValueError: If the specified cnn_model is not supported.
    """
    def __init__(
        self, hidden_size, n_layers, dropout_rate, n_classes, pretrained=True,
        cnn_model='resnet34', fine_tune_last_block=False, bidirectional=False,
        rnn_layer_norm=False, temporal_attention=False,
    ):
        super(LRCN, self).__init__()

        # Set up the selected 2D CNN feature extractor.
        if cnn_model == 'resnet18':
            base_cnn = models.resnet18(pretrained=pretrained)
        elif cnn_model == 'resnet34':
            base_cnn = models.resnet34(pretrained=pretrained)
        elif cnn_model == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            base_cnn = models.resnet50(weights=weights)
        elif cnn_model == 'resnet101':
            base_cnn = models.resnet101(pretrained=pretrained)
        elif cnn_model == 'resnet152':
            base_cnn = models.resnet152(pretrained=pretrained)
        elif cnn_model == 'convnext_tiny':
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            base_cnn = models.convnext_tiny(weights=weights)
        else:
            raise ValueError(f'Unsupported CNN backbone: {cnn_model}')

        if cnn_model == 'convnext_tiny':
            # Preserve the pretrained normalization and flattening before the classifier.
            num_features = base_cnn.classifier[-1].in_features
            base_cnn.classifier[-1] = nn.Identity()
        else:
            num_features = base_cnn.fc.in_features
            base_cnn.fc = Identity()
        self.base_model = base_cnn
        self.cnn_model = cnn_model

        # Optionally freeze early CNN stages while fine-tuning the final stage.
        self.fine_tune_last_block = fine_tune_last_block
        if self.fine_tune_last_block:
            self.base_model.requires_grad_(False)
            if self.cnn_model == 'convnext_tiny':
                self.base_model.features[-1].requires_grad_(True)
                self.base_model.classifier.requires_grad_(True)
            else:
                self.base_model.layer4.requires_grad_(True)

        # A BiLSTM needs the complete sequence to learn from later frames.
        self.bidirectional = bidirectional
        self.rnn = nn.LSTM(
            num_features, hidden_size, n_layers, batch_first=True,
            bidirectional=bidirectional,
        )
        
        num_directions = 2 if bidirectional else 1
        recurrent_features = hidden_size * num_directions
        # Normalize features within each timestep, independently of batch size.
        self.rnn_norm = nn.LayerNorm(recurrent_features) if rnn_layer_norm else nn.Identity()

        # Define dropout for regularization and the final class-logit layer.
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(recurrent_features, n_classes)

        # Create the scorer last to preserve initialization of existing layers.
        self.attention = nn.Sequential(
            nn.Linear(recurrent_features, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        ) if temporal_attention else None

    def forward(self, x, lengths):
        """
        Forward pass for the LRCN model.
        
        The input tensor x is expected to have the shape:
            (batch_size, time_steps, channels, height, width)
        
        At each time step, only genuine frames enter the CNN. Their features are
        collected in chronological order and passed through the LSTM in one
        complete, packed sequence. Packing excludes padding from both recurrent
        directions. Optional LayerNorm acts on each timestep's feature vector.
        Only genuine normalized outputs contribute to mean or attention pooling.
        Attention weights are normalized over time independently for each video.

        Args:
            x (Tensor): Input tensor of shape (batch_size, time_steps, channels, height, width).
            lengths (Tensor): Integer lengths of each sequence, between 1 and x.size(1).

        Returns:
            Tensor: Output logits for each sample in the batch with shape (batch_size, n_classes).

        Raises:
            ValueError: If lengths do not contain one valid positive length per video.
        """
        bs = x.size(0)
        # PyTorch's packing operation requires sequence lengths on the CPU.
        lengths_cpu = lengths.detach().to(device='cpu', dtype=torch.long)
        if lengths_cpu.ndim != 1 or lengths_cpu.numel() != bs or bs == 0:
            raise ValueError('lengths must contain one positive length per video')
        if (lengths_cpu < 1).any() or (lengths_cpu > x.size(1)).any():
            raise ValueError('lengths must be between 1 and the number of input frames')
        ts = int(lengths_cpu.max().item())
        lengths_device = lengths_cpu.to(x.device)

        # Retain the original per-timestep CNN batches and exclude padded images.
        features = []
        for idx in range(ts):
            active = lengths_device > idx
            active_features = self.base_model(x[active, idx])
            # Restore batch order without breaking gradients to the real frames.
            frame_features = active_features.new_zeros((bs, active_features.size(1)))
            frame_features[active] = active_features
            features.append(frame_features)

        sequence = torch.stack(features, dim=1)  # [batch, time, CNN features]
        packed_features = pack_padded_sequence(
            sequence, lengths_cpu, batch_first=True, enforce_sorted=False,
        )
        packed_outputs, _state = self.rnn(packed_features)
        out, _lengths = pad_packed_sequence(packed_outputs, batch_first=True)
        out = self.rnn_norm(out)

        # Unpacking restores the original batch order, including unsorted lengths.
        positions = torch.arange(out.size(1), device=out.device)
        # True for real frames, False for padding.
        valid = positions[None, :] < lengths_device[:, None]
        # Mask after normalization so its learned bias cannot reintroduce padding.
        out = out.masked_fill(~valid.unsqueeze(-1), 0)
        if self.attention is None:
            out = out.sum(dim=1) / lengths_device[:, None]
        else:
            scores = self.attention(out).squeeze(-1)
            # Exclude padding before softmax so it receives exactly zero weight.
            scores = scores.masked_fill(~valid, float('-inf'))
            weights = torch.softmax(scores, dim=1)
            out = (out * weights.unsqueeze(-1)).sum(dim=1)
        
        # Apply dropout to the summary
        out = self.dropout(out)
        
        # Pass the final output through the fully-connected layer to get class logits.
        out = self.fc(out)
        return out

    def train(self, mode=True):
        """Control frozen CNN stages while enabling the trainable stage's regularization."""
        super().train(mode)

        if mode and self.fine_tune_last_block:
            # Keep frozen stages deterministic and preserve ResNet BatchNorm statistics.
            self.base_model.eval()
            if self.cnn_model == 'convnext_tiny':
                # ConvNeXt uses LayerNorm; enable stochastic depth in its final stage.
                self.base_model.features[-1].train()
                self.base_model.classifier.train()

        return self
