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
    Dropout is applied before the final fully-connected layer that produces class logits.

    Args:
        hidden_size (int): Number of features in the hidden state of the LSTM.
        n_layers (int): Number of recurrent layers in the LSTM.
        dropout_rate (float): Dropout rate applied before the final classification layer.
        n_classes (int): Number of output classes.
        pretrained (bool, optional): Load ImageNet CNN weights. Default is True.
        cnn_model (str, optional): ResNet variant or 'convnext_tiny'. Default is 'resnet34'.
        fine_tune_last_block (bool): Train only the final CNN stage and, for ConvNeXt, its final feature normalization. The LSTM and video classifier stay trainable.
    
    Raises:
        ValueError: If the specified cnn_model is not supported.
    """
    def __init__(self, hidden_size, n_layers, dropout_rate, n_classes, pretrained=True, cnn_model='resnet34', fine_tune_last_block=False):
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

        # Define the LSTM to process the sequence of frame features.
        self.rnn = nn.LSTM(num_features, hidden_size, n_layers, batch_first=True)
        
        # Define dropout for regularization.
        self.dropout = nn.Dropout(dropout_rate)
        
        # Final fully-connected layer to produce logits for each class.
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x, lengths):
        """
        Forward pass for the LRCN model.
        
        The input tensor x is expected to have the shape:
            (batch_size, time_steps, channels, height, width)
        
        For each time step, the CNN extracts features only from real frames so that padding
        cannot affect BatchNorm statistics. Features are passed through the LSTM sequentially.
        Only real-frame outputs contribute to the mean used by dropout and the classifier.

        Args:
            x (Tensor): Input tensor of shape (batch_size, time_steps, channels, height, width).
            lengths (Tensor): Positive lengths of each sequence, no greater than x.size(1).

        Returns:
            Tensor: Output logits for each sample in the batch with shape (batch_size, n_classes).
        """
        bs = x.size(0)
        # Ignore any tail timesteps that contain only padding.
        ts = int(lengths.max().item())
        lengths = lengths.to(x.device)
        
        # Process the first frame separately to initialize the LSTM hidden and cell states.
        idx = 0
        y = self.base_model(x[:, idx])
        out, (hn, cn) = self.rnn(y.unsqueeze(1))
        outputs = [out]
        
        # Exclude padded rows before the CNN; masking after the LSTM is too late for BatchNorm.
        for idx in range(1, ts):
            active = lengths > idx
            active_features = self.base_model(x[active, idx])
            # Restore batch order without breaking gradients to the real frames.
            y = active_features.new_zeros((bs, active_features.size(1)))
            y[active] = active_features
            out, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))
            outputs.append(out)

        # Concatenate the outputs from all time steps along the time dimension.
        out = torch.cat(outputs, dim=1)  # [batch, time, hidden]
        positions = torch.arange(out.size(1), device=out.device)
        # True for real frames, False for padding.
        valid = positions[None, :] < lengths[:, None]
        out = out.masked_fill(~valid.unsqueeze(-1), 0)
        out = out.sum(dim=1) / lengths[:, None]
        
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
