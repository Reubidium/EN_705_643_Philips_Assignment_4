"""
Module: models.py

This module defines the LRCN (Long-term Recurrent Convolutional Network) model for video
classification. The LRCN model combines a 2D CNN backbone (e.g., ResNet) for spatial feature
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
    
    This model uses a ResNet backbone as a 2D CNN to extract spatial features from each video frame.
    An LSTM network is then used to model the temporal dynamics across the sequence of frame features.
    Dropout is applied before the final fully-connected layer that produces class logits.

    Args:
        hidden_size (int): Number of features in the hidden state of the LSTM.
        n_layers (int): Number of recurrent layers in the LSTM.
        dropout_rate (float): Dropout rate applied before the final classification layer.
        n_classes (int): Number of output classes.
        pretrained (bool, optional): If True, uses a ResNet model pretrained on ImageNet. Default is True.
        cnn_model (str, optional): Specifies the ResNet variant to use as the backbone.
                                   Options: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'.
                                   Default is 'resnet34'.
    
    Raises:
        ValueError: If the specified cnn_model is not supported.
    """
    def __init__(self, hidden_size, n_layers, dropout_rate, n_classes, pretrained=True, cnn_model='resnet34', fine_tune_last_block=False):
        super(LRCN, self).__init__()

        # Set up the ResNet backbone as a 2D CNN feature extractor.
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
        else:
            raise ValueError('The input CNN backbone is not supported, please choose a valid ResNet variant.')

        # Retrieve the number of features output by the CNN's original fully-connected layer.
        num_features = base_cnn.fc.in_features
        
        # Replace the original fc layer with an identity mapping so that raw features are returned.
        base_cnn.fc = Identity()
        self.base_model = base_cnn

        # Optionally freeze the early layers of the CNN and only fine-tune the last block (layer4) along with the LSTM and classifier.
        self.fine_tune_last_block = fine_tune_last_block
        if self.fine_tune_last_block:
            self.base_model.requires_grad_(False)
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
        
        For each time step (frame), the CNN backbone extracts features. These features are then
        passed through the LSTM sequentially. The output from the last time step is then passed
        through dropout and the final fully-connected layer to produce the class logits.

        Args:
            x (Tensor): Input tensor of shape (batch_size, time_steps, channels, height, width).
            lengths (Tensor): A tensor containing the actual lengths of each sequence in the batch.

        Returns:
            Tensor: Output logits for each sample in the batch with shape (batch_size, n_classes).
        """
        bs, ts, c, h, w = x.shape  # batch_size, time_steps, channels, height, width
        
        # Process the first frame separately to initialize the LSTM hidden and cell states.
        idx = 0
        y = self.base_model(x[:, idx])
        out, (hn, cn) = self.rnn(y.unsqueeze(1))
        outputs = [out]
        
        # Iterate over the remaining frames, feeding each frame's features into the LSTM.
        for idx in range(1, ts):
            y = self.base_model(x[:, idx])
            out, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))
            outputs.append(out)

        # Concatenate the outputs from all time steps along the time dimension.
        out = torch.cat(outputs, dim=1)  # [batch, time, hidden]
        lengths = lengths.to(out.device)
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
        super().train(mode)

        if mode and self.fine_tune_last_block:
            # Use stored BatchNorm statistics throughout the CNN.
            self.base_model.eval()

            # Allow the final CNN block's statistics to update.
            self.base_model.layer4.train()

        return self
