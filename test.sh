#!/bin/bash
python run.py \
    --ckpt ./runs/20261028_205151_248095/models/best_model_wts.pt \
    --cnn_backbone convnext_tiny \
    --rnn_hidden_size 100 \
    --rnn_n_layers 1 \
    --bidirectional \
    --rnn_layer_norm \
    --n_classes 50 \
    --fr_per_vid 16 \
    --batch_size 16 \
    --mode eval
