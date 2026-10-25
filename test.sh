#!/bin/bash
python run.py \
    --ckpt ./models/best_model_wts.pt \
    --n_classes 50 \
    --batch_size 16 \
    --mode eval
