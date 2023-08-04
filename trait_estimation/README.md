# Trait Estimation

Scripts and files in this directory are used to train and evaluate the driver trait inference model.
Please refer to the [project webpage](sites.google.com/illinois.edu/perp) for an overview.

## Requirements

- Python Environment for the project

## Directory Structure

This directory contains the following files:

- `dataloader.py`: defines the dataloader for
- `model.py`: Defines the TraitVAE model
- `train.py`: Trains and evaluates the TraitVAE model
- `vae_utils.py`: Utility functions for plotting

## Training a Model

Run the following command in your terminal after activating the project environment:

```bash
python train.py \
    --epochs 1 \
    --learning_rate 0.0001 \
    --validation_freq 1 \
    --encoder_dim 32 \
    --decoder_dim 32 \
    --latent_dim 2 \
    --batch_size 16 \
    --state_dim 3 \
    --train_set_path "../dataset/data/offset_2.5/train" \
    --val_set_path "../dataset/data/offset_2.5/val" \
    --save_path "./save" \
    --kl_beta 0.0001 \
    --reconstruction_beta 1
```

Vary the parameters as needed to train different models.
The `train_set_path` and `val_set_path` arguments can also take lists of directories that contain the preprocessed trajectories for mixing and matching.
