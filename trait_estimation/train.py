#!/usr/bin/env python

__author__ = "Aamir Hasan"
__version__ = "1.0"
__email__ = "hasanaamir215@gmail.com; aamirh2@illinois.edu"

"""_Trains a VAE for Driver Trait Estimation_.
"""

from argparse import ArgumentParser
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from model import TraitVAE
from dataloader import TraitEstimationDataLoader
import datetime
from pathlib import Path
from os.path import exists, join
from os import makedirs
import json
import logging
from sys import stdout
from vae_utils import plot_latent_space, plot_loss_curves


def process(
    model: TraitVAE,
    train_set: torch.utils.data.DataLoader,
    val_set: torch.utils.data.DataLoader,
    config: dict,
):
    """_Trains the model on the datasets passed in._

    Args:
        model (TraitVAE): _Model to be trained_
        train_set (torch.utils.data.Dataloader): _The training dataset_
        val_set (torch.utils.data.Dataloader): _The validation dataset_
        config (dict): _Config passed into the trainer_
    """
    losses_train = []
    losses_val = []
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), config.learning_rate)

    total_train_start = datetime.datetime.now()
    for epoch in range(1, config.epochs + 1):
        # Train for this epoch
        logging.info(f"Beginning training for epoch {epoch}/{config.epochs}")

        train_start = datetime.datetime.now()
        train_loss, train_loss_kl, train_loss_recon = train_epoch(
            model, train_set, criterion, optimizer, epoch, config
        )
        train_time = datetime.datetime.now() - train_start

        # log the training losses
        losses_train.append(train_loss)
        config.writer.add_scalar("Loss/train", train_loss, epoch)
        config.writer.add_scalar("Loss/train", train_loss, epoch)
        config.writer.add_scalar("Loss/train_KL", train_loss_kl, epoch)
        config.writer.add_scalar("Loss/train_reconstruction", train_loss_recon, epoch)
        logging.info(f"\nDone training for epoch {epoch}/{config.epochs}")

        logging.info(
            "[%d/%d]Training: Total Loss: %.4f\tKL Loss: %.4f\tReconstruction Loss: %.4f\tTrain time: %d s"
            % (
                epoch,
                config.epochs,
                train_loss,
                train_loss_kl,
                train_loss_recon,
                train_time.seconds,
            )
        )

        # Validate for this epoch
        if epoch % config.validation_freq == 0:
            logging.info(f"Beginning validation for epoch {epoch}/{config.epochs}")
            val_start = datetime.datetime.now()
            with torch.no_grad():
                val_loss, val_loss_kl, val_loss_recon = train_epoch(
                    model, val_set, criterion, optimizer, epoch, config, val=True
                )
            val_time = datetime.datetime.now() - val_start

            # log the validation losses
            losses_val.append(val_loss)
            config.writer.add_scalar("Loss/val", val_loss, epoch)
            config.writer.add_scalar("Loss/val_KL", val_loss_kl, epoch)
            config.writer.add_scalar("Loss/val_recon", val_loss_recon, epoch)

            logging.info(
                "[%d/%d]Validation: Total Loss: %.4f\tKL Loss: %.4f\tReconstruction Loss: %.4f\tValidation time: %d s"
                % (
                    epoch,
                    config.epochs,
                    val_loss,
                    val_loss_kl,
                    val_loss_recon,
                    val_time.seconds,
                )
            )

            logging.info(f"Done validating for epoch {epoch}/{config.epochs}")
        logging.info("\n")

    # plot the loss curves
    plot_loss_curves(
        losses_train,
        np.arange(1, config.epochs + 1),
        "Training Loss",
        config.save_path / "train_loss.jpg",
    )
    plot_loss_curves(
        losses_val,
        np.arange(1, config.epochs + 1, config.validation_freq),
        "Validation Loss",
        config.save_path / "val_loss.jpg",
    )

    total_train_time = datetime.datetime.now() - total_train_start
    logging.info(
        f"Done training for {config.epochs} epochs, \
        total training time: {total_train_time.seconds} s"
    )


def train_epoch(
    model: TraitVAE,
    dataset: torch.utils.data.DataLoader,
    criterion,
    optimizer,
    epoch: int,
    config: dict,
    val: bool = False,
) -> Tuple[List[float], List[float], List[float]]:
    """_Runs one epoch for the trainer_

    Args:
        model (TraitVAE): _The model to be used_
        dataset (torch.utils.data.Dataloader): _Data to run the epoch on_
        criterion (_Loss Function_): _Loss function to use for the reconstructions_
        optimizer (_Optimizer_): _Optimizer to be used_
        epoch (int): _Current epoch number_
        config (dict): _Config passed into the trainer_
        val (bool, optional): _If this is a validation epoch_. Defaults to False.

    Returns:
        _List[float, float, float]_: _total error, kl divergence error, and reconstruction loss_
    """
    total_error = 0
    total_KL_div = 0
    total_reconstruction = 0

    for i, data in enumerate(dataset):
        model.zero_grad()

        real_data = data[0].float().to(config.device)

        # run data through model
        encoded, z_mean, z_log_var, decoded = model(real_data)
        # calculate KL divergence
        kl_divergence = -0.5 * torch.sum(
            1 + z_log_var - z_mean**2 - torch.exp(z_log_var), axis=1
        )
        kl_divergence = kl_divergence.mean()
        # calculate reconstruction loss
        reconstruction_error = criterion(decoded, real_data)

        # total loss
        loss = (
            config.reconstruction_beta * reconstruction_error
            + config.kl_beta * kl_divergence
        )

        # perform step
        if not val:
            loss.backward()
            optimizer.step()

        logging.info(
            "[%d/%d][%d/%d]\tLoss: %.4f\tKL Divergence: %.4f\tReconstruction Loss: %.4f"
            % (
                epoch,
                config.epochs,
                i,
                len(dataset),
                loss.item(),
                kl_divergence,
                reconstruction_error,
            )
        )

        # plot the latent space if validating
        if val:
            plot_latent_space(
                encoded.detach().cpu(),
                data[1].detach().cpu(),
                f"Epoch {epoch}",
                config.save_path / f"val_{epoch}.jpg",
            )

        total_error += loss.item()
        total_KL_div += kl_divergence
        total_reconstruction += reconstruction_error

    total_error /= len(dataset)
    total_KL_div /= len(dataset)
    total_reconstruction /= len(dataset)

    return total_error, total_KL_div, total_reconstruction


def parse_args() -> Dict:
    """_Parse the arguments passed into the file._

    Returns:
        dict: _Dictionary containing all the arguments passed in._
    """
    parser = ArgumentParser()

    parser.add_argument(
        "--latent_dim", type=int, default=2, help="Dimension of the latent vector"
    )
    parser.add_argument(
        "--encoder_dim", type=int, default=2, help="Dimension of the encoder"
    )
    parser.add_argument(
        "--decoder_dim", type=int, default=2, help="Dimension of the decoder"
    )
    parser.add_argument(
        "--n_encoder_layers",
        type=int,
        default=1,
        help="Number of layers in the encoder",
    )
    parser.add_argument(
        "--n_decoder_layers",
        type=int,
        default=1,
        help="Number of layers in the decoder",
    )
    parser.add_argument(
        "--state_dim", type=int, default=3, help="Dimension of the input states"
    )
    parser.add_argument(
        "--traj_len", type=int, default=20, help="Length of the trajectories"
    )

    parser.add_argument(
        "--reconstruction_beta",
        type=float,
        default=1.0,
        help="Weight of reconstruction term in loss",
    )
    parser.add_argument(
        "--kl_beta", type=float, default=0.01, help="Weight of kl divergence in loss"
    )

    parser.add_argument(
        "--epochs", type=int, default=2, help="Number of epochs to training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate for training"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )

    parser.add_argument(
        "--validation_freq",
        type=int,
        default=1,
        help="Epoch frequency at which to run validation code",
    )

    parser.add_argument(
        "--seed", type=int, default=20956, help="Random seed for all randomization"
    )

    parser.add_argument(
        "--train_set_path",
        type=str,
        nargs="+",
        default="./dataset/all_combined/traj_processed_leader",
        help="Path to the directory to load the train dataset from",
    )
    parser.add_argument(
        "--train_set_size",
        type=int,
        default=-1,
        help="random n samples of the train set to train on. -1 takes the whole dataset.",
    )
    parser.add_argument(
        "--val_set_size",
        type=int,
        default=-1,
        help="random n samples of the val set to train on. -1 takes the whole dataset.",
    )
    parser.add_argument(
        "--val_set_path",
        type=str,
        nargs="+",
        default="./dataset/val_all_combined/traj_processed_leader",
        help="Path to the directory to load the validation dataset from",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="./save",
        help="Path to the directory to save the models and trajectories",
    )

    parser.add_argument(
        "--show_log_on_out",
        action="store_true",
        help="flag if the log output should be shown on the screen",
    )

    parser.add_argument(
        "--val_only",
        action="store_true",
        help="flag if the a model should be loaded and validated only",
    )
    parser.add_argument(
        "--load_path",
        type=str,
        default="./save/exp_0/model_TraitVAE.pt",
        help="Path to where to load the model from",
    )

    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.val_only:
        args.validation_freq = 1
        args.epochs = 1

    # making new save directory
    i = 0

    while exists(args.save_path + f"/exp_{i}"):
        i += 1
    args.save_path = args.save_path + f"/exp_{i}"
    makedirs(args.save_path)

    # Saving configs
    with open(Path(args.save_path) / "args.json", "w") as f:
        json.dump(args.__dict__, f, indent=2)

    print(f"Saving all files to {args.save_path}")

    args.train_set_path = [Path(train_path) for train_path in args.train_set_path]
    args.val_set_path = [Path(val_path) for val_path in args.val_set_path]
    args.save_path = Path(args.save_path)
    args.load_path = Path(args.load_path)

    args.writer = SummaryWriter(args.save_path / "tb_log")

    return args


def main():
    config = parse_args()

    torch.manual_seed(config.seed)

    logging.basicConfig(
        filename=config.save_path / "train.log",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.NOTSET,
    )
    if config.show_log_on_out:
        logging.getLogger().addHandler(logging.StreamHandler(stdout))
    print(f"Logs written to {config.save_path}/train.log.")

    # instantiate model
    model = TraitVAE(config).to(config.device)

    # Load the model if needed
    if config.val_only:
        model.load_state_dict(torch.load(config.load_path)["state_dict"])
        model.eval()
        logging.info(f"Loaded model from {config.load_path}")

    # load the data feeders
    train_feeder = TraitEstimationDataLoader(config.train_set_path)
    val_feeder = TraitEstimationDataLoader(config.val_set_path)

    # if needed, change the sizes of the datafeeders
    if config.train_set_size > 0 and config.train_set_size < len(train_feeder):
        train_feeder = torch.utils.data.Subset(
            train_feeder,
            np.random.randint(0, len(train_feeder), size=(config.train_set_size)),
        )
    if config.val_set_size > 0 and config.val_set_size < len(val_feeder):
        val_feeder = torch.utils.data.Subset(
            val_feeder,
            np.random.randint(0, len(val_feeder), size=(config.val_set_size)),
        )

    train_set = torch.utils.data.DataLoader(
        dataset=train_feeder,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_set = torch.utils.data.DataLoader(
        dataset=val_feeder,
        batch_size=len(val_feeder),
        num_workers=0,
    )

    logging.info(
        f"Datasets loaded with {len(train_set)} and \
        {len(val_set)} samples for train and validation respectively \
        with batch size: {config.batch_size}"
    )

    logging.info(f"Starting Training for {config.epochs} Epochs.")

    logging.getLogger("PIL").setLevel(logging.WARNING)

    process(model, train_set, val_set, config)

    config.writer.flush()

    # saving the model
    torch.save(
        {
            "state_dict": model.state_dict(),
        },
        config.save_path / f"model_{model.__class__.__name__}.pt",
    )
    print("Done.")


if __name__ == "__main__":
    main()
