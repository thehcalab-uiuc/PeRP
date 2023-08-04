#!/usr/bin/env python

__author__ = "Aamir Hasan"
__version__ = "1.0"
__email__ = "hasanaamir215@gmail.com; aamirh2@illinois.edu"

"""_Utils file_"""

from typing import Dict, List
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from os import makedirs
import numpy as np
from pathlib import Path
import json
import pdb


class NamedDict:
    def __init__(self, attributes_dict):
        self.__dict__ = attributes_dict


def get_config(config_path: Path) -> Dict:
    """_Loads the TraitVAE Config arguments_

    Args:
        config_path (_Path_): _path to the config file_

    Returns:
        Dict: _The arguments from the file_
    """
    args = {}
    with open(config_path, "r") as config_file:
        train_args = json.load(config_file)
        args["encoder_dim"] = train_args["encoder_dim"]
        args["decoder_dim"] = train_args["decoder_dim"]
        args["latent_dim"] = train_args["latent_dim"]
        args["state_dim"] = train_args["state_dim"]
        args["n_encoder_layers"] = train_args["n_encoder_layers"]
        args["n_decoder_layers"] = train_args["n_decoder_layers"]

        args["train_set_path"] = train_args["train_set_path"]
        args["train_set_size"] = train_args["train_set_size"]

    args = NamedDict(args)
    return args


LABELS = ["-2", "-1", "0", "1", "2"]
COLORS = [0, 1, 2, 3, 4]


def plot_latent_space(
    latent_space_reps: np.array,
    labels: np.array,
    title: str,
    save_path: Path,
    val_start: int = -1,
):
    """_Plots the latent vectors passed in shaded with the class labels_

    Args:
        latent_space_reps (np.array): _The latent vectors of shape (N ,2)_
        labels (np.array): _The labels of shape (N)_
        title (str): _The title for the plot_
        save_path (Path): _Path to save the plot at_
        val_start (int, optional): _The index where the validation samples start_. Defaults to -1.
    """
    plt.figure(figsize=(8, 8))
    # If there are no validation points, just plot the latent space points
    if val_start == -1:
        label_colors = [COLORS[int(label)] for label in labels]
        sc = plt.scatter(
            latent_space_reps[:, 0],
            latent_space_reps[:, 1],
            c=label_colors,
            cmap=plt.cm.tab10,
        )
        handles, _ = sc.legend_elements(prop="colors", num=COLORS)
        label_names = LABELS[:5]
    else:
        # first plot the train points which are almost transparent
        train_colors = [COLORS[int(label)] for label in labels[:val_start]]
        sc = plt.scatter(
            latent_space_reps[:val_start, 0],
            latent_space_reps[:val_start, 1],
            c=train_colors,
            cmap=plt.cm.tab10,
            alpha=0.1,
        )
        train_handles, _ = sc.legend_elements(prop="colors", num=COLORS)

        # then plot the validation points
        val_colors = [COLORS[int(label)] for label in labels[val_start:]]
        sc = plt.scatter(
            latent_space_reps[val_start:, 0],
            latent_space_reps[val_start:, 1],
            c=val_colors,
            marker="x",
            s=100,
            cmap=plt.cm.tab10,
        )
        val_handles, _ = sc.legend_elements(prop="colors", num=COLORS)
        handles = train_handles + val_handles

        label_names = LABELS[:5]

        if -1 in labels:
            label_names = label_names + ["CitySim"]

    plt.legend(
        handles,
        label_names,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
        ncol=5,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_loss_curves(
    losses: List[float], epochs: List[int], title: str, save_path: Path
):
    """_Plots the loss curves for the input losses_

    Args:
        losses (list): _The losses_
        epochs (list): _Epochs_
        title (str): _Title for plot_
        save_path (Path): _Path to save figure_
    """
    if len(losses) > 0:
        plt.figure(figsize=(8, 8))
        plt.plot(epochs, losses)
        plt.title(title)
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
