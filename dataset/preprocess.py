#!/usr/bin/env python

__author__ = "Aamir Hasan"
__version__ = "0.1"
__email__ = "hasanaamir215@gmail.com; aamirh2@illinois.edu"

"""
Pre-processes the simulated sumo data to be used by the dataloader.
"""

from pathlib import Path
from typing import Dict, List
import pickle
from os.path import exists
from os import makedirs, listdir
from argparse import ArgumentParser


def process(
    load_path: Path,
    save_path: Path,
    title_prefix: str,
    time_period: int = 20,
    overlap: int = 0,
):
    """_Processes the data file to extract trajectories._

    Args:
        load_path (Path): _Path to the input npy array of shape (num_vehicles, timesteps, state_dim)_
        save_path (Path): _Path to the directory to save the trajectories_
        title_prefix (str): _Prefix to save the trajectories_
        time_period (int, optional): _Length of the trajectories to extract_. Defaults to 20.
        overlap (int, optional): _overlap between two consecutive trajectories_. Defaults to 0.
    """
    if not exists(save_path):
        makedirs(save_path)

    with open(load_path, "rb") as file:
        run_data = pickle.load(file)
    data = run_data["traj"]
    label = run_data["label"]

    count = 0

    # state_size can be used later to determine which of the state variables should be retained
    time_steps, state_size = data.shape

    assert time_steps > time_period
    assert time_period > overlap

    n_trajectories = int((time_steps - time_period) / (time_period - overlap))

    # extract the trajectories for that time period
    for trajectory_num in range(n_trajectories):
        # save the trajectory to be loaded by the dataloader
        trajectory_start = trajectory_num * (time_period - overlap)
        trajectory = data[trajectory_start : trajectory_start + time_period, :]

        with open(
            save_path / f"{title_prefix}_{trajectory_num}.pkl", "wb"
        ) as output_file:
            pickle.dump({"trajectory": trajectory, "class": label}, output_file)
            count += 1

    print(f"Generated {count} trajectories!")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--input_path",
        type=str,
        default="../sumo/collect_sim_dataset/data/run_0/traj_out/traj_0.pkl",
        help="Path to the trajectory pkl file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./dataset/train/0",
        help="Path to the directory to store the output folder",
    )
    parser.add_argument(
        "--title_prefix",
        type=str,
        default="",
        help="prefix to save trajectory files with",
    )
    parser.add_argument(
        "--overlap", type=int, default=0, help="overlap between two trajectory"
    )
    parser.add_argument(
        "--time_period", type=int, default=20, help="Length of the output trajectories"
    )
    args = parser.parse_args()

    path = Path(args.input_path)
    process(
        path,
        save_path=Path(args.output_path),
        title_prefix=args.title_prefix,
        overlap=args.overlap,
        time_period=args.time_period,
    )
