#!/usr/bin/env python

__author__ = "Aamir Hasan"
__version__ = "0.1"
__email__ = "hasanaamir215@gmail.com; aamirh2@illinois.edu"

"""
Dataloader for driver trait estimation module.
"""

from pathlib import Path
from os import listdir
import pickle
from typing import List, Tuple
import torch
import torch.utils.data as data
import pdb
import numpy as np


class TraitEstimationDataLoader(data.Dataset):
    """_Dataloader for Trait Estimation models. The dataloader takes in any
    directory that has each sample as a .pkl file which contains the trajectory
    and label._
    """

    def __init__(self, load_path: List[Path]):
        """_Initializes the dataloader from the data in the load path._

        Args:
            load_path (_List[Path]_): _List of paths to the directory with the data files._
        """
        self.load_path = load_path
        self.inputs_list = []
        for data_path in load_path:
            # loading the list of all the files in the data directory
            self.inputs_list += [
                data_path / fname
                for fname in listdir(data_path)
                if fname.endswith(".pkl")
            ]

    def __getitem__(self, index: int) -> Tuple[np.array, int]:
        """_Retrieves a sample in the dataloader by index._

        Args:
            index (_int_): _index of the sample to return._

        Returns:
            _np.array(length, state_dim), int_: _The trajectory and the class it belongs to_
        """
        with open(self.inputs_list[index], "rb") as sample_file:
            input_data = pickle.load(sample_file)
        return input_data["trajectory"], int(input_data["class"])

    def __len__(self) -> int:
        """_summary_

        Returns:
            _int_: _Number of samples in the dataset._
        """
        return len(self.inputs_list)


if __name__ == "__main__":
    # load the feeder
    train_paths = [
        "../dataset/offset_2.5/train",
    ]
    feeder = TraitEstimationDataLoader([Path(path_i) for path_i in train_paths])

    # Feed the datafeeder into the dataloader
    dataset = torch.utils.data.DataLoader(
        dataset=feeder,
        batch_size=len(feeder),
        shuffle=True,
        num_workers=0,
    )

    # iterate through the dataset
    for batch in dataset:
        # checking the size of the batch
        print(batch[0].shape, batch[1].shape)
        all_speeds = batch[0].reshape(-1, 3)

        # Some dataset stats to help with normalization later
        print(
            f"Car Speed:\n\tmean:{all_speeds[:, 0].mean()}, std:{all_speeds[:, 0].std()}, min:{all_speeds[:, 0].min()}, max:{all_speeds[:, 0].max()}"
        )
        print(
            f"Distance:\n\tmean:{all_speeds[:, 1].mean()}, std:{all_speeds[:, 1].std()}, min:{all_speeds[:, 1].min()}, max:{all_speeds[:, 1].max()}"
        )
        print(
            f"Leader Speed:\n\tmean:{all_speeds[:, 2].mean()}, std:{all_speeds[:, 2].std()}, min:{all_speeds[:, 2].min()}, max:{all_speeds[:, 2].max()}"
        )

        # histogram to see distribution in the classes
        print(np.unique(batch[1], return_counts=True))

        # stop so that you can look at the data
        pdb.set_trace()
