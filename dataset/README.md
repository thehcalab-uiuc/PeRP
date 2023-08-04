# Dataset

Scripts and files in this directory are used to simulate and collect a trajectory dataset.

The dataset is constructed by simulating driver instruction following via offsets from the PC policies.
See code for details.

## Requirements

- [SUMO](https://eclipse.dev/sumo/)
- Python Environment for the project.

## Directory Structure

The directory contains the following files:

- `traj_collector.py`: script to simulate and save the trajectories.
- `preprocess.py`: Script to segment trajectories into smaller sizes
- `run.sh`: Bash script that runs the above python scripts and creates the dataset with a 80-20 split.

## Creating a dataset

Run the `run.sh` script. Modifying the parameters in `run.sh` will result in the creation of different number of samples.
The script can be modified to create trajectories with different:

- lengths
- driver instruction offsets
- number of samples
- train/val split
