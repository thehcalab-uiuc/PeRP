# PeRP

Scripts and files in this directory are used to train and evaluate PeRP and the other baselines.
Please refer to the [project webpage](sites.google.com/illinois.edu/perp) for an overview of the baselines and PeRP.

The code presented here is derived from [FLOW](https://github.com/mit-wu-lab/automatic_vehicular_control) and [Piecewise constant policies](https://ieeexplore.ieee.org/document/9564789).

## Requirements

- Python Environment for the project
- [SUMO](https://eclipse.dev/sumo/)

## Directory Structure

This directory contains the following directories and files (note that only important files are presented here. Files integral to FLOW are not expanded upon):

- `pexps`:
  - `env.py`: Defines the environment super class for integration with SUMO
  - `exp.py`: Defines the RL training workflow and experiment definitions
  - `ring.py`: Defines the exact environment used for training. Includes reward function and workflow for each step in simulation.
- `run.sh`: Script to run PeRP and other baseline trainings
- `eval.sh`: Script to run PeRP and other baseline evaluation
- `result_parser.py`: Helper script to parse csvs produced by the simulation.

Additional details about each file can be found in inline documentation.

## Usage

Two scripts `run.sh` and `eval.sh` are provided to make training and evaluation easy.

### Instructions

- Load project environment
- (Optional) Train custom PC policy and DTI
  - If you train custom PC policy, use `create_config_yaml.sh` to ensure that PeRP training runs smoothly
- Train PeRP or other baseline model using `run.sh`
- Run `create_config_yaml.sh` to ensure that the config file is parsed correctly
- Evaluate PeRP or other baseline model using `eval.sh`
- Use `results_parser.py` to curate evaluation statistics as needed. Please read the file for usage instructions.

#### Running PeRP

Ensure that the following flags are set as instructed to train and evaluate PeRP:

- `simple_trait_inference` set to `False`
- `vae_trait_inference` set to `True` and provide a valid `trait_inference_path`
- `perp` set to `True`

#### Running RP

Ensure that the following flags are set as instructed to train and evaluate RP:

- `simple_trait_inference` set to `False`
- `vae_trait_inference` set to `False`
- `perp` set to `True`

#### Running TA-RP

Ensure that the following flags are set as instructed to train and evaluate TA-RP:

- `simple_trait_inference` set to `True`
- `vae_trait_inference` set to `False`
- `perp` set to `True`

#### Evaluating PCP

Ensure that the following flags are set as instructed to evaluate PCP:

- `simple_trait_inference` set to `False`
- `vae_trait_inference` set to `False`
- `perp` set to `False`

#### Rendering Simulation

Ensure that `render` flag is set to `True` if you want to see the SUMO simulation rendered.
