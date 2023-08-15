# PeRP

This repository contains the code for our paper titled "PeRP: Personalized Residual Policies For Congestion Mitigation Through Co-operative Advisory Autonomy" in ITSC 2023.
For more details, please refer to the [project website](https://sites.google.com/illinois.edu/perp).

## Abstract

Intelligent driving systems can be used to mitigate congestion through simple actions, thus improving many socioeconomic factors such as commute time and gas costs. However, these systems assume precise control over autonomous vehicle fleets, and are hence limited in practice as they fail to account for uncertainty in human behavior. Piecewise Constant (PC) Policies address these issues by structurally modeling the likeness of human driving to reduce traffic congestion in dense scenarios to provide action advice to be followed by human drivers. However, PC policies assume that all drivers behave similarly. To this end, we develop a co-operative advisory system based on PC policies with a novel driver trait conditioned Personalized Residual Policy, PeRP. PeRP advises drivers to behave in ways that mitigate traffic congestion. We first infer the driver’s intrinsic traits on how they follow instructions in an unsupervised manner with a variational autoencoder. Then, a policy conditioned on the inferred trait adapts the action of the PC policy to provide the driver with a personalized recommendation. Our system is trained in simulation with novel driver modeling of instruction adherence. We show that our approach successfully mitigates congestion while adapting to different driver behaviors, with 4 to 22% improvement in average speed over baselines.

## Setup

1. Install Python3.8 (The code may work with other versions of Python, but 3.8 is highly recommended).
2. Import the following conda environment:

```bash
conda env create -n perp --file environment.yml
```

## Overview

The repository is divided into multiple subdirectories for every submodule.
Please refer to the `README.md` in each subdirectory for details on how to use the module.
This repository is organized as follows:

- The `dataset/` folder contains scripts to curate the dataset to train the trait estimation VAE model.
- The `trait_estimation/` folder contains scripts to train the trait estimation VAE model.
- The `perp/` folder contains scripts simulate and train PeRP policies.
- The `saved/` folder contains saved files such as map data and precomputed trait estimation, and PCP models.

## Running Instructions

- Install [SUMO](https://eclipse.dev/sumo/)
- Create conda environment using `environment.yml`.
- The `saved` directory contains PC policies and the trait inference model used in the paper.
  - To train a custom PC policy refer to [PCP_REPO](https://ieeexplore.ieee.org/document/9564789)
  - To train a custom Driver trait inference model:
    - Create a dataset using the instruction in `dataset/README.md`
    - Train a DTI model using the instructions in `trait_inference/README.md`
- Follow the instructions in `perp/README.md` to train a perp model.

- **NOTE:** All compute was performed on Ubuntu 18.04. *Using Linux is strongly encouraged.*

## Citation

If you find the code or the paper useful for your research, please cite our paper:

```bibtex
@inproceedings{hasan2023perp,
  title={{PeRP}: Personalized Residual Policies For Congestion Mitigation Through Co-operative Advisory Autonomy},
  author={Hasan, Aamir and Chakraborty, Neeloy and Chen, Haonan and Cho, Jung-Hoon and Wu, Cathy and Driggs-Campbell, Katherine},
  booktitle={IEEE International Conference on Intelligent Transportation Systems (ITSC)},
  year={2023}
}
```

## Credits

Other contributors:  

- [Neeloy Chakraborty](https://theneeloy.github.io/)
- [Haonan Chen](https://www.linkedin.com/in/haonan-chen-7a4339153/)
- [Jung-Hoon Cho](https://www.junghooncho.com/)
- [Cathy Wu](http://www.wucathy.com/)
- [Katherine Driggs-Campbell](http://krdc.web.illinois.edu/)

## Contact

If you have any questions or find any bugs, please feel free to open an issue or pull request.
