#!/usr/bin/env python

__author__ = "Aamir Hasan"
__version__ = "1.0"
__email__ = "hasanaamir215@gmail.com; aamirh2@illinois.edu"

"""_Parses the train log and generates a CSV with the losses._
"""

from pathlib import Path
from argparse import ArgumentParser
from os.path import exists
from os import makedirs, listdir
import re
import csv
from typing import Dict, List, Tuple
import yaml
import pandas as pd
import pdb

def get_hyperparams(folder_path: Path, col_names: list) -> List:
    """_Processes the folder to extract the meta data_

    Args:
        folder_path (Path): _Path to the folder with the logs._
        col_names (list): _Columns names to extract_

    Returns:
        List: _List of the columns extracted and the number of epochs in training_
    """
    with open(folder_path / "config.yaml", 'rb') as config_file:
        config = yaml.load(config_file, Loader=yaml.BaseLoader)
        ret_list = [config[col_name] for col_name in col_names]
    return ret_list

def get_trends(folder_path: Path, trend_cols, eval=False) -> List[float]:
    """_Extracts the losses from the train log_

    Args:
        folder_path (Path): _Path to the folder with the train logs_
        epochs (int): _Number of epochs during training_

    Returns:
        List[float]: _Training losses_
    """
    try:
        results_file = pd.read_csv(folder_path / ("results.csv" if eval else "train_results.csv"))
        ret = []
        for metric in trend_cols:
            ret += results_file[metric].tolist()
        return ret
    except:
        return []
    

def parse_args() -> Dict:
    """_Parses the arguments to the executable_

    Returns:
        _dict_: _Dictionary of all the arguments passed in_
    """    
    parser = ArgumentParser()

    parser.add_argument('--load_path', 
        type=str, 
        default='./results/no_trait_infernce',
        help='Path to the directory with the saved trajectories')

    parser.add_argument('--csv_save_path', 
        type=str, 
        default='.',
        help='Path to the save the results csv')

    parser.add_argument('--save_name', 
        type=str, 
        default='rewards_no_trait',
        help='Name of the saved csv')

    parser.add_argument('--eval', 
        action='store_true', 
        default=False,
        help='Results are in eval mode.')

    args = parser.parse_args()
    
    args.load_path = Path(args.load_path)
    args.csv_save_path = Path(args.csv_save_path)

    return args


def main():
    args = parse_args()

    if not exists(args.csv_save_path):
        makedirs(args.csv_save_path)

    with open(args.csv_save_path / f"{args.save_name}.csv", 'w') as f:
        writer = csv.writer(f, delimiter=',')
        if args.eval:
            col_names = ['pcp_path']
            line_placeholders = ['reward trendline', 'speed trendline',  'emissions trendline',  'fuel trendline']
            # line_placeholders = []
            trend_metrics = ['reward_mean', 'speed_rl', 'speed_rl_std', 'co2_emission', 'fuel', 'collisions']
        else:
            col_names = ['pcp_path', 'lr', 'beta_speed', 'beta_error']
            line_placeholders = ['reward trendline', 'speed trendline',  'emissions trendline',  'fuel trendline', 'policy loss trendline']
            trend_metrics = ['reward_mean', 'speed', 'co2_emission', 'fuel', 'collisions', 'policy_loss']
        
        trend_cols = []
        for metric in trend_metrics:
            if args.eval:
                trend_cols += [f"{metric}_{i}" for i in range(100)]
            else:
                trend_cols += [f"{metric}_{i}" for i in range(100)]
        writer.writerow(['folder_name'] + col_names + line_placeholders + trend_cols)

        # go through all folders in directory to get metadata
        for folder_name in listdir(args.load_path):
            hyperparams = get_hyperparams(args.load_path / folder_name, col_names)
            trends = get_trends(args.load_path / folder_name, trend_metrics, eval=args.eval)
            writer.writerow([folder_name] + hyperparams + ['' for metric in line_placeholders] + trends)

    # Report
    print("Done!")

if __name__ == "__main__":
    main()
