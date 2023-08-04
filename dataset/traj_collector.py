#!/usr/bin/env python

__author__ = "Neeloy Chakraborty; Aamir Hasan"
__version__ = "1"
__email__ = "neeloyc2@illnois.edu; hasanaamir215@gmail.com; aamirh2@illinois.edu"

"""
Pre-processes the simulated sumo data to be used by the dataloader.
"""

# https://sumo.dlr.de/docs/sumo.html
# https://sumo.dlr.de/docs/TraCI.html
# https://sumo.dlr.de/docs/TraCI/Interfacing_TraCI_from_Python.html
# https://sumo.dlr.de/docs/Tutorials/index.html#traci_tutorials
# https://sumo.dlr.de/docs/Definition_of_Vehicles%2C_Vehicle_Types%2C_and_Routes.html
# https://sumo.dlr.de/docs/TraCI/Vehicle_Value_Retrieval.html
# https://arxiv.org/pdf/1704.05566.pdf
# https://github.com/eclipse/sumo/issues/7694

import os
import sys
import pickle
from shutil import copy2
import random
from copy import deepcopy
import pdb
import re
import logging

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
    print("imported sumo")
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import numpy as np
import traci
import sumolib
from argparse import ArgumentParser

from pathlib import Path
import glob

sys.path.append("../perp/pexps")
from ring import Ring
from ut import FFN

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    """_Parses the arguments passed into the script_

    Returns:
        _args_: _The arguments passed in as an object_
    """
    p = ArgumentParser(
        prog="SUMO Trajectory Data Collector",
        description="Generates a randomized route.xml file for SUMO vehicles for a given map, and runs a SUMO simulator to collect trajectories.",
    )

    p.add_argument(
        "--map_name",
        type=str,
        metavar="MAP",
        default="ring",
        required=False,
        help="Name of base map to load.",
    )

    p.add_argument(
        "--output_path",
        type=str,
        default="./data",
        help="path for the output directory",
    )

    p.add_argument(
        "--pcp_path",
        type=str,
        default=".,/saved/pcp_models/hc_20",
        help="path for the PCP policy",
    )

    p.add_argument(
        "--run_name",
        type=str,
        metavar="RUN",
        default="default",
        required=False,
        help="Name of subfolder to create under data dir.",
    )

    p.add_argument(
        "--num_runs",
        type=int,
        metavar="NRUN",
        default=1,
        required=False,
        help="Number of different runs to execute.",
    )

    p.add_argument(
        "--num_veh",
        type=int,
        metavar="VEH",
        default=40,
        help="Number of vehicles to spawn in each environment.",
    )

    p.add_argument(
        "--driver_offset",
        type=float,
        metavar="DOFF",
        default=2.5,
        help="how much the first offset for the driver speed is.",
    )

    p.add_argument(
        "--time",
        type=int,
        metavar="T",
        default=1000,
        help="Max time limit of each simulator.",
    )

    p.add_argument(
        "--warmup",
        type=int,
        metavar="W",
        default=600,
        help="Warm up time for each simulation.",
    )

    p.add_argument(
        "--step_length",
        type=float,
        metavar="L",
        default=0.1,
        help="Real world seconds simulated per time step.",
    )

    p.add_argument(
        "--render", action="store_true", default=False, help="Render SUMO GUI."
    )

    p.add_argument(
        "--verbose", action="store_true", default=False, help="Show output on terminal."
    )

    args = p.parse_args()

    args.warmup = max(1, args.warmup)

    return args


def setup(args):
    """_Sets up the different folders and other logistics needed for the runs_

    Args:
        args (_Object_): _The arguments for the script_

    Returns:
        _args_: _the arguments for the script_
    """
    args.ring_map_path = Path("../saved/map_files/ring.net.xml")

    # Create subfolder for map
    args.run_dir = Path(args.output_path) / args.run_name
    os.makedirs(args.run_dir, exist_ok=True)

    args.xml_dir = args.run_dir / "xml"
    os.makedirs(args.xml_dir, exist_ok=False)

    args.traj_dir = args.run_dir / "traj_out"
    os.makedirs(args.traj_dir, exist_ok=False)

    logging.basicConfig(
        filename=args.run_dir / "collector.log",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.NOTSET,
    )
    if args.verbose:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    print(f"Logs written to {args.run_dir}/collector.log.")

    # Save arguments to current data run directory
    args.data_args_file = args.run_dir / "data_args.pickle"
    with open(args.data_args_file, "wb") as handle:
        pickle.dump(args, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return args


def generate_sumo_files(args):
    """_Generates the sumo files required for the simulation_

    Args:
        args (_Object_): _Arguments for the script_
    """
    # Copy original map to current data run dir
    copy2(args.ring_map_path, args.xml_dir / "map.net.xml")

    # Dict with vType definitions for different behavior types
    behavior_vtype_dict = {
        "human": '<vType id="human" color="white" carFollowModel="IDM" minGap="2.5" tau="1" accel="2.6" decel="4.5" maxSpeed="35" speedFactor="1.0" speedDev="0.1" impatience="0.5" sigma="0.2"/>',
        "rl": '<vType id="rl" color="red" carFollowModel="IDM" minGap="2.5" tau="1" accel="2.6" decel="4.5" maxSpeed="35" speedFactor="1.0" speedDev="0.1" impatience="0.5" sigma="0.2"/>',
    }

    # Fill in route xml with random vehicle behavior ordering
    curr_route_path = args.xml_dir / "route.rou.xml"
    with open(curr_route_path, "w") as handle:
        handle.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        handle.write(
            '<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">\n'
        )
        handle.write("\t<!-- VTypes -->\n")
        for beh_name in behavior_vtype_dict.keys():
            handle.write("\t{}\n".format(behavior_vtype_dict[beh_name]))
        handle.write("\t<!-- Routes -->\n")
        handle.write('\t<route id="r_0" edges="-3 -0 -1 -2 -3"/>\n')
        handle.write('\t<route id="r_1" edges="-1 -2 -3 -0 -1"/>\n')
        handle.write("\t<!-- Other additionals -->\n")
        handle.write('\t<rerouter id="rr_0" edges="-3">\n')
        handle.write('\t\t<interval begin="0.00" end="1000000000.00">\n')
        handle.write('\t\t\t<routeProbReroute id="r_0"/>\n')
        handle.write("\t\t</interval>\n")
        handle.write("\t</rerouter>\n")
        handle.write('\t<rerouter id="rr_1" edges="-1">\n')
        handle.write('\t\t<interval begin="0.00" end="1000000000.00">\n')
        handle.write('\t\t\t<routeProbReroute id="r_1"/>\n')
        handle.write("\t\t</interval>\n")
        handle.write("\t</rerouter>\n")
        handle.write("\t<!-- Vehicles, persons and containers (sorted by depart) -->\n")

        curr_beh_choices = ["rl", "human"]
        for veh_id in range(args.num_veh):
            curr_beh = curr_beh_choices[0] if veh_id == 0 else curr_beh_choices[1]
            if veh_id <= 18:
                route = 0
            else:
                route = 1
            handle.write(
                '\t<vehicle id="{}" type="{}" depart="0.00" departLane="0" departPos="free" route="r_{}"/>\n'.format(
                    veh_id, curr_beh, route
                )
            )
        handle.write("</additional>\n")

    args.rl_id = "0"

    sumo_binary = sumolib.checkBinary("sumo-gui" if args.render else "sumo")
    args.sumoCmd = [
        sumo_binary,
        "--net-file",
        f"{args.xml_dir / 'map.net.xml'}",
        "--additional-files",
        f"{args.xml_dir / 'route.rou.xml'}",
        "--step-length",
        f"{args.step_length}",
        "--time-to-teleport",
        "-1",
        "--collision.action",
        "remove",
        "--collision.check-junctions",
        "true",
        "--max-depart-delay",
        "0.5",
        "--random",
        "true",
        "--start",
        "true",
        "--delay",
        "200" if args.render else "0",
    ]
    if args.render:
        args.sumoCmd = args.sumoCmd + ["--quit-on-end"]

    return args


def load_policy(args):
    """_Loads the piecewise constant policy_

    Args:
        args (_Object_): _Arguments for the script_

    Returns:
        _PCPolicy_: _The piecewise constant policy_
    """
    numbers = re.compile(r"(\d+)")

    def numericalSort(value):
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    # load pcp config
    pcp_config = Ring(args.pcp_path).setdefaults(
        use_critic=True,
        perp=False,
        _n_obs=3,
    )
    args.pcp_config = pcp_config

    # load the Piecewise constant policy
    pcp_model = pcp_config.get("model_cls", FFN)(pcp_config)
    best_model_path = max(glob.glob(f"{args.pcp_path}/models/*"), key=numericalSort)
    pcp_model.load_state_dict(torch.load(best_model_path, map_location=device)["net"])
    args.pcp_model = pcp_model.eval()
    logging.info(f"loaded PCP from {best_model_path}")

    return args


def start_sumo(args):
    """_Starts the sumo simulation_

    Args:
        args (_Object_): _Arguments for the script_
    """
    logging.info("Running " + " ".join(args.sumoCmd))
    traci.start(args.sumoCmd)
    logging.info("Started sumo")


def close_sumo():
    """_Ends the sumo simulation_"""
    traci.close()
    logging.info("Closed sumo")


def get_rl_obs(args):
    """_Retrieves the current timesteps observation for the "RL" vehicle_

    Args:
        args (_Object_): _Arguments for the script_

    Returns:
        _np.array_: _Observations at the current timestep_
    """
    veh_speed = traci.vehicle.getSpeed(args.rl_id)
    leader_id, leader_gap = traci.vehicle.getLeader(
        args.rl_id, args.pcp_config.circumference_max
    )

    leader_speed = traci.vehicle.getSpeed(leader_id)

    # update observation
    obs = [
        veh_speed / args.pcp_config.max_speed,
        leader_speed / args.pcp_config.max_speed,
        leader_gap / args.pcp_config.circumference_max,
    ]
    obs = np.clip(obs, 0, 1) * (1 - args.pcp_config.low) + args.pcp_config.low
    obs = obs.astype(np.float32)

    return obs


def main(args):
    # Setup the simulation
    args = setup(args)
    args = generate_sumo_files(args)
    args = load_policy(args)

    for run_num in range(args.num_runs):
        logging.info(f"Starting run {run_num + 1} / {args.num_runs}")
        start_sumo(args)

        step = 0
        vehicle_data = []

        driver_trait = random.sample(
            [
                -2 * args.driver_offset,
                -1 * args.driver_offset,
                0,
                1 * args.driver_offset,
                2 * args.driver_offset,
            ],
            1,
        )[0]
        logging.info(f"Driver Trait is {driver_trait}")

        logging.info("Starting warm up")

        # warmup to let the system start
        while step < args.warmup:
            traci.simulationStep()
            obs = get_rl_obs(args)
            step += 1
        logging.info("Done with warmup.\nStarting collection.")

        # let the pcp policy takeover
        while step < args.warmup + args.time:
            pcp_pred = args.pcp_model(
                torch.tensor(obs), value=False, policy=True, argmax=False
            )
            pcp_speed = (
                pcp_pred.action.item()
                / (args.pcp_config.n_actions - 1)
                * args.pcp_config.max_speed
            )

            for _ in range(args.pcp_config.hc_param):
                traci.simulationStep()

                # update observation
                obs = get_rl_obs(args)

                # perform action
                perform_action = pcp_speed + np.random.normal(driver_trait, 0.5)
                perform_action = np.clip(perform_action, 0, args.pcp_config.max_speed)
                logging.info(
                    f"Step {step}: PCP speed: {pcp_speed}. Driving at: {perform_action}"
                )
                traci.vehicle.slowDown(args.rl_id, perform_action, 1e-3)

                # save the observations to use later
                vehicle_data.append(obs)
                step += 1

        # end simulation
        close_sumo()

        # save the data from the run
        run_data = np.array(vehicle_data, dtype=float)
        run_save_path = args.traj_dir / f"traj_{run_num}.pkl"
        with open(run_save_path, "wb") as f:
            pickle.dump(
                {
                    "traj": run_data,
                    "label": int((driver_trait / args.driver_offset) + 2),
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        logging.info(f"Saved generated dataset to {run_save_path}")


if __name__ == "__main__":
    main(parse_args())
