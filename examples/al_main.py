import numpy as np
import os
import yaml
from pymongo import MongoClient
from dask.distributed import Client, LocalCluster
import argparse

from ase.io import Trajectory
from ase.optimize.bfgs import BFGS
from ase.calculators.vasp import Vasp

from al_mlp.atomistic_methods import Relaxation
from al_mlp.utils import calculate_surface_k_points
from al_mlp.ml_potentials.flare_pp_calc import FlarePPCalc
from al_mlp.online_learner.online_learner import OnlineLearner

from ocpmodels.common.utils import setup_imports
from al_mlp.ml_potentials.ocp_calc import OCPCalculator

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-yml", required=True, help="Path to the config file")
    return parser

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config_yml, "r"))
    initial_traj = Trajectory(config["links"]["traj"])
    initial_structure = initial_traj[0]
    images = []

    mongo_db = None
    if "MONGOC" in os.environ:
        mongo_string = os.environ["MONGOC"]
        mongo_db = MongoClient(mongo_db)["al_db"]
    else:
        print("no recording to mongo db")


    # calculate kpts
    config["vasp"]["kpts"] = calculate_surface_k_points(initial_structure)


    dbname = "flare_" + str(initial_structure.get_chemical_formula()) + "_oal"
    oal_initial_structure = initial_structure

    # declare parent calc
    parent_calc = Vasp(**config["vasp"])

    # declare ml calc
    ml_potential = OCPCalculator(config, pbc_graph=True, checkpoint=config["links"]["checkpoint_path"])

    # declare online learner
    learner = OnlineLearner(
        config["learner"],
        images,
        ml_potential,
        parent_calc,
        mongo_db=mongo_db,
    )

    if os.path.exists("dft_calls.db"):
        os.remove("dft_calls.db")

    if mongo_db is not None:
        with open("runid.txt", "a") as f:
            f.write(str(learner.mongo_wrapper.run_id) + "\n")

    oal_relaxation = Relaxation(
        oal_initial_structure,
        BFGS,
        fmax=config["relaxation"]["fmax"],
        steps=config["relaxation"]["steps"],
        maxstep=config["relaxation"]["maxstep"],
    )

    oal_relaxation.run(
        learner,
        filename=dbname,
        replay_traj=True,
        max_parent_calls=config["relaxation"]["max_parent_calls"],
    )

    if hasattr(parent_calc, "close"):
        parent_calc.close()
