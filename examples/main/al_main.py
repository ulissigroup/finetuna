import os
import yaml
from pymongo import MongoClient
import argparse

from ase.io import Trajectory
from ase.optimize.bfgs import BFGS
from ase.optimize.sciopt import SciPyFminCG
from ase.calculators.vasp import Vasp
from ase.db import connect

from al_mlp.atomistic_methods import Relaxation
from al_mlp.offline_learner.offline_learner import OfflineActiveLearner
from al_mlp.utils import calculate_surface_k_points
from al_mlp.online_learner.online_learner import OnlineLearner
from al_mlp.online_learner.delta_learner import DeltaLearner
from al_mlp.online_learner.warm_start_learner import WarmStartLearner

from al_mlp.ml_potentials.flare_pp_calc import FlarePPCalc

from ocpmodels.common.relaxation.ase_utils import OCPCalculator

# possibly remove these later when we switch to only OCPCalculator
# from al_mlp.base_calcs.ocp_model import OCPModel
# from experimental.zitnick.models import spinconv_grad11


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-yml", required=True, help="Path to the config file")
    return parser


def do_between_learner_and_run(learner, mongo_db):
    """
    boiler plate stuff to do between starting the learner and starting the run
    """

    if os.path.exists("dft_calls.db"):
        os.remove("dft_calls.db")

    if mongo_db is not None:
        with open("runid.txt", "a") as f:
            f.write(str(learner.mongo_wrapper.run_id) + "\n")


def run_relaxation(
    oal_initial_structure,
    config,
    learner,
    dbname,
    mongo_db,
):
    do_between_learner_and_run(learner, mongo_db)

    optimizer_str = config["relaxation"].get("optimizer", "BFGS")

    if optimizer_str == "BFGS":
        optimizer_alg = BFGS
        replay_traj_bool = True
        maxstep = config["relaxation"]["maxstep"]
    elif optimizer_str == "CG":
        optimizer_alg = SciPyFminCG
        replay_traj_bool = False
        maxstep = None
    else:
        ValueError("Invalid optimizer name (" + optimizer_str + ") provided")

    oal_relaxation = Relaxation(
        oal_initial_structure,
        optimizer_alg,
        fmax=config["relaxation"]["fmax"],
        steps=config["relaxation"]["steps"],
        maxstep=maxstep,
    )

    oal_relaxation.run(
        learner,
        filename=dbname,
        replay_traj=replay_traj_bool,
        max_parent_calls=config["relaxation"]["max_parent_calls"],
    )

    return oal_relaxation


def main(args):
    config_yml = args.config_yml
    basedir = config_yml[: config_yml.rindex("/") + 1]
    os.chdir(basedir)

    config = yaml.safe_load(open(config_yml, "r"))
    initial_traj = Trajectory(config["links"]["traj"])
    initial_structure = initial_traj[0]
    images = []

    if "images_path" in config["links"] and config["links"]["images_path"] is not None:
        with connect(config["links"]["images_path"]) as pretrain_db:
            for row in pretrain_db.select():
                image = row.toatoms(attach_calculator=False)
                image.calc.implemented_properties.append("energy")
                image.calc.implemented_properties.append("forces")
                images.append(image)

    mongo_db = None
    if "MONGOC" in os.environ:
        mongo_string = os.environ["MONGOC"]
        mongo_db = MongoClient(mongo_string)["al_db"]
    else:
        print("no recording to mongo db")

    # calculate kpts
    if "kpts" not in config["vasp"]:
        config["vasp"]["kpts"] = calculate_surface_k_points(initial_structure)

    dbname = "flare_" + str(initial_structure.get_chemical_formula()) + "_oal"
    oal_initial_structure = initial_structure

    # declare parent calc
    parent_calc = Vasp(**config["vasp"])

    # declare base calc (if path is given)
    if "ocp" in config:
        if "model_path" in config["ocp"] and "checkpoint_path" in config["ocp"]:
            base_calc = OCPCalculator(
                config_yml=config["ocp"]["model_path"],
                checkpoint=config["ocp"]["checkpoint_path"],
            )
        elif "checkpoint_path" in config["ocp"]:
            base_calc = OCPCalculator(
                checkpoint=config["ocp"]["checkpoint_path"],
            )

    # use given ml potential class
    potential_class = config["links"].get("ml_potential", "flare")
    if potential_class == "flare":
        # declare ml calc
        ml_potential = FlarePPCalc(config["flare"], [initial_structure] + images)

    # use given learner class
    learner_class = config["links"].get("learner_class", "online")
    if learner_class == "online":
        # declare online learner
        learner = OnlineLearner(
            config["learner"],
            images,
            ml_potential,
            parent_calc,
            mongo_db=mongo_db,
            optional_config=config,
        )

        run_relaxation(
            oal_initial_structure,
            config,
            learner,
            dbname,
            mongo_db,
        )

    elif learner_class == "delta":
        # declare online learner
        learner = DeltaLearner(
            config["learner"],
            images,
            ml_potential,
            parent_calc,
            base_calc=base_calc,
            mongo_db=mongo_db,
            optional_config=config,
        )

        run_relaxation(
            oal_initial_structure,
            config,
            learner,
            dbname,
            mongo_db,
        )

    elif learner_class == "warmstart":
        # declare warmstart online learner
        learner = WarmStartLearner(
            config["learner"],
            images,
            ml_potential,
            parent_calc,
            base_calc=base_calc,
            mongo_db=mongo_db,
            optional_config=config,
        )

        run_relaxation(
            oal_initial_structure,
            config,
            learner,
            dbname,
            mongo_db,
        )

    elif learner_class == "offline":
        # set atomistic method
        config["learner"]["atomistic_method"] = {}
        config["learner"]["atomistic_method"]["initial_traj"] = config["links"]["traj"]
        config["learner"]["atomistic_method"]["fmax"] = config["relaxation"]["fmax"]
        config["learner"]["atomistic_method"]["steps"] = config["relaxation"]["steps"]
        config["learner"]["atomistic_method"]["maxstep"] = config["relaxation"][
            "maxstep"
        ]

        # declare learner
        learner = OfflineActiveLearner(
            learner_params=config["learner"],
            training_data=images,
            ml_potential=ml_potential,
            parent_calc=parent_calc,
            base_calc=base_calc,
            mongo_db=mongo_db,
            optional_config=config,
        )

        # do boilerplate stuff
        do_between_learner_and_run(learner, mongo_db)

        # start run
        learner.learn()

    else:
        print("No valid learner class given")

    # close parent_calc (if it needs to be closed, i.e. VaspInteractive)
    if hasattr(parent_calc, "close"):
        parent_calc.close()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)