import os
from pymongo import MongoClient

from ase.io import Trajectory
from ase.optimize.bfgs import BFGS
from ase.optimize.sciopt import SciPyFminCG
from ase.calculators.vasp import Vasp
from ase.calculators.emt import EMT
from vasp_interactive import VaspInteractive
from ase.db import connect

from al_mlp.atomistic_methods import Relaxation
from al_mlp.offline_learner.offline_learner import OfflineActiveLearner
from al_mlp.utils import calculate_surface_k_points
from al_mlp.online_learner.online_learner import OnlineLearner
from al_mlp.online_learner.delta_learner import DeltaLearner
from al_mlp.online_learner.warm_start_learner import WarmStartLearner

from al_mlp.ml_potentials.flare_pp_calc import FlarePPCalc
from al_mlp.ml_potentials.flare_calc import FlareCalc
from al_mlp.ml_potentials.flare_ocp_descriptor_calc import FlareOCPDescriptorCalc
from al_mlp.ml_potentials.ocpd_gp_calc import OCPDGPCalc
from al_mlp.ml_potentials.ocpd_nn_calc import OCPDNNCalc

from ocpmodels.common.relaxation.ase_utils import OCPCalculator


def do_between_learner_and_run(learner, mongo_db):
    """
    boiler plate stuff to do between starting the learner and starting the run
    """

    if os.path.exists("dft_calls.db"):
        os.remove("dft_calls.db")

    if mongo_db is not None:
        with open("runid.txt", "a") as f:
            f.write(str(learner.logger.mongo_wrapper.run_id) + "\n")


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
        replay_method = config["relaxation"]["replay_method"]
        maxstep = config["relaxation"]["maxstep"]
    elif optimizer_str == "CG":
        optimizer_alg = SciPyFminCG
        replay_method = False
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
        replay_traj=replay_method,
        max_parent_calls=config["relaxation"]["max_parent_calls"],
        online_ml_fmax=config["learner"]["fmax_verify_threshold"],
        check_final=config["relaxation"].get("check_final", False),
    )

    return oal_relaxation


def active_learning(config):
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
    parent_str = config["links"].get("parent_calc", "vasp")
    if parent_str == "vasp":
        parent_calc = Vasp(**config["vasp"])
    elif parent_str == "vasp_interactive":
        parent_calc = VaspInteractive(**config["vasp"])
    elif parent_str == "emt":
        parent_calc = EMT()

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
    elif potential_class == "pyflare":
        ml_potential = FlareCalc(
            config.get("pyflare", {}), [initial_structure] + images
        )
    elif potential_class == "flare_ocp_descriptor":
        ml_potential = FlareOCPDescriptorCalc(
            model_path=config["ocp"]["model_path"],
            checkpoint_path=config["ocp"]["checkpoint_path"],
            flare_params=config.get("pyflare", {}),
            initial_images=[initial_structure] + images,
        )
    elif potential_class == "ocpd_gp":
        ml_potential = OCPDGPCalc(
            model_path=config["ocp"]["model_path"],
            checkpoint_path=config["ocp"]["checkpoint_path"],
            gp_params=config.get("gp", {}),
        )
    elif potential_class == "ocpd_nn":
        ml_potential = OCPDNNCalc(
            initial_structure,
            model_path=config["ocp"]["model_path"],
            checkpoint_path=config["ocp"]["checkpoint_path"],
            nn_params=config.get("nn", {}),
        )

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

    return learner.info
