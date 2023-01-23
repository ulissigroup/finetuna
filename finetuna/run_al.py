import os
from pymongo import MongoClient

from ase.io import Trajectory
from ase.optimize.bfgs import BFGS
from ase.optimize.sciopt import SciPyFminCG
from ase.calculators.vasp import Vasp
from ase.calculators.emt import EMT
from vasp_interactive import VaspInteractive
from ase.db import connect
from ase.calculators.socketio import SocketIOCalculator
from ase.calculators.espresso import Espresso

from finetuna.atomistic_methods import Relaxation
from finetuna.offline_learner.offline_learner import OfflineActiveLearner
from finetuna.utils import calculate_surface_k_points
from finetuna.online_learner.online_learner import OnlineLearner
from finetuna.online_learner.delta_learner import DeltaLearner

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
    initial_index = config["links"].get("initial_index", 0)
    initial_structure = initial_traj[initial_index]
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

    dbname = (
        str(config["links"]["ml_potential"])
        + "_"
        + str(initial_structure.get_chemical_formula())
        + "_oal"
    )
    oal_initial_structure = initial_structure

    # begin setting up parent calc
    parent_str = config["links"].get("parent_calc", "vasp")
    # calculate kpts
    kpts = calculate_surface_k_points(initial_structure)
    # declare parent calc
    if parent_str == "vasp":
        if "kpts" not in config["vasp"]:
            config["vasp"]["kpts"] = kpts
        parent_calc = Vasp(**config["vasp"])
    elif parent_str == "vasp_interactive":
        parent_calc = VaspInteractive(**config["vasp"])
        if "kpts" not in config["vasp"]:
            config["vasp"]["kpts"] = kpts
    elif parent_str == "emt":
        parent_calc = EMT()
    elif parent_str == "espresso":
        espresso_config = {
            "command": f"mpirun -np 4 /opt/qe-7.0/bin/pw.x -in espresso.pwi --ipi unix:UNIX > espresso.pwo",
            "pseudopotentials": {
                key: f"{key}.UPF"
                for key, value in initial_traj[0].symbols.formula._count.items()
            },
            "tstress": True,
            "tprnfor": True,
            "kpts": kpts,
            "input_data": {
                "control": {
                    "verbosity": "high",
                    "calculation": "scf",
                    "disk_io": "nowf",
                },
                "system": {
                    "input_dft": "PBE",
                    "occupations": "smearing",
                    "smearing": "mv",
                    "degauss": 0.01,
                    "ecutwfc": 60,
                },
                "electrons": {
                    "electron_maxstep": 200,
                    "mixing_mode": "local-TF",
                    "conv_thr": 1e-8,
                },
            },
            "pseudo_path": "/home/jovyan/working/espresso_stuff/gbrv_pseudopotentials",
        }
        for key, value in config.get("espresso", {}).items():
            espresso_config[key] = value
        os.environ["ESPRESSO_PSEUDO"] = espresso_config.pop("pseudo_path")
        espresso = Espresso(**espresso_config)
        parent_calc = SocketIOCalculator(
            espresso,
            **config.get(
                "socket",
                {
                    "unixsocket": "unix",
                },
            ),
        )

    # declare base calc (if path is given)
    if "base_calc" in config:
        if (
            "model_path" in config["base_calc"]
            and "checkpoint_path" in config["base_calc"]
        ):
            base_calc = OCPCalculator(
                config_yml=config["base_calc"]["model_path"],
                checkpoint=config["base_calc"]["checkpoint_path"],
            )
        elif "checkpoint_path" in config["base_calc"]:
            base_calc = OCPCalculator(
                checkpoint=config["base_calc"]["checkpoint_path"],
            )

    # use given ml potential class
    potential_class = config["links"].get("ml_potential", "flare")
    # declare ml calc
    if potential_class == "ft_en":
        from finetuna.ml_potentials.finetuner_ensemble_calc import FinetunerEnsembleCalc

        ml_potential = FinetunerEnsembleCalc(
            checkpoint_paths=config["ocp"]["checkpoint_path_list"],
            mlp_params=config.get("finetuner", {}),
        )
    elif potential_class == "ft":
        from finetuna.ml_potentials.finetuner_calc import FinetunerCalc

        ml_potential = FinetunerCalc(
            checkpoint_path=config["ocp"]["checkpoint_path"],
            mlp_params=config.get("finetuner", {}),
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
