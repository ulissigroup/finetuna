#!/usr/bin/env python
import numpy as np
import os
import sys
import copy
import yaml
import ase
from ase.calculators.vasp.vasp import Vasp
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS
from vasp_interactive import VaspInteractive
from finetuna.ml_potentials.finetuner_ensemble_calc import FinetunerEnsembleCalc
from finetuna.online_learner.online_learner import OnlineLearner
from finetuna.atomistic_methods import parent_only_replay
import argparse


def main(args):
    vasp_interactive = VaspInteractive()
    vasp_interactive.read_incar(filename=args.path + "INCAR")
    vasp_interactive.read_kpoints(filename=args.path + "KPOINTS")
    vasp_interactive.read_potcar(filename=args.path + "POTCAR")

    initial_structure = ase.io.read(args.path + "POSCAR")

    yaml_file = open(args.config)
    parsed_yaml_file = yaml.load(yaml_file, Loader=yaml.FullLoader)
    learner_params = parsed_yaml_file["learner"]
    finetuner = parsed_yaml_file["finetuner"]
    optional_config = parsed_yaml_file.get("optional_config", None)

    pretrain_dataset = []
    ml_potential = FinetunerEnsembleCalc(
        model_classes=parsed_yaml_file["ocp"]["model_class_list"],
        model_paths=parsed_yaml_file["ocp"]["model_path_list"],
        checkpoint_paths=parsed_yaml_file["ocp"]["checkpoint_path_list"],
        mlp_params=finetuner,
    )

    with vasp_interactive as parent_calc:
        onlinecalc = OnlineLearner(
            learner_params,
            [],
            ml_potential,
            parent_calc,
            optional_config=optional_config,
        )
        initial_structure.calc = onlinecalc
        dyn = BFGS(
            initial_structure,
            trajectory="online_al.traj",
            maxstep=parsed_yaml_file["relaxation"].get("maxstep", None),
        )
        dyn.attach(parent_only_replay, 1, initial_structure.calc, dyn)
        dyn.run(
            fmax=parsed_yaml_file["relaxation"].get("fmax", 0.03),
            steps=parsed_yaml_file["relaxation"].get("steps", None),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "-path", type=str, default="", help="Path to the VASP input directory"
    )
    args = parser.parse_args()
    main(args)
