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
from pymatgen.io.vasp.inputs import Kpoints, Incar
from vasp_interactive import VaspInteractive
from finetuna.ml_potentials.finetuner_ensemble_calc import FinetunerEnsembleCalc
from finetuna.online_learner.online_learner import OnlineLearner
from finetuna.atomistic_methods import parent_only_replay
import argparse
from importlib_resources import files


def main(args):
    # Initialize VASP interactive calculator with VASP input from the path provided
    print("------Initializing VASP Interactive Calculator------")
    vasp_interactive = VaspInteractive()
    incar = Incar.from_file(os.path.join(args.path, "INCAR"))
    kpoints = Kpoints.from_file(os.path.join(args.path, "KPOINTS"))
    vasp_inputs = {
        "kpts": kpoints.kpts[0],
    }
    for k in incar.keys():
        vasp_inputs[k.lower()] = incar[k]
        
    os.rename(os.path.join(args.path, "INCAR"), os.path.join(args.path, "INCAR_original"))
    os.rename(os.path.join(args.path, "KPOINTS"), os.path.join(args.path, "KPOINTS_original"))

    # Set convergence criteria as EDIFFG in VASP flag, default to 0.03 eV/A
    if -vasp_interactive.exp_params.get("ediffg") == 0:
        relax_fmax = 0.03
    else:
        relax_fmax = -vasp_interactive.exp_params.get("ediffg")

    # Read the initial structure
    initial_structure = ase.io.read(os.path.join(args.path, "POSCAR"))
    os.rename(os.path.join(args.path, "POSCAR"), os.path.join(args.path, "POSCAR_original"))

    # Parse the config file
    yaml_file = open(args.config)
    # Set VASP command
    parsed_yaml_file = yaml.load(yaml_file, Loader=yaml.FullLoader)
    # Set up learner, finetuner
    learner_params = parsed_yaml_file["learner"]
    learner_params["fmax_verify_threshold"] = relax_fmax

    finetuner = parsed_yaml_file["finetuner"]
    optional_config = parsed_yaml_file.get("optional_config", None)
    # Set up Finetuner calculator
    print("------Setting up Finetuner Calculator------")
    ml_potential = FinetunerEnsembleCalc(
        checkpoint_paths=[args.checkpoint],
        mlp_params=finetuner,
    )
    with VaspInteractive(**vasp_inputs) as parent_calc:
        parent_calc.read_potcar(filename=os.path.join(args.path, "POTCAR"))
        os.rename(os.path.join(args.path, "POTCAR"), os.path.join(args.path, "POTCAR_original"))

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
            trajectory=parsed_yaml_file["relaxation"].get("trajname", "online_al.traj"),
            maxstep=parsed_yaml_file["relaxation"].get("maxstep", None),
        )
        dyn.attach(parent_only_replay, 1, initial_structure.calc, dyn)

        print(
            f"------Starting Relaxation. Terminate when Fmax <{relax_fmax} eV/A------"
        )
        dyn.run(
            fmax=relax_fmax,
        )
        print("------Relaxation Ends------")
        print(
            f"------{onlinecalc.parent_calls} VASP Single Point Calculations Required------"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VASP Input Wrapper for Finetuna")
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint",
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default=os.path.abspath(""),
        help="Path to the VASP input directory",
    )
    parser.add_argument(
        "-con",
        "--config",
        type=str,
        default=files("finetuna.vasp_wrapper").joinpath("sample_config.yml"),
        help="Path to the config",
    )
    args = parser.parse_args()
    main(args)
