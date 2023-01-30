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
        "xc": args.xc,
    }
    for k in incar.keys():
        vasp_inputs[k.lower()] = incar[k]

    # Read the initial structure
    initial_structure = ase.io.read(os.path.join(args.path, "POSCAR"))
    if args.path == "":
        os.rename(
            os.path.join(args.path, "POSCAR"),
            os.path.join(args.path, "POSCAR_original"),
        )
        os.rename(
            os.path.join(args.path, "INCAR"), os.path.join(args.path, "INCAR_original")
        )
        os.rename(
            os.path.join(args.path, "KPOINTS"),
            os.path.join(args.path, "KPOINTS_original"),
        )

    # Set convergence criteria as EDIFFG in VASP flag, default to 0.03 eV/A
    if -vasp_interactive.exp_params.get("ediffg") == 0:
        relax_fmax = 0.03
    else:
        relax_fmax = -vasp_interactive.exp_params.get("ediffg")

    # Parse the config file
    yaml_file = open(args.config)
    # Set VASP command
    parsed_yaml_file = yaml.load(yaml_file, Loader=yaml.FullLoader)

    with open(os.path.abspath("") + "/run_config.yml", "w") as savefile:
        yaml.dump(
            {
                **parsed_yaml_file,
                "vasp_inputs": vasp_inputs,
                "checkpoint": args.checkpoint,
            },
            savefile,
        )

    if not args.vasponly:
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
        if args.vasponly:
            initial_structure.calc = parent_calc
            dyn = BFGS(initial_structure, trajectory="vasp_inter_bfgs_relax.traj")
            print(
                f"------Starting Relaxation. Terminate when Fmax <{relax_fmax} eV/A------"
            )
            dyn.run(
                fmax=relax_fmax,
            )
            print("------Relaxation Ends------")

        else:
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
                trajectory=parsed_yaml_file["relaxation"].get(
                    "trajname", "finetuna.traj"
                ),
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
        "-xc",
        "--xc",
        type=str,
        required=True,
        help="Exchange-correlation functional for VASP, e.g.: pbe, rpbe, beef-vdw. Check https://wiki.fysik.dtu.dk/ase/ase/calculators/vasp.html#exchange-correlation-functionals for the keys.",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default=None,
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
    parser.add_argument(
        "-vasponly",
        "--vasponly",
        action="store_true",
        help="Run a VASP Interactive relaxation with BFGS, no FineTuna!",
    )
    args = parser.parse_args()
    print(args.vasponly is False)
    if args.vasponly is False and args.checkpoint is None:
        parser.error("Please provide a path to the checkpoint for using FineTuna.")
    main(args)
