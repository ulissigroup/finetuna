import os
import sys
import copy
import numpy as np
import pandas as pd
from ase.db import connect
from ase.calculators.singlepoint import SinglePointCalculator as sp
from ase.calculators.calculator import Calculator
from al_mlp.utils import convert_to_singlepoint, compute_with_calc
from al_mlp.bootstrap import bootstrap_ensemble
from al_mlp.bootstrap import non_bootstrap_ensemble
from al_mlp.ensemble_calc import EnsembleCalc, make_ensemble
from al_mlp.calcs import DeltaCalc
from al_mlp.utils import copy_images

__author__ = "Muhammed Shuaibi"
__email__ = "mshuaibi@andrew.cmu.edu"


class OnlineActiveLearner(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        learner_params,
        trainer,
        parent_dataset,
        parent_calc,
        n_ensembles=10,
        n_cores="max",
    ):
        Calculator.__init__(self)

        self.n_ensembles = n_ensembles
        self.parent_calc = parent_calc
        self.trainer = trainer
        self.learner_params = learner_params
        self.n_cores = n_cores
        self.ensemble_sets, self.parent_dataset = non_bootstrap_ensemble(
            parent_dataset, n_ensembles=n_ensembles
        )
        trained_calcs = make_ensemble(self.ensemble_sets, self.trainer)
        self.ensemble_calc = EnsembleCalc(trained_calcs)
        self.uncertain_tol = learner_params["uncertain_tol"]
        self.parent_calls = 0

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        ensemble_calc_copy = copy.deepcopy(self.ensemble_calc)
        energy_pred = ensemble_calc_copy.get_potential_energy(atoms)
        force_pred = ensemble_calc_copy.get_forces(atoms)
        uncertainty = atoms.info["uncertainty"][0]
        uncertainty_tol = self.uncertain_tol
        print(uncertainty)

        # if uncertainty >= uncertainty_tol:
        if self.parent_calls > 2:
            print("DFT required")
            new_data = atoms.copy()
            new_data.set_calculator(copy.copy(self.parent_calc))

            energy_pred = new_data.get_potential_energy(apply_constraint=False)
            force_pred = new_data.get_forces(apply_constraint=False)
            new_data.set_calculator(
                sp(atoms=new_data, energy=energy_pred, forces=force_pred)
            )

            self.ensemble_sets, self.parent_dataset = non_bootstrap_ensemble(
                self.parent_dataset,
                compute_with_calc([new_data], self.parent_calc),
                n_ensembles=self.n_ensembles,
            )
            # self.parent_calls += 1

            trained_calcs = make_ensemble(self.ensemble_sets, self.trainer)
            self.ensemble_calc = EnsembleCalc(trained_calcs)
        self.parent_calls += 1

        self.results["energy"] = energy_pred
        self.results["forces"] = force_pred
