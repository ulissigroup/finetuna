from ase.calculators.calculator import all_changes
from ase.calculators.mixing import LinearCombinationCalculator
from ase.calculators.calculator import Calculator
from ase.calculators.calculator import PropertyNotImplementedError
import copy
import numpy as np
from finetuna.ml_potentials.finetuner_calc import FinetunerCalc


class DeltaCalc(LinearCombinationCalculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, calcs, mode, refs, atoms=None):
        """Implementation of sum of calculators.

        calcs: list
            List of two :mod:`ase.calculators` objects.
            First calculator must be the parent Calculator in "sub" mode,
            or the delta trained Calculator in "add" mode.
            Second calculator must be the base Calculator.
        mode: string
            "sub" or "add" is used to specify if the difference
            or the sum of the results are calculated.
        refs: list of atoms
            Same atoms with respective calculators attached.
            The first atoms must have the parent Calculator.
            The second atoms must have the base Calculator.
        atoms: Atoms object
            Optional :class:`~ase.Atoms` object to which the calculator will
            be attached.
        """
        if mode == "sub":
            weights = [1, -1]
        elif mode == "add":
            weights = [1, 1]
        else:
            raise ValueError('mode must be "add" or "sub"!')
        if calcs[0] == calcs[1]:
            raise ValueError("Calculators cannot be the same!")

        super().__init__(calcs, weights, atoms)
        self.refs = refs
        self.diff_ref = False
        if (
            not self.refs[0].get_chemical_symbols()
            == self.refs[1].get_chemical_symbols()
        ):
            self.ref1_idx = np.array(
                [
                    atom.symbol in self.refs[1].get_chemical_symbols()
                    for atom in self.refs[0]
                ]
            )
            self.diff_ref = True
        self.mode = mode
        self.parent_results = calcs[0].results
        self.base_results = calcs[1].results
        self.force_calls = 0

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        """Calculates the desired property.
        Supports single point calculators with precalculated properties.
        "sub" mode: calculates the delta between the two given calculators.
        "add" mode: calculates the predicted value given the predicted delta
        calculator and the base calc.
        """
        if self.calcs[0] is self.refs[0].calc:
            raise ValueError("calc[0] and refs[0] calc are the same")
        if self.calcs[1] is self.refs[1].calc:
            raise ValueError("calc[1] and refs[1] calc are the same")
        if self.diff_ref:
            if atoms is not None:
                self.atoms = atoms.copy()
            atoms_list = [atoms, atoms[self.ref1_idx]]
            for atoms, calc in zip(atoms_list, self.calcs):
                if calc.calculation_required(atoms, ["energy", "forces"]):
                    calc.calculate(atoms, ["energy", "forces"], system_changes)
            self.results["energy"] = (
                self.calcs[0].results["energy"] - self.calcs[1].results["energy"]
            )

        else:
            super().calculate(atoms, properties, system_changes)
        self.parent_results = self.calcs[0].results
        self.base_results = self.calcs[1].results
        if self.diff_ref:
            zeros = np.zeros(self.parent_results["forces"].shape)
            zeros[self.ref1_idx] = self.base_results["forces"]
            self.base_results["forces"] = zeros
            properties = ["energy"]
        shared_properties = list(
            set(self.parent_results).intersection(self.base_results)
        )
        for w, calc in zip(self.weights, self.calcs):
            for k in shared_properties:
                if k not in properties:
                    if k not in self.results:
                        self.results[k] = w * calc.results[k]
                    else:
                        self.results[k] += w * calc.results[k]

        if "energy" in self.results:
            if self.mode == "sub":
                self.results["energy"] -= self.refs[0].get_potential_energy(
                    apply_constraint=False
                )
                self.results["energy"] += self.refs[1].get_potential_energy(
                    apply_constraint=False
                )
            else:
                self.results["energy"] -= self.refs[1].get_potential_energy(
                    apply_constraint=False
                )
                self.results["energy"] += self.refs[0].get_potential_energy(
                    apply_constraint=False
                )
        self.force_calls += 1

    def get_property(self, name, atoms=None, allow_calculation=True):
        if name not in self.implemented_properties:
            raise PropertyNotImplementedError(
                "{} property not implemented".format(name)
            )

        if atoms is None:
            atoms = self.atoms
            system_changes = []
        else:
            self.calcs[0].system_changes = self.calcs[0].check_state(atoms)
            if self.calcs[0].system_changes:
                self.calcs[0].reset()
            self.calcs[1].system_changes = self.calcs[1].check_state(atoms)
            if self.calcs[1].system_changes:
                self.calcs[1].reset()
        result = super().get_property(
            name, atoms=atoms, allow_calculation=allow_calculation
        )
        for calc in self.calcs:
            if hasattr(calc, "system_changes"):
                calc.system_changes = None
        return result

    def reset(self):
        """
        Unlike linearcombinationcalc, this does not clear all previous results recursively from all of the calculators.
        Instead it only clears delta calc and leaves the subcalcs untouched,
        then get property manually calls reset on them if they have system changes
        """
        super(LinearCombinationCalculator, self).reset()


class CounterCalc(Calculator):
    implemented_properties = ["energy", "forces", "uncertainty"]
    """
    Parameters
    --------------
        calc: object. Parent calculator to track force calls"""

    def __init__(self, calc, **kwargs):
        super().__init__()
        self.calc = copy.deepcopy(calc)
        self.force_calls = 0

    def calculate(self, atoms, properties, system_changes):
        super().calculate(atoms, properties, system_changes)
        calc = self.calc
        self.results["energy"] = calc.get_potential_energy(atoms)
        self.results["forces"] = calc.get_forces(atoms)
        self.force_calls += 1


class ClonedFinetunerCalc(FinetunerCalc):
    def __init__(self, finetuner_calc: FinetunerCalc):
        self.finetuner_calc = finetuner_calc
        FinetunerCalc.__init__(
            self, finetuner_calc.checkpoint_path, finetuner_calc.mlp_params
        )
        self.ml_model = True

    def load_trainer(self):
        self.trainer = self.finetuner_calc.trainer
