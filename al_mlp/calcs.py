from ase.calculators.calculator import all_changes
from ase.calculators.mixing import LinearCombinationCalculator


class DeltaCalc(LinearCombinationCalculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, calcs, mode, refs, atoms=None):
        """Implementation of sum of calculators.

        calcs: list
            List of two :mod:`ase.calculators` objects. First calculator must be the parent Calculator in "sub" mode,
            or the delta trained Calculator in "add" mode. Second calculator must be the base Calculator.
        mode: string
            "sub" or "add" is used to specify if the difference or the sum of the results are calculated.
        refs: list of atoms
            Same atoms with respective calculators attached. The first atoms must have the parent Calculator.
            The second atoms must have the base Calculator.
        atoms: Atoms object
            Optional :class:`~ase.Atoms` object to which the calculator will be attached.
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
        self.mode = mode
        self.parent_results = calcs[0].results
        self.base_results = calcs[1].results

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        """Calculates the desired property.
        Supports single point calculators with precalculated properties.
        "sub" mode: calculates the delta between the two given calculators.
        "add" mode: calculates the predicted value given the predicted delta calculator and the base calc.
        """
        self.calcs[0].results = self.parent_results
        self.calcs[1].results = self.base_results
        super().calculate(atoms, properties, system_changes)

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

        if "forces" in self.results:
            if self.mode == "sub":
                self.results["forces"] -= self.refs[0].get_forces()
                self.results["forces"] += self.refs[1].get_forces()
            else:
                self.results["forces"] -= self.refs[1].get_forces()
                self.results["forces"] += self.refs[0].get_forces()
