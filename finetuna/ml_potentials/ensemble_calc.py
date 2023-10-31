from ase.calculators.calculator import all_changes
from finetuna.ml_potentials.ml_potential_calc import MLPCalc
from finetuna.utils import compute_with_calc


class EnsembleCalc(MLPCalc):
    """
    Basic ensemble calculator to make use of already constructed calculators.
    No settings, simply feed in constructed calculators.
    Calculate will return mean values for forces and energies, and will store each individual calculators forces and energies

    Parameters
    ----------
    calcs: dict
        dictionary containing names of calculators as keys and calculators as values
    mlp_params: dict
        dictionary of parameters to be passed to the ml potential model in init_model()
    """

    implemented_properties = ["energy", "forces", "stds"]

    def __init__(
        self,
        calcs: dict,
        mlp_params: dict = {},
    ):
        self.calcs = calcs
        MLPCalc.__init__(self, mlp_params=mlp_params)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Calculate properties including: energy, forces, uncertainties.

        Args:
            atoms: ase Atoms object
        """
        super().calculate(
            atoms=atoms, properties=properties, system_changes=system_changes
        )

        self.results["members"] = {}
        mean_energy = None
        mean_forces = None
        for key, value in self.calcs.items():
            [atoms_copy] = compute_with_calc([atoms], value)
            energy = atoms_copy.get_potential_energy()
            forces = atoms_copy.get_forces()
            if mean_energy is None:
                mean_energy = energy
            else:
                mean_energy += energy
            if mean_forces is None:
                mean_forces = forces
            else:
                mean_forces += forces
            self.results["members"][key] = atoms_copy
        self.results["energy"] = mean_energy / len(self.calcs)
        self.results["forces"] = forces / len(self.calcs)
