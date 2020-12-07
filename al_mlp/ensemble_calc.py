import numpy as np
from ase.calculators.calculator import Calculator

__author__ = "Muhammed Shuaibi"
__email__ = "mshuaibi@andrew.cmu.edu"


class EnsembleCalc(Calculator):
    """Atomistics Machine-Learning Potential (AMP) ASE calculator
    Parameters
    ----------
     model : object
         Class representing the regression model. Input arguments include training
         images, descriptor type, and force_coefficient. Model structure and training schemes can be
         modified directly within the class.

     label : str
         Location to save the trained model.

    """

    implemented_properties = ["energy", "forces", "uncertainty"]

    def __init__(self, trained_calcs):
        Calculator.__init__(self)
        self.trained_calcs = trained_calcs

    def calculate_stats(self, energies, forces):
        median_idx = np.argsort(energies)[len(energies) // 2]
        energy_median = energies[median_idx]
        forces_median = forces[median_idx]
        max_forces_var = np.max(np.var(forces, axis=0))
        return energy_median, forces_median, max_forces_var

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        energies = []
        forces = []

        for calc in self.trained_calcs:
            energies.append(calc.get_potential_energy(atoms))
            forces.append(calc.get_forces(atoms))
        energies = np.array(energies)
        forces = np.array(forces)
        energy_pred, force_pred, uncertainty = self.calculate_stats(energies, forces)

        self.results["energy"] = energy_pred
        self.results["forces"] = force_pred
        atoms.info["uncertainty"] = np.array([uncertainty])
