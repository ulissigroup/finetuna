from ase.calculators.calculator import Calculator, all_changes
import numpy as np
from numpy.core.numeric import zeros_like


class Dummy(Calculator):
    implemented_properties = ["energy", "forces", "stress", "stds"]

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        image = atoms
        natoms = len(image)
        energy = 0.0
        forces = np.zeros((natoms, 3))
        self.results["energy"] = energy
        self.results["forces"] = forces
        self.results["stress"] = -np.array([0, 0, 0, 0, 0, 0])
        self.results["force_stds"] = zeros_like(forces)
        atoms.info["max_force_stds"] = np.nanmax(self.results["force_stds"])
