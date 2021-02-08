from ase.calculators.calculator import Calculator, all_changes
import numpy as np


class Dummy(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, images, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.images = images

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        image = atoms
        natoms = len(image)
        energy = 0.0
        forces = np.zeros((natoms, 3))
        self.results["energy"] = energy
        self.results["forces"] = forces
