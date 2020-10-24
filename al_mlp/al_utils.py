import os
import copy
import numpy as np

import ase.io
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator as sp

from matplotlib import pyplot as plt


class CounterCalc(Calculator):
    implemented_properties = ["energy", "forces", "uncertainty"]
    """Parameters
    --------------
        calc: object. Parent calculator to track force calls."""

    def __init__(self, calc, **kwargs):
        super().__init__()
        self.calc = calc
        self.force_calls = 0

    def calculate(self, atoms, properties, system_changes):
        super().calculate(atoms, properties, system_changes)
        calc = copy.deepcopy(self.calc)
        self.results["energy"] = calc.get_potential_energy(atoms)
        self.results["forces"] = calc.get_forces(atoms)
        self.force_calls += 1


def attach_sp_calc(images):
    "Converts images calculators to single point calculators to avoid double calculations'"
    if isinstance(images, list):
        for image in images:
            construct_sp(image)
    else:
        construct_sp(images)
    return images


def construct_sp(image):
    sample_energy = image.get_potential_energy(apply_constraint=False)
    sample_forces = image.get_forces(apply_constraint=False)
    image.set_calculator(sp(atoms=image, energy=sample_energy, forces=sample_forces))
    return image


def write_to_db(database, queried_images):
    for image in queried_images:
        database.write(image)


def compute_loss(a, b):
    return np.mean(np.sqrt(np.sum((a - b) ** 2, axis=1)))


def progressive_plot(
    filename, true_relaxed, samples_per_iter, num_iterations, save_to="./"
):
    os.makedirs(save_to, exist_ok=True)
    distance_rmse = []
    data_size = list(range(0, samples_per_iter * num_iterations + 1, samples_per_iter))

    for i in range(len(data_size)):
        ml_relaxed = ase.io.read("{}_iter_{}.traj".format(filename, i), "-1")
        loss = compute_loss(ml_relaxed.positions, true_relaxed.positions)
        distance_rmse.append(loss)
    plt.plot(np.array(data_size) + 1, distance_rmse)
    plt.xlabel("Training Images")
    plt.xticks(np.array(data_size) + 1)
    plt.ylabel("Distance RMSE")
    plt.title("AL Relaxation Learning Curve")
    plt.savefig(save_to + ".png", dpi=300)
