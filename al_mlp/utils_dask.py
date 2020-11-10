from al_mlp.utils import convert_to_singlepoint, copy_images
import dask.bag as db
import tempfile
from ase.calculators.singlepoint import SinglePointCalculator as sp
import copy


def calculate(atoms):
    with tempfile.TemporaryDirectory() as tmp_dir:
        atoms.get_calculator().set(directory=tmp_dir)
        sample_energy = atoms.get_potential_energy(apply_constraint=False)
        sample_forces = atoms.get_forces(apply_constraint=False)
        atoms.set_calculator(
            sp(atoms=atoms, energy=sample_energy, forces=sample_forces)
        )
    return atoms


def compute_with_calc(images, calculator, use_dask=False):
    """
    Calculates forces and energies of images with calculator.
    Returned images have singlepoint calculators.
    Uses dask for efficient parallelization.

    Parameters
    ----------

    images: list
        List of ase atoms images to be calculated.
    calc: ase Calculator object
        Calculator used to get forces and energies.
    """
    images = copy_images(images)
    for image in images:
        image.set_calculator(copy.deepcopy(calculator))
    images_bag = db.from_sequence(images)
    return images_bag.map(calculate).compute()
