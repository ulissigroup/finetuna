# adapted from
# https://github.com/ulissigroup/ulissigroup_docker_images/tree/master/kubernetes_examples/dask-vasp
import dask.bag as db
import tempfile
from ase.calculators.singlepoint import SinglePointCalculator as sp
import copy


def copy_images(images):
    """
    Copies images and returns the new instances.
    The new images also have copied calculators.

    Parameters
    ----------

    images: list
        List of ase atoms images to be copied.
    """
    new_images = []
    for image in images:
        calc = image.get_calculator()
        new_image = image.copy()
        new_image.set_calculator(copy.deepcopy(calc))
        new_images.append(new_image)
    return new_images


def calculate(atoms):
    with tempfile.TemporaryDirectory() as tmp_dir:
        atoms.get_calculator().set(directory=tmp_dir)
        sample_energy = atoms.get_potential_energy(apply_constraint=False)
        sample_forces = atoms.get_forces(apply_constraint=False)
        atoms.set_calculator(
            sp(atoms=atoms, energy=sample_energy, forces=sample_forces)
        )
    return atoms


def convert_to_singlepoint(images):
    """
    Replaces the attached calculators with singlepoint calculators

    Parameters
    ----------

    images: list
        List of ase atoms images with attached calculators for forces and energies.
    """

    images = copy_images(images)
    singlepoint_images = []
    to_be_calculated = []
    for image in images:
        if isinstance(image.get_calculator(), sp):
            singlepoint_images.append(image)
        else:
            to_be_calculated.append(image)
    calculated_images = compute_with_calc(to_be_calculated)
    singlepoint_images += calculated_images
    return singlepoint_images


def fix_dask_energy_datatype(image):
    """
    When dask returns atoms objects, the datatype of the energy
    value gets altered. This function makes sure that the correct
    type is returned.

    Parameters
    ----------

    images: ase Atoms object
        Image with attached singlepoint calculators.
    """
    image.calc.results["energy"] = float(image.calc.results["energy"])


def compute_with_calc(images, calculator=None):
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
        If None, uses the calculator attached to the images.
    """
    images = copy_images(images)
    if calculator is not None:
        for image in images:
            image.set_calculator(copy.deepcopy(calculator))
    images_bag = db.from_sequence(images)
    images_bag_computed = images_bag.map(calculate)
    singlepoint_images = images_bag_computed.compute()
    for image in singlepoint_images:
        fix_dask_energy_datatype(image)
    return singlepoint_images
