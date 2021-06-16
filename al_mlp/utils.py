import os
import copy
from ase.calculators.singlepoint import SinglePointCalculator as sp
from ase.db.core import check
from al_mlp.calcs import DeltaCalc
import numpy as np


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
    # cwd = os.getcwd()
    for image in images:
        if isinstance(image.get_calculator(), sp):
            singlepoint_images.append(image)
            continue
        # os.makedirs("./vasp_temp", exist_ok=True)
        # os.chdir("./vasp_temp")
        sample_energy = image.get_potential_energy(apply_constraint=False)
        sample_forces = image.get_forces(apply_constraint=False)
        if isinstance(image.get_calculator(), DeltaCalc):
            image.info["parent energy"] = image.get_calculator().parent_results[
                "energy"
            ]
            image.info["base energy"] = image.get_calculator().base_results["energy"]
            image.info["parent fmax"] = np.max(
                np.abs(image.get_calculator().parent_results["forces"])
            )

        sp_calc = sp(atoms=image, energy=float(sample_energy), forces=sample_forces)
        sp_calc.implemented_properties = ["energy", "forces"]
        image.set_calculator(sp_calc)
        # image.get_potential_energy()
        # image.get_forces()

        # image.calc.results["energy"] = float(image.calc.results["energy"])

        # sp_calc = sp(atoms=image, **image.calc.results)
        # sp_calc.implemented_properties = list(image.calc.results.keys())

        # image.set_calculator(sp_calc)
        singlepoint_images.append(image)
        # os.chdir(cwd)
        # os.system("rm -rf ./vasp_temp")

    return singlepoint_images


def compute_with_calc(images, calculator):
    """
    Calculates forces and energies of images with calculator.
    Returned images have singlepoint calculators.

    Parameters
    ----------

    images: list
        List of ase atoms images to be calculated.
    calculator: ase Calculator object
        Calculator used to get forces and energies.
    """

    images = copy_images(images)
    for image in images:
        image.set_calculator(calculator)
    return convert_to_singlepoint(images)


def subtract_deltas(images, base_calc, refs):
    """
    Produces the delta values of the image with precalculated values.
    This function is intended to be used by images that have
    precalculated forces and energies using the parent calc,
    that are attached to the image via a singlepoint calculator.
    This avoids having to recalculate results by a costly
    parent calc.

    Parameters
    ----------

    images: list
        List of ase atoms images to be calculated.
        Images should have singlepoint calculators with results.
    base_calc: ase Calculator object
        Calculator used as the baseline for taking delta subtraction.
    refs: list
        List of two images, they have results from parent and base calc
        respectively
    """

    images = copy_images(images)
    for image in images:
        parent_calc_sp = image.calc
        delta_sub_calc = DeltaCalc([parent_calc_sp, base_calc], "sub", refs)
        image.set_calculator(delta_sub_calc)
    return convert_to_singlepoint(images)


def copy_images(images):
    """
    Copies images and returns the new instances.
    The new images DO NOT have copied calculators.

    Parameters
    ----------

    images: list
        List of ase atoms images to be copied.
    """
    new_images = []
    for image in images:
        calc = image.get_calculator()
        new_image = image.copy()
        new_image.set_calculator(calc)
        new_images.append(new_image)
    return new_images


def write_to_db(database, queried_images, datatype="-", parentE="-", baseE="-"):
    for image in queried_images:
        database.write(
            image,
            key_value_pairs={"type": datatype, "parentE": parentE, "baseE": baseE},
        )


def write_to_db_online(
    database,
    queried_images,
    info,
):
    for image in queried_images:
        database.write(
            image,
            key_value_pairs={
                "check": info.get("check"),
                "uncertainty": info.get("uncertainty", "-"),
                "tolerance": info.get("tolerance", "-"),
                "parentE": info.get("parentE", "-"),
                "parentFmax": info.get("parentFmax", "-"),
                "parentF": info.get("parentF", "-"),
                "oalF": info.get("oalF", "-"),
            },
        )
