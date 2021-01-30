import ase.io
from .online_relaxation_test import run_oal
from al_mlp.atomistic_methods import Relaxation
from ase.calculators.emt import EMT
import numpy as np
from ase.cluster.icosahedron import Icosahedron
from ase.optimize import BFGS

# Set up parent calculator and image environment
initial_structure = Icosahedron("Cu", 2)
initial_structure.rattle(0.1)
initial_structure.set_pbc(True)
initial_structure.set_cell([20, 20, 20])

EMT_initial_structure = initial_structure.copy()
EMT_initial_structure.set_calculator(EMT())
EMT_structure_optim = Relaxation(EMT_initial_structure, BFGS, fmax=0.05, steps=100)
EMT_structure_optim.run(EMT(), "CuNP_emt")

OAL_initial_structure = initial_structure.copy()
OAL_initial_structure.set_calculator(EMT())
OAL_structure_optim = run_oal(OAL_initial_structure, "CuNP_oal")


def oal_CuNP_energy():
    EMT_image = EMT_structure_optim.get_trajectory()[-1]
    EMT_image.set_calculator(EMT())
    OAL_image = OAL_structure_optim.get_trajectory()[-1]
    OAL_image.set_calculator(EMT())

    assert allclose(
        EMT_traj[-1].get_potential_energy(),
        OAL_traj[-1].get_potential_energy(),
        atol=0.1,
    )


def oal_CuNP_forces():
    EMT_image = EMT_structure_optim.get_trajectory()[-1]
    EMT_image.set_calculator(EMT())
    OAL_image = OAL_structure_optim.get_trajectory()[-1]
    OAL_image.set_calculator(EMT())

    assert allclose(EMT_traj[-1].get_forces(), OAL_traj[-1].get_forces(), atol=0.05)

def oal_CuNP_calls():

    # What I want here is the number of EMT calls; I don't think that this is
    # what get_trajectory actually does
    OAL_images = OAL_structure_optim.get_trajectory()
    EMT_images = EMT_structure_optim.get_trajectory()

    assert len(OAL_images)<0.5*len(EMT_IMAGES)
