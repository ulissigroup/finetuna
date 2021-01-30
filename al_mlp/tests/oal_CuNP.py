import ase.io
from .online_relaxation_test import run_oal
from al_mlp.atomistic_methods import Relaxation
from ase.calculators.emt import EMT
import numpy as np
from ase.cluster.icosahedron import Icosahedron
from ase.optimize import BFGS

#Set up parent calculator and image environment
initial_structure = Icosahedron('Cu',2)
initial_structure.rattle(0.1)
initial_structure.set_calculator(EMT())
initial_structure.set_pbc(True)
initial_structure.set_cell([20,20,20])

EMT_structure_optim = Relaxation(initial_structure,BFGS,fmax=0.05,steps = 100)
EMT_structure_optim.run(EMT(), 'CuNP_emt')
OAL_structure_optim = run_oal(initial_structure,'CuNP_oal')

def oal_CuNP_energy():
    EMT_traj = EMT_structure_optim.get_trajectory()
    OAL_traj = OAL_structure_optim.get_trajectory()

    assert(allclose(EMT_traj[-1].get_potential_energy(),
                    OAL_traj[-1].get_potential_energy(), atol=0.1))

def oal_CuNP_forces():
    EMT_traj = EMT_structure_optim.get_trajectory()
    OAL_traj = OAL_structure_optim.get_trajectory()

    assert(allclose(EMT_traj[-1].get_forces(),
                    OAL_traj[-1].get_forces(),atol=0.05))
