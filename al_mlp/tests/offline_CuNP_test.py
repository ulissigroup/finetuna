from offline_relaxation_test import run_offline_al
from al_mlp.atomistic_methods import Relaxation
from al_mlp.calcs import CounterCalc
from ase.calculators.emt import EMT
import numpy as np
from ase.cluster.icosahedron import Icosahedron
from ase.optimize import BFGS
import unittest

# Set up parent calculator and image environment
initial_structure = Icosahedron("Cu", 2)
initial_structure.rattle(0.1)
initial_structure.set_pbc(True)
initial_structure.set_cell([20, 20, 20])


EMT_initial_structure = initial_structure.copy()
emt_counter = CounterCalc(EMT())
EMT_initial_structure.set_calculator(emt_counter)
EMT_structure_optim = Relaxation(EMT_initial_structure, BFGS, fmax=0.05, steps=30)
EMT_structure_optim.run(emt_counter, "CuNP_emt")

Offline_AL_initial_structure = initial_structure.copy()
Offline_AL_initial_structure.set_calculator(EMT())
Offline_AL_relaxation = Relaxation(
    Offline_AL_initial_structure, BFGS, fmax=0.05, steps=30, maxstep=0.04
)
Offline_AL_learner, Offline_AL_traj = run_offline_al(
    Offline_AL_relaxation, [Offline_AL_initial_structure], "CuNP_offline_al", EMT()
)


def oal_CuNP_energy():
    EMT_image = EMT_structure_optim.get_trajectory("CuNP_emt")[-1]
    EMT_image.set_calculator(EMT())
    # Offline_AL_image = Offline_AL_structure_optim.get_trajectory("CuNP_offline_al")[-1]
    # Offline_AL_energies = [
    #    image.get_potential_energy() for image in Offline_AL_final_structure
    # ]
    Offline_AL_final_structure = Offline_AL_traj[-1]
    Offline_AL_final_structure.set_calculator(EMT())

    assert np.allclose(
        EMT_image.get_potential_energy(),
        Offline_AL_final_structure.get_potential_energy(),
        atol=0.1,
    )


def oal_CuNP_forces():
    EMT_image = EMT_structure_optim.get_trajectory("CuNP_emt")[-1]
    EMT_image.set_calculator(EMT())
    # Offline_AL_image = Offline_AL_structure_optim.get_trajectory("CuNP_offline_al")[-1]
    Offline_AL_final_structure = Offline_AL_traj[-1]
    Offline_AL_final_structure.set_calculator(EMT())
    # Offline_AL_forces = [image.get_forces() for image in Offline_AL_final_structure]

    assert np.allclose(
        EMT_image.get_forces(), Offline_AL_final_structure.get_forces(), atol=0.05
    )


def oal_CuNP_calls():
    print(Offline_AL_learner.parent_calls)
    print(emt_counter.force_calls)
    assert Offline_AL_learner.parent_calls < 0.5 * emt_counter.force_calls


class TestMethods(unittest.TestCase):
    # def test_delta_sub(self):
    #    test_delta_sub()

    # def test_delta_add(self):
    #    test_delta_add()

    def test_oal_CuNP_energy(self):
        oal_CuNP_energy()

    def test_oal_CuNP_forces(self):
        oal_CuNP_forces()

    def test_oal_CuNP_calls(self):
        oal_CuNP_calls()


if __name__ == "__main__":
    from dask.distributed import Client, LocalCluster
    from al_mlp.ensemble_calc import EnsembleCalc

    cluster = LocalCluster(n_workers=4, processes=True, threads_per_worker=1)
    client = Client(cluster)
    EnsembleCalc.set_executor(client)
    unittest.main(warnings="ignore")
