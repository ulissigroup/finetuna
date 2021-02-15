from .test_setup.offline_relaxation_test import run_offline_al
from al_mlp.atomistic_methods import Relaxation
from al_mlp.calcs import CounterCalc
from ase.calculators.emt import EMT
import numpy as np
from ase.cluster.icosahedron import Icosahedron
from ase.optimize import BFGS
import unittest

# Set up parent calculator and image environment


class offline_CuNP(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        initial_structure = Icosahedron("Cu", 2)
        initial_structure.rattle(0.1)
        initial_structure.set_pbc(True)
        initial_structure.set_cell([20, 20, 20])

        EMT_initial_structure = initial_structure.copy()
        cls.emt_counter = CounterCalc(EMT())
        EMT_initial_structure.set_calculator(cls.emt_counter)
        cls.EMT_structure_optim = Relaxation(
            EMT_initial_structure, BFGS, fmax=0.05, steps=30
        )
        cls.EMT_structure_optim.run(cls.emt_counter, "CuNP_emt")

        offline_initial_structure = initial_structure.copy()
        offline_initial_structure.set_calculator(EMT())
        Offline_relaxation = Relaxation(
            offline_initial_structure, BFGS, fmax=0.05, steps=30, maxstep=0.05
        )
        cls.offline_learner, cls.trained_calc, cls.Offline_traj = run_offline_al(
            Offline_relaxation, [offline_initial_structure], "CuNP_offline_al", EMT()
        )
        cls.EMT_image = cls.EMT_structure_optim.get_trajectory("CuNP_emt")[-1]
        cls.EMT_image.set_calculator(EMT())
        cls.offline_final_structure = cls.Offline_traj[-1]
        cls.offline_final_structure.set_calculator(cls.trained_calc)
        cls.description = "CuNP"
        return super().setUpClass()

    def test_offline_CuNP_energy(self):
        assert np.allclose(
            self.EMT_image.get_potential_energy(),
            self.offline_final_structure.get_potential_energy(),
            atol=0.05,
        )

    def test_offline_CuNP_forces(self):
        print(f"EMT force: {self.EMT_image.get_forces()}")
        print(f"AL force: {self.offline_final_structure.get_forces()}")

        assert np.allclose(
            np.abs(self.EMT_image.get_forces()),
            np.abs(self.offline_final_structure.get_forces()),
            atol=0.05,
        )

    def test_offline_CuNP_calls(self):
        assert self.offline_learner.iterations <= 0.7 * self.emt_counter.force_calls
