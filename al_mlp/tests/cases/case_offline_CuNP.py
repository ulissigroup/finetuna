from al_mlp.tests.setup.offline_relaxation_test import run_offline_al
from al_mlp.atomistic_methods import Relaxation
from al_mlp.calcs import CounterCalc
from ase.calculators.emt import EMT
import numpy as np
from ase.cluster.icosahedron import Icosahedron
from ase.optimize import BFGS
import unittest
from al_mlp.utils import compute_with_calc

FORCE_THRESHOLD = 0.05
ENERGY_THRESHOLD = 0.03
# Set up parent calculator and image environment


class offline_CuNP(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        initial_structure = Icosahedron("Cu", 2)
        initial_structure.rattle(0.1)
        initial_structure.set_pbc(True)
        initial_structure.set_cell([20, 20, 20])

        EMT_initial_structure = initial_structure.copy()
        parent_calc = EMT()
        cls.emt_counter = CounterCalc(parent_calc)
        EMT_initial_structure.set_calculator(cls.emt_counter)
        cls.EMT_structure_optim = Relaxation(
            EMT_initial_structure, BFGS, fmax=0.01, steps=30, maxstep=0.05
        )
        cls.EMT_structure_optim.run(cls.emt_counter, "CuNP_emt")

        offline_initial_structure = compute_with_calc(
            [initial_structure.copy()], parent_calc
        )[0]
        Offline_relaxation = Relaxation(
            offline_initial_structure, BFGS, fmax=0.01, steps=30, maxstep=0.05
        )
        cls.offline_learner, cls.trained_calc, cls.Offline_traj = run_offline_al(
            Offline_relaxation,
            [offline_initial_structure],
            "CuNP_offline_al",
            parent_calc,
        )
        cls.EMT_image = cls.EMT_structure_optim.get_trajectory("CuNP_emt")[-1]
        cls.EMT_image.set_calculator(parent_calc)
        cls.offline_final_structure_AL = cls.Offline_traj[-1]
        cls.offline_final_structure_AL.set_calculator(cls.trained_calc)
        cls.offline_final_structure_EMT = cls.Offline_traj[-1]
        cls.offline_final_structure_EMT.set_calculator(parent_calc)
        cls.description = "CuNP"
        return super().setUpClass()

    def test_energy_AL_EMT(self):
        assert np.allclose(
            self.EMT_image.get_potential_energy(),
            self.offline_final_structure_AL.get_potential_energy(),
            atol=ENERGY_THRESHOLD,
        ), str(
            "Learner energy inconsistent:\n"
            + str(self.EMT_image.get_potential_energy())
            + "or Parent energy inconsistent:\n"
            + str(self.offline_final_structure_AL.get_potential_energy())
            + "\nwith Energy Threshold: "
            + str(ENERGY_THRESHOLD)
        )

    def test_energy_EMT_EMT(self):
        assert np.allclose(
            self.EMT_image.get_potential_energy(),
            self.offline_final_structure_EMT.get_potential_energy(),
            atol=ENERGY_THRESHOLD,
        ), str(
            "Learner energy inconsistent:\n"
            + str(self.EMT_image.get_potential_energy())
            + "or Parent energy inconsistent:\n"
            + str(self.offline_final_structure_EMT.get_potential_energy())
            + "\nwith Energy Threshold: "
            + str(ENERGY_THRESHOLD)
        )

    def test_offline_CuNP_forces(self):
        forces = self.offline_final_structure_AL.get_forces()
        fmax = np.sqrt((forces ** 2).sum(axis=1).max())

        assert fmax <= FORCE_THRESHOLD, str(
            "Learner forces inconsistent:\n"
            + str(fmax)
            + "\nwith Force Threshold: "
            + str(FORCE_THRESHOLD)
        )

    def test_offline_CuNP_calls(self):
        assert (
            self.offline_learner.parent_calls <= 0.5 * self.emt_counter.force_calls
        ), str(
            "total calls:"
            + str(self.offline_learner.parent_calls)
            + " not less than: "
            + str(self.emt_counter.force_calls * 0.5)
        )
