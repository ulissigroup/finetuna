import unittest

from al_mlp.tests.setup.delta_relaxation_setup import run_delta_al
from al_mlp.atomistic_methods import Relaxation
from al_mlp.calcs import CounterCalc
from ase.calculators.emt import EMT
import numpy as np
from ase.cluster.icosahedron import Icosahedron
from ase.optimize import BFGS
from al_mlp.utils import convert_to_singlepoint

FORCE_THRESHOLD = 0.05
ENERGY_THRESHOLD = 0.01


class delta_CuNP(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Set up parent calculator and image environment
        cls.initial_structure = Icosahedron("Cu", 2)
        cls.initial_structure.rattle(0.1)
        cls.initial_structure.set_pbc(True)
        cls.initial_structure.set_cell([20, 20, 20])

        # Run relaxation with the parent calc
        EMT_initial_structure = cls.initial_structure.copy()
        cls.emt_counter = CounterCalc(EMT())
        EMT_initial_structure.set_calculator(cls.emt_counter)
        cls.EMT_structure_optim = Relaxation(
            EMT_initial_structure, BFGS, fmax=FORCE_THRESHOLD, steps=30
        )
        cls.EMT_structure_optim.run(cls.emt_counter, "CuNP_emt")

        # Run relaxation with active learning
        OAL_initial_structure = cls.initial_structure.copy()
        OAL_initial_structure.set_calculator(EMT())
        OAL_relaxation = Relaxation(
            OAL_initial_structure, BFGS, fmax=0.05, steps=60, maxstep=0.04
        )
        cls.OAL_learner, cls.OAL_structure_optim = run_delta_al(
            OAL_relaxation,
            [OAL_initial_structure],
            ["Cu"],
            "CuNP_oal",
            EMT(),
        )

        # Retain images of the final structure from both relaxations
        cls.EMT_image = cls.EMT_structure_optim.get_trajectory("CuNP_emt")[-1]
        cls.EMT_image.set_calculator(EMT())
        cls.OAL_image = cls.OAL_structure_optim.get_trajectory("CuNP_oal")[-1]
        cls.OAL_image.set_calculator(EMT())
        cls.description = "CuNP"
        return super().setUpClass()

    def test_oal_CuNP_energy(self):
        assert np.allclose(
            self.EMT_image.get_potential_energy(),
            self.OAL_image.get_potential_energy(),
            atol=ENERGY_THRESHOLD,
        ), str(
            "Learner energy inconsistent:\n"
            + str(self.EMT_image.get_potential_energy())
            + "or Parent energy inconsistent:\n"
            + str(self.OAL_image.get_potential_energy())
            + "\nwith Energy Threshold: "
            + str(ENERGY_THRESHOLD)
        )

    def test_oal_CuNP_forces(self):
        forces = self.OAL_image.get_forces()
        fmax = np.sqrt((forces**2).sum(axis=1).max())

        assert fmax <= FORCE_THRESHOLD, str(
            "Learner forces inconsistent:\n"
            + str(fmax)
            + "\nwith Force Threshold: "
            + str(FORCE_THRESHOLD)
        )

    def test_oal_CuNP_calls(self):

        print("OAL calls: %d" % self.OAL_learner.parent_calls)
        print("EMT calls: %d" % self.emt_counter.force_calls)

        assert self.OAL_learner.parent_calls <= 0.5 * self.emt_counter.force_calls, str(
            "total calls: "
            + str(self.OAL_learner.parent_calls)
            + " not less than: "
            + str(0.5 * self.emt_counter.force_calls)
        )

    def test_delta_get_ml_prediction(self):
        atoms_copy = self.OAL_learner.parent_dataset[-1].copy()

        atoms_ML = self.OAL_learner.get_ml_prediction(atoms_copy.copy())
        delta_sub_energy = atoms_ML.get_potential_energy()
        delta_sub_forces = atoms_ML.get_forces()

        atoms_copy = atoms_copy.copy()
        atoms_copy.set_calculator(self.OAL_learner.ml_potential)
        (atoms_ml_trained_on_diff_,) = convert_to_singlepoint([atoms_copy])
        ml_trained_on_diff_energy = atoms_ml_trained_on_diff_.get_potential_energy()
        ml_trained_on_diff_forces = atoms_ml_trained_on_diff_.get_forces()

        atoms_copy = atoms_copy.copy()
        atoms_copy.set_calculator(self.OAL_learner.base_calc)
        (atoms_base,) = convert_to_singlepoint([atoms_copy])
        base_energy = atoms_base.get_potential_energy()
        base_forces = atoms_base.get_forces()

        parent_ref_energy = self.OAL_learner.refs[0].get_potential_energy()
        base_ref_energy = self.OAL_learner.refs[1].get_potential_energy()

        # ml = (parent -parent ref) - (base - base ref)
        # parent = (base - base ref) + (ml + parent ref)
        delta_hand_energy = (ml_trained_on_diff_energy + parent_ref_energy) + (
            base_energy - base_ref_energy
        )
        delta_hand_forces = ml_trained_on_diff_forces + base_forces

        assert np.allclose(
            delta_sub_energy,
            delta_hand_energy,
            atol=ENERGY_THRESHOLD,
        ), str(
            "DeltaLearner get_ml_prediction() energy inconsistent:\n"
            + str(delta_sub_energy)
            + "\nwith hand calculated ML prediction delta:\n"
            + str(delta_hand_energy)
            + "\ncomposed of:\n  the ML trained on difference prediction:\n"
            + str(ml_trained_on_diff_energy)
            + "\n  the base calc prediction:\n"
            + str(base_energy)
            + "\n  the parent ref:\n"
            + str(parent_ref_energy)
            + "\n  the base ref:\n"
            + str(base_ref_energy)
            + "\nfor Energy Threshold: "
            + str(ENERGY_THRESHOLD)
        )

        assert np.allclose(
            delta_sub_forces,
            delta_hand_forces,
            atol=FORCE_THRESHOLD,
        ), str(
            "DeltaLearner get_ml_prediction() forces inconsistent:\n"
            + str(delta_sub_forces)
            + "\nwith hand calculated ML prediction delta:\n"
            + str(delta_hand_forces)
            + "\ncomposed of:\n  the ML trained on difference prediction:\n"
            + str(ml_trained_on_diff_forces)
            + "\n  the base calc prediction:\n"
            + str(base_forces)
            + "\nfor Force Threshold: "
            + str(FORCE_THRESHOLD)
        )

    def test_delta_add_to_dataset(self):
        atoms_copy = self.OAL_learner.parent_dataset[-1].copy()

        atoms_copy.set_calculator(EMT())
        [atoms_delta_sub] = self.OAL_learner.add_to_dataset(atoms_copy)
        delta_sub_energy = atoms_delta_sub.get_potential_energy()
        delta_sub_forces = atoms_delta_sub.get_forces()

        atoms_copy = atoms_copy.copy()
        atoms_copy.set_calculator(EMT())
        (atoms_parent,) = convert_to_singlepoint([atoms_copy])
        parent_energy = atoms_parent.get_potential_energy()
        parent_forces = atoms_parent.get_forces()

        atoms_copy = atoms_copy.copy()
        atoms_copy.set_calculator(self.OAL_learner.base_calc)
        (atoms_base,) = convert_to_singlepoint([atoms_copy])
        base_energy = atoms_base.get_potential_energy()
        base_forces = atoms_base.get_forces()

        parent_ref_energy = self.OAL_learner.refs[0].get_potential_energy()
        base_ref_energy = self.OAL_learner.refs[1].get_potential_energy()

        delta_hand_energy = (delta_sub_energy + parent_ref_energy) + (
            base_energy - base_ref_energy
        )
        delta_hand_forces = delta_sub_forces + base_forces

        assert self.OAL_learner.parent_dataset[-1] == atoms_parent

        assert np.allclose(
            delta_hand_energy,
            parent_energy,
            atol=ENERGY_THRESHOLD,
        ), str(
            "DeltaLearner add_to_dataset() + base calc energy inconsistent:\n"
            + str(delta_hand_energy)
            + "with calculated parent prediction:\n"
            + str(parent_energy)
            + "\ncomposed of:\n  the add_to_dataset() parent prediction:\n"
            + str(delta_sub_energy)
            + "\n  the base calc prediction:\n"
            + str(base_energy)
            + "\n  the parent ref:\n"
            + str(parent_ref_energy)
            + "\n  the base ref:\n"
            + str(base_ref_energy)
            + "\nfor Energy Threshold: "
            + str(ENERGY_THRESHOLD)
        )

        assert np.allclose(
            delta_hand_forces,
            parent_forces,
            atol=FORCE_THRESHOLD,
        ), str(
            "DeltaLearner add_to_dataset() + base calc forces inconsistent:\n"
            + str(delta_hand_forces)
            + "with calculated parent prediction:\n"
            + str(parent_forces)
            + "\ncomposed of:\n  the add_to_dataset() parent prediction:\n"
            + str(delta_sub_forces)
            + "\n  the base calc prediction:\n"
            + str(base_forces)
            + "\nfor Force Threshold: "
            + str(FORCE_THRESHOLD)
        )
