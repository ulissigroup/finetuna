from al_mlp.atomistic_methods import Relaxation
from al_mlp.calcs import CounterCalc
from ase.calculators.emt import EMT
from al_mlp.run_al import active_learning
import numpy as np
from ase.cluster.icosahedron import Icosahedron
from ase.optimize import BFGS
from ase.io import Trajectory

FORCE_THRESHOLD = 0.05
ENERGY_THRESHOLD = 0.01


class BaseOnlineCuNP:
    @classmethod
    def setUpClass(cls) -> None:
        # Set up parent calculator and image environment
        initial_structure = Icosahedron("Cu", 2)
        initial_structure.rattle(0.1)
        initial_structure.set_pbc(True)
        initial_structure.set_cell([20, 20, 20])

        # Run relaxation with the parent calc
        EMT_initial_structure = initial_structure.copy()
        cls.emt_counter = CounterCalc(EMT())
        EMT_initial_structure.set_calculator(cls.emt_counter)
        cls.EMT_structure_optim = Relaxation(
            EMT_initial_structure, BFGS, fmax=FORCE_THRESHOLD, steps=30
        )
        cls.EMT_structure_optim.run(cls.emt_counter, "CuNP_emt")

        # Run relaxation with active learning
        chemical_formula = initial_structure.get_chemical_formula()
        al_config = cls.get_al_config()
        al_config["links"]["traj"] = "CuNP_emt.traj"
        cls.oal_results_dict = active_learning(al_config)
        dbname = (
            str(al_config["links"]["ml_potential"])
            + "_"
            + str(chemical_formula)
            + "_oal"
        )
        cls.OAL_image = Trajectory(dbname + ".traj")[-1]
        cls.OAL_image.set_calculator(EMT())

        # Retain images of the final structure from both relaxations
        cls.EMT_image = cls.EMT_structure_optim.get_trajectory("CuNP_emt")[-1]
        cls.EMT_image.set_calculator(EMT())
        cls.description = "CuNP"
        return super().setUpClass()

    @classmethod
    def get_al_config(cls) -> dict:
        al_config = {
            "links": {
                "learner_class": "online",
                "parent_calc": "emt",
            },
            "learner": {
                "stat_uncertain_tol": 1,  # online
                "dyn_uncertain_tol": 1000000000,  # online
                "fmax_verify_threshold": 0.03,  # online
                "tolerance_selection": "min",
                "no_position_change_steps": 10,
                "min_position_change": 0.0005,
                "num_initial_points": 3,
                "initial_points_to_keep": [0, 2],
                "wandb_init": {
                    "wandb_log": False,
                },
                "logger": {
                    "uncertainty_quantify": False,
                    "pca_quantify": False,
                },
            },
            "relaxation": {
                "fmax": 0.03,
                "steps": 200,
                "maxstep": 0.04,
                "max_parent_calls": None,
                "replay_method": "parent_only",
                "check_final": True,
            },
            # "dataset": {
            #     "normalize_labels": True,
            #     "target_mean": -0.7554450631141663,
            #     "target_std": 2.887317180633545,
            #     "grad_target_mean": 0.0,
            #     "grad_target_std": 2.887317180633545,
            # },
        }
        return al_config

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
            + "\nafter "
            + str(self.oal_results_dict["current_step"])
            + " steps"
        )

    def test_oal_CuNP_forces(self):
        forces = self.OAL_image.get_forces()
        fmax = np.sqrt((forces ** 2).sum(axis=1).max())

        assert fmax <= FORCE_THRESHOLD, str(
            "Learner forces inconsistent:\n"
            + str(fmax)
            + "\nwith Force Threshold: "
            + str(FORCE_THRESHOLD)
            + "\nafter "
            + str(self.oal_results_dict["current_step"])
            + " steps"
        )

    def test_oal_CuNP_calls(self):
        print("OAL calls: %d" % self.oal_results_dict["parent_calls"])
        print("EMT calls: %d" % self.emt_counter.force_calls)

        fraction_of_parent_calls = 1
        assert (
            self.oal_results_dict["parent_calls"]
            <= fraction_of_parent_calls * self.emt_counter.force_calls
        ), str(
            "total calls: "
            + str(self.oal_results_dict["parent_calls"])
            + " not less than: "
            + str(fraction_of_parent_calls * self.emt_counter.force_calls)
            + "\nafter "
            + str(self.oal_results_dict["current_step"])
            + " steps"
        )
