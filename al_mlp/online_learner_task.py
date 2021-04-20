from fireworks.core.firework import FWAction, FiretaskBase
from fireworks.utilities.fw_utilities import explicit_serialize
from al_mlp.online_learner import OnlineLearner
from al_mlp.atomistic_methods import Relaxation
from amptorch.trainer import AtomsTrainer
from ase.io import Trajectory
from atomate.vasp.database import VaspCalcDb
import jsonpickle


@explicit_serialize
class OnlineLearnerTask(FiretaskBase):
    def run_task(self, fw_spec):
        # learner params for the meat of the OnlineLearner
        learner_params_str = fw_spec.get("learner_params")
        trainer_config_str = fw_spec.get("trainer_config")
        images_str = fw_spec.get("parent_dataset")  # This has to be a full path
        filename = fw_spec.get("filename")
        init_struct_filename = fw_spec.get(
            "init_structure_path"
        )  # This has to be a full path
        db_path = fw_spec.get("db_path", None)  # This is the path to a db config

        # Decode the str back to objects to conduct the OAL
        learner_params = jsonpickle.decode(learner_params_str)
        trainer_config = jsonpickle.decode(trainer_config_str)
        # Load the parent dataset if any
        images = [atoms for atoms in Trajectory(images_str)]

        # Make sure each NN in the ensemble gets the same initial parent_dataset
        trainer_config["dataset"][
            "rawdata"
        ] = images  # for onlinelearner this will often be empty

        # Load the initial structure to be relaxed
        OAL_initial_structure = Trajectory(init_struct_filename)[
            0
        ]  # right now this needs to be a full path

        # Set up the trainer for the online learner
        trainer = AtomsTrainer(trainer_config)

        # Set up the online calc
        online_calc = OnlineLearner(
            learner_params, trainer, images, learner_params["parent_calc"]
        )

        # Set up the Relaxer
        OAL_Relaxer = Relaxation(
            OAL_initial_structure,
            learner_params["optim_relaxer"],
            fmax=learner_params["f_max"],
            steps=learner_params["steps"],
            maxstep=learner_params["maxstep"],
        )

        # Run the relaxation with online calc
        OAL_Relaxer.run(online_calc, filename=filename)

        OAL_image = OAL_Relaxer.get_trajectory(filename)[-1]
        OAL_image.set_calculator(learner_params["parent_calc"])

        # We can store arbitrary amounts of data into a clean (without fireworks metadata) collection called oal. Let's start with
        # just the final energy, the number of parent calls and the number of optimization steps
        # Make a connection to the database

        if db_path is not None:
            db = VaspCalcDb.from_db_file(db_path, admin=True)
            db.db["oal"].insert_one(
                {
                    "final_energy": OAL_image.get_potential_energy(),
                    "optimizer_steps": len(OAL_Relaxer.get_trajectory(filename)),
                    "parent_calls": online_calc.parent_calls,
                    "learner_params": learner_params_str,
                }
            )

        return FWAction(
            stored_data={
                "final_energy": OAL_image.get_potential_energy(),
                "parent_calls": online_calc.parent_calls,
                "optimizer_steps": len(OAL_Relaxer.get_trajectory(filename)),
                "learner_params": learner_params_str,
            },
        )
