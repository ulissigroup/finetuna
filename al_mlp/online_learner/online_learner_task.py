from fireworks.core.firework import FWAction, FiretaskBase
from fireworks.utilities.fw_utilities import explicit_serialize
from al_mlp.online_learner.online_learner import OnlineLearner
from al_mlp.atomistic_methods import Relaxation
from amptorch.trainer import AtomsTrainer
from ase.io import Trajectory
from atomate.vasp.database import VaspCalcDb
from pymatgen.io.ase import AseAtomsAdaptor as AAA
from pymongo import MongoClient
import jsonpickle


@explicit_serialize
class OnlineLearnerTask(FiretaskBase):
    def run_task(self, fw_spec):

        # Tell the client how to connect to the Dask LocalCluster
        from al_mlp.ml_potentials.amptorch_ensemble_calc import AmptorchEnsembleCalc
        from dask.distributed import Client
        client = Client(scheduler_file=fw_spec.get("scheduler_file"))
        AmptorchEnsembleCalc.set_executor(client)

        # learner params for the meat of the OnlineLearner
        learner_params_str = fw_spec.get("learner_params")
        images_str = fw_spec.get("parent_dataset")  # This has to be a full path
        filename = fw_spec.get("filename")
        task_name = fw_spec.get("task_name")
        init_struct_filename = fw_spec.get(
            "init_structure_path"
        )  # This has to be a full path
        db_path = fw_spec.get("db_path", None)  # This is the path to a db config
        # Get the fw_id
        fw_id = self.fw_id
        # Get the latest launch_id for this FW
        launch_id = self.launchpad.launches.find({"fw_id": self.fw_id}).sort([('launch_id',-1)]).limit(1)[0]['launch_id']
        # Decode the str back to objects to conduct the OAL
        learner_params = jsonpickle.decode(learner_params_str)
        # Load the parent dataset if any
        images = [atoms for atoms in Trajectory(images_str)]

        # Load the initial structure to be relaxed
        OAL_initial_structure = Trajectory(init_struct_filename)[
            0
        ]  # right now this needs to be a full path

        # Set up the trainer for the online learner
        parent_calc = learner_params['parent_calc']
        # Instantiate the FlarePP constructor object
        if learner_params['ml_potential'].__name__ == 'FlarePPCalc':
            # Retrieve the flare params from spec
            flare_params = fw_spec.get("flare_params")
            ml_potential = learner_params['ml_potential'](flare_params,
                                                         [OAL_initial_structure])


        # Set up the online calc
        online_calc = OnlineLearner(
            learner_params, 
            images,
            ml_potential,
            parent_calc,
            task_name,
            launch_id,
            fw_id
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
        OAL_image.set_calculator(parent_calc)

        # Make a connection to the database

        client = MongoClient('mongodb://fw_oal_admin:gfde223223222rft3@mongodb07.nersc.gov:27017/fw_oal')
        db = client.get_database('fw_oal')
        calcs_reversed = {'task_label': task_name,
                          'calcs_reversed': [{'output':{'energy': OAL_image.get_potential_energy(),
                                                       'structure': AAA.get_structure(OAL_image).as_dict(), # To be consistent with how OptimizeFW
                                                       # stores the relaxed object
                                                       }}]} # keep the same data structure as Javi's workflow
        print(calcs_reversed)
        db['tasks'].insert_one(calcs_reversed) # Store the final relaxed structure and energy into the tasks collection

        return FWAction(
            stored_data={
                "final_energy": OAL_image.get_potential_energy(),
                "parent_calls": online_calc.parent_calls,
                "optimizer_steps": len(OAL_Relaxer.get_trajectory(filename)),
                "learner_params": learner_params_str,
            },
        )
