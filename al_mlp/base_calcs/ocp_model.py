from ase.atoms import Atoms
import yaml
import os
import lmdb
import pickle
import torch

from ocpmodels.preprocessing import AtomsToGraphs
from ase.calculators.calculator import Calculator, all_changes
from torch_geometric.data import Batch
import copy

from ocpmodels.trainers.forces_trainer import ForcesTrainer


class OCPModel(Calculator):
    implemented_properties = ["energy", "forces"]
    nolabel = True

    def __init__(
        self,
        model_path,
        checkpoint_path,
        dataset=None,
        a2g=None,
        task=None,
        identifier="active_learner_base_calc",
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)

        self.model_path = model_path
        self.checkpoint_path = checkpoint_path
        self.dataset = dataset
        self.a2g = a2g
        self.task = task
        self.identifier = identifier
        self.kwargs = kwargs

        self.model_dict = {}
        with open(model_path) as model_yaml:
            self.model_dict = yaml.safe_load(model_yaml)
        self.model_dict["optim"]["num_workers"] = 4
        # model_dict["model"]["freeze"] = False

        if not task:
            task = {
                "dataset": "trajectory_lmdb",  # dataset used for the S2EF task
                "description": "S2EF for active learning base calc",
                "type": "regression",
                "metric": "mae",
                "labels": ["potential energy"],
                "grad_input": "atomic forces",
                "train_on_free_atoms": True,
                "eval_on_free_atoms": True,
            }

        if not dataset:
            dataset = [
                {
                    "src": "/home/jovyan/shared-datasets/OC20/s2ef/30k/train",
                    "normalize_labels": False,
                }
            ]

        if not a2g:
            a2g = AtomsToGraphs(
                max_neigh=50,
                radius=6,
                r_energy=True,
                r_forces=True,
                r_distances=False,
                r_edges=True,
                r_fixed=True,
            )
        self.a2g = a2g
        self.a2g_predict = copy.deepcopy(self.a2g)
        self.a2g_predict.r_forces = False
        self.a2g_predict.r_energy = False

        self.trainer = ForcesTrainer(
            task=task,
            model=self.model_dict["model"],
            dataset=dataset,
            optimizer=self.model_dict["optim"],
            identifier=identifier,
            is_debug=True,
            is_vis=False,
            cpu=True,
        )

        self.trainer.load_pretrained(checkpoint_path=checkpoint_path, ddp_to_dp=True)

    def calculate(self, atoms=None, properties=["forces"], system_changes=all_changes):
        Calculator.calculate(
            self, atoms=atoms, properties=properties, system_changes=system_changes
        )
        atoms_list = []
        if isinstance(atoms, Atoms):
            atoms_list = [atoms]
        else:
            atoms_list = atoms

        data_objects = self.ase_to_data(atoms_list, self.a2g_predict)
        batch = self.data_to_batch(data_objects)

        prediction = self.trainer.predict(batch)

        energy = prediction["energy"][0]
        forces = prediction["forces"][0]

        self.results["energy"] = energy
        self.results["forces"] = forces

    @staticmethod
    def ase_to_data(ase_images_list, a2g, disable_tqdm=True):
        os.makedirs("s2ef", exist_ok=True)
        db = lmdb.open(
            "s2ef/sample.lmdb",
            map_size=1099511627776 / 2,  # * 2,
            subdir=False,
            meminit=False,
            map_async=True,
        )
        data_objects = a2g.convert_all(ase_images_list, disable_tqdm=disable_tqdm)
        for fid, data in enumerate(data_objects):
            data.sid = torch.LongTensor([0])
            data.fid = torch.LongTensor([fid])
            txn = db.begin(write=True)
            txn.put(f"{fid}".encode("ascii"), pickle.dumps(data, protocol=-1))
            txn.commit()
        txn = db.begin(write=True)
        txn.put(
            f"length".encode("ascii"),
            pickle.dumps(len(data_objects), protocol=-1),
        )
        txn.commit()
        db.sync()
        db.close()
        return data_objects

    @staticmethod
    def data_to_batch(data_objects):
        batch = Batch.from_data_list(data_objects)
        try:
            n_neighbors = []
            for i, data in enumerate(data_objects):
                n_index = data.edge_index[1, :]
                n_neighbors.append(n_index.shape[0])
            batch.neighbors = torch.tensor(n_neighbors)
        except NotImplementedError:
            print("LMDB does not contain edge index information, set otf_graph=True")
        return batch

    def get_params(self):
        params = {"checkpoint": self.checkpoint_path}
        params.update(self.model_dict)
        return params