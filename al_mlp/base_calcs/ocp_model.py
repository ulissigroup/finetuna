from ase.atoms import Atoms
import yaml
import os
import lmdb
import pickle
import torch

from ocpmodels.preprocessing import AtomsToGraphs
from ocpmodels.trainers import ForcesTrainer
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

        model_dict = {}
        with open(model_path) as model_yaml:
            model_dict = yaml.safe_load(model_yaml)

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
                    "src": "/home/jovyan/working/ocp/data/s2ef/2M/train/",
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

        self.trainer = ForcesTrainer(
            task=task,
            model=model_dict["model"],
            dataset=dataset,
            optimizer=model_dict["optim"],
            identifier=identifier,
            cpu=True,
        )

        self.trainer.load_pretrained(checkpoint_path=checkpoint_path)

    def calculate(self, atoms=None, properties=["forces"], system_changes=all_changes):
        Calculator.calculate(
            self, atoms=atoms, properties=properties, system_changes=system_changes
        )
        atoms_list = []
        if isinstance(atoms, Atoms):
            atoms_list = [atoms]
        else:
            atoms_list = atoms
        data_objects = self.ase_to_data(atoms_list, self.a2g)
        batch = self.data_to_batch(data_objects)

        prediction = self.trainer._forward([batch])

        energy = prediction["energy"].data.numpy()[0]
        forces = prediction["forces"].data.numpy()

        self.results["energy"] = energy
        self.results["forces"] = forces

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k is not "trainer":
                setattr(result, k, copy.deepcopy(v, memo))
            else:
                setattr(result, k, v)
        return result

    @staticmethod
    def ase_to_data(ase_images_list, a2g, disable_tqdm=True):
        os.makedirs("s2ef", exist_ok=True)
        db = lmdb.open(
            "s2ef/sample.lmdb",
            map_size=1099511627776/2, # * 2,
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
