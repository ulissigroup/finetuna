import glob
import random
import time
import copy
import logging
import yaml

import ase.io
from ase.calculators.calculator import Calculator

from ocpmodels.common.relaxation.ase_utils import OCPCalculator
from ocpmodels.common.utils import setup_imports
from al_mlp.online_learner.forces_trainer_uncertainty import ForcesTrainer
from ocpmodels.preprocessing import AtomsToGraphs


class OCPCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, config_yml, pbc_graph=False, checkpoint=None):
        """
        OCP-ASE Calculator

        Args:
            config: File Path
                Path specifying trainer for ML predictions. See config/ directory.
        """
        setup_imports()
        Calculator.__init__(self)

        if isinstance(config_yml, str):
            config = yaml.safe_load(open(config_yml, "r"))
            if "includes" in config:
                for include in config["includes"]:
                    include_config = yaml.safe_load(open(include, "r"))
                    config.update(include_config)
        else:
            config = config_yml

        # Save config so obj can be transported over network (pkl)
        self.config = copy.deepcopy(config)
        self.config["pbc_graph"] = pbc_graph
        self.config["checkpoint"] = checkpoint
        self.trainer = ForcesTrainer(
            task=config["task"],
            model=config["model"],
            dataset=config["dataset"],
            optimizer=config["optim"],
            normalizer=config["dataset"],
            identifier="",
            logger="wandb",
            is_debug=True,
        )

        if checkpoint is not None:
            self.load_checkpoint(checkpoint)

        self.pbc_graph = pbc_graph
        self.a2g = AtomsToGraphs(
            max_neigh=50,
            radius=6,
            r_energy=False,
            r_forces=False,
            r_distances=False,
        )

    def train(self, parent_dataset, new_dataset=None):
        self.trainer.train()

    def load_checkpoint(self, checkpoint_path):
        """
        Load existing trained model

        Args:
            checkpoint_path: string
                Path to trained model
        """
        try:
            self.trainer.load_checkpoint(checkpoint_path)
        except NotImplementedError:
            logging.warning("Unable to load checkpoint!")

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        data_object = self.a2g.convert(atoms)
        batch = data_list_collater([data_object])
        if self.pbc_graph:
            edge_index, cell_offsets, neighbors = radius_graph_pbc(
                batch, 6, 50, batch.pos.device
            )
            batch.edge_index = edge_index
            batch.cell_offsets = cell_offsets
            batch.neighbors = neighbors
        predictions = self.trainer.predict(
            batch, per_image=False, disable_tqdm=True
        )

        self.results["energy"] = predictions["energy"].item()
        self.results["forces"] = predictions["forces"].cpu().numpy()

