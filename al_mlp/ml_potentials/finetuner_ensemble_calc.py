from al_mlp.ml_potentials.finetuner_calc import FinetunerCalc
from al_mlp.ml_potentials.ml_potential_calc import MLPCalc
import numpy as np
import copy
import time


class FinetunerEnsembleCalc(FinetunerCalc):
    """
    FinetunerEnsembleCalc.
    ML potential calculator class that implements an ensemble of partially frozen ocp models.

    Parameters
    ----------
    model_classes: list[str]
        list of paths to classnames, corresponding to checkpoints and configs, e.g.
        [
            "Gemnet",
            "Spinconv",
            "Dimenetpp",
        ]

    model_paths: list[str]
        list of paths to model configs, corresponding to classes list, e.g.
        [
            '/home/jovyan/working/ocp/configs/s2ef/all/gemnet/gemnet-dT.yml',
            '/home/jovyan/working/ocp/configs/s2ef/all/spinconv/spinconv_force.yml',
            '/home/jovyan/working/ocp/configs/s2ef/all/dimenet_plus_plus/dpp_forceonly.yml',
        ]

    checkpoint_paths: list[str]
        list of paths checkpoints, corresponding to classes list, e.g.
        [
            '/home/jovyan/shared-datasets/OC20/checkpoints/s2ef/gemnet_t_direct_h512_all.pt'
            '/home/jovyan/shared-datasets/OC20/checkpoints/s2ef/spinconv_force_centric_all.pt',
            '/home/jovyan/shared-datasets/OC20/checkpoints/s2ef/dimenetpp_all_forceonly.pt',
        ]

    mlp_params: dict
        dictionary of parameters to be passed to be used for initialization of the model/calculator
    """

    def __init__(
        self,
        model_classes: "list[str]",
        model_paths: "list[str]",
        checkpoint_paths: "list[str]",
        mlp_params: dict = {},
    ) -> None:

        self.model_classes = model_classes
        self.model_paths = model_paths
        self.checkpoint_paths = checkpoint_paths

        self.finetuner_calcs = []
        for i in range(len(self.model_classes)):
            self.finetuner_calcs.append(
                FinetunerCalc(
                    model_name=self.model_classes[i],
                    model_path=self.model_paths[i],
                    checkpoint_path=self.checkpoint_paths[i],
                    mlp_params=copy.deepcopy(mlp_params),
                )
            )

        self.ml_model = False
        if "tuner" not in mlp_params:
            mlp_params["tuner"] = {}
        self.ensemble_method = mlp_params["tuner"].get("ensemble_method", "mean")
        MLPCalc.__init__(self, mlp_params=mlp_params)

    def init_model(self):
        self.model_name = "ensemble"
        self.ml_model = True

        for finetuner in self.finetuner_calcs:
            finetuner.init_model()

    def train_ocp(self, dataset):
        for finetuner in self.finetuner_calcs:
            start = time.time()
            finetuner.train_ocp(dataset)
            end = time.time()
            print(
                "Time to train "
                + str(finetuner.model_name)
                + " on "
                + str(len(dataset))
                + " pts: "
                + str(end - start)
                + " seconds"
            )

    def calculate_ml(self, atoms, properties, system_changes) -> tuple:
        """
        Give ml model the ocp_descriptor to calculate properties : energy, forces, uncertainties.

        Args:
            ocp_descriptor: list object containing the descriptor of the atoms object

        Returns:
            tuple: (energy, forces, energy_uncertainty, force_uncertainties)
        """
        energy_list = []
        forces_list = []
        for finetuner in self.finetuner_calcs:
            energy, forces = finetuner.trainer.get_atoms_prediction(atoms)
            energy_list.append(energy)
            forces_list.append(forces)

        if self.ensemble_method == "mean":
            e_mean = np.mean(energy_list)
            f_mean = np.mean(forces_list, axis=0)
        elif self.ensemble_method == "leader":
            e_mean = energy_list[0]
            f_mean = forces_list[0]
        else:
            raise ValueError("invalid ensemble method provided")

        self.train_counter += 1
        e_std = np.std(energy_list)
        f_stds = np.std(forces_list, axis=0)

        return e_mean, f_mean, e_std, f_stds
