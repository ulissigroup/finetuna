from al_mlp.ml_potentials.finetuner_calc import FinetunerCalc
from ocpmodels.datasets.trajectory_lmdb import data_list_collater
from .stochastic_spinconv_model import spinconv
import numpy as np


class FinetunerStochasticSpinconvCalc(FinetunerCalc):
    """
    FinetunerStochasticSpinconvCalc
    ML potential calculator class that implements the partially frozen Stochastic Spinconv:
        -freezing some layers and unfreezing some for finetuning
        -handles uncertainty all on its own using its own internal stochasticity

    Parameters
    ----------
    model_path: str
        path to Spinconv model config, e.g. '/home/jovyan/working/al_mlp/al_mlp/ml_potentials/stochastic_spinconv/stochastic_spinconv.yml'

    checkpoint_path: str
        path to Spinconv model checkpoint, e.g. '/home/jovyan/shared-scratch/adeesh/uncertainty/spcv-2M-uncertainty-cp.pt'

    mlp_params: dict
        dictionary of parameters to be passed to be used for initialization of the model/calculator
    """

    def __init__(
        self,
        model_path: str,
        checkpoint_path: str,
        mlp_params: dict = {},
    ) -> None:
        FinetunerCalc.__init__(
            self,
            model_name="spinconv",
            model_path=model_path,
            checkpoint_path=checkpoint_path,
            mlp_params=mlp_params,
        )

    def calculate_ml(self, atoms, properties, system_changes) -> tuple:
        """
        Give ml model atoms object by calling ocp_calc to calculate properties : energy, forces, uncertainties.

        Args:
            ocp_descriptor: list object containing the descriptor of the atoms object

        Returns:
            tuple: (energy, forces, energy_uncertainty, force_uncertainties)
        """
        self.train_counter += 1

        data_object = self.ocp_calc.a2g.convert(atoms)
        batch = data_list_collater([data_object])

        energy, forces = self.ocp_calc.trainer.model([batch])
        e_mean = energy.detach().numpy()[0][0]
        f_mean = forces.detach().numpy()

        forces_uncertainty = self.ocp_calc.trainer.model.module.forces_uncertainty
        f_stds = forces_uncertainty.detach().numpy()

        e_std = 0
        f_std = np.average(f_stds).item()

        return e_mean, f_mean, e_std, f_std
