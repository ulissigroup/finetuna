from sklearn.gaussian_process._gpr import GaussianProcessRegressor
from al_mlp.ml_potentials.ocpd_calc import OCPDCalc
import numpy as np


class GPOCPDCalc(OCPDCalc):
    implemented_properties = ["energy", "forces", "stds"]

    def __init__(
        self,
        model_path: str,
        checkpoint_path: str,
        gp_params: dict = {},
    ):
        self.kernel = None
        if "kernel" in gp_params:
            kernel_dict = gp_params.pop("kernel")
            if kernel_dict is None:
                self.kernel = None
            else:
                raise ValueError("invalid kernel dict given")

        super().__init__(model_path, checkpoint_path, mlp_params=gp_params)

    def init_model(self):
        self.ml_model = True
        self.f_model = GaussianProcessRegressor(kernel=self.kernel, **self.mlp_params)
        self.e_model = GaussianProcessRegressor(kernel=self.kernel, **self.mlp_params)

    def calculate_ml(self, ocp_descriptor) -> tuple:
        f_mean, f_std = self.f_model.predict([ocp_descriptor], return_std=True)
        e_mean, e_std = self.e_model.predict([ocp_descriptor], return_std=True)
        f_mean = f_mean[0].reshape(3, -1).T
        return e_mean[0], f_mean, e_std[0], f_std[0]

    def fit(self, parent_energies, parent_forces, parent_descriptors):
        flattened_parent_forces = []
        for forces in parent_forces:
            flattened_parent_forces.append(forces.flatten())
        self.f_model.fit(parent_descriptors, flattened_parent_forces)
        self.e_model.fit(parent_descriptors, parent_energies)

    def partial_fit(self, new_energies, new_forces, new_descriptors):
        raise NotImplementedError
