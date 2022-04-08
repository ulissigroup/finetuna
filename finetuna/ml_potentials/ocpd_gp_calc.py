from sklearn.gaussian_process._gpr import GaussianProcessRegressor
from finetuna.ml_potentials.ocpd_calc import OCPDCalc
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    WhiteKernel,
    RBF,
    Matern,
    RationalQuadratic,
)


class OCPDGPCalc(OCPDCalc):
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
            if kernel_dict is not None:
                self.kernel = ConstantKernel(
                    constant_value=1, constant_value_bounds="fixed"
                )
                for key, value in kernel_dict.items():
                    if key == "ConstantKernel":
                        new_kernel = ConstantKernel(**value["params"])
                    elif key == "WhiteKernel":
                        new_kernel = WhiteKernel(**value["params"])
                    elif key == "RBF":
                        new_kernel = RBF(**value["params"])
                    elif key == "Matern":
                        new_kernel = Matern(**value["params"])
                    elif key == "RationalQuadratic":
                        new_kernel = RationalQuadratic(**value["params"])
                    else:
                        raise ValueError("invalid kernel dict given")

                    if value["operation"] == "sum":
                        self.kernel += new_kernel
                    elif value["operation"] == "product":
                        self.kernel *= new_kernel
                    else:
                        raise ValueError("no/invalid kernel operation given")

        super().__init__(model_path, checkpoint_path, mlp_params=gp_params)

    def init_model(self):
        self.ml_model = True
        self.f_model = GaussianProcessRegressor(kernel=self.kernel, **self.mlp_params)
        self.e_model = GaussianProcessRegressor(kernel=self.kernel, **self.mlp_params)

    def calculate_ml(self, ocp_descriptor) -> tuple:
        f_mean, f_std = self.f_model.predict([ocp_descriptor[1]], return_std=True)
        e_mean, e_std = self.e_model.predict([ocp_descriptor[0]], return_std=True)
        f_mean = f_mean[0].reshape(3, -1).T
        return e_mean[0], f_mean, e_std[0], f_std[0]

    def fit(
        self, parent_energies, parent_forces, parent_e_descriptors, parent_f_descriptors
    ):
        flattened_parent_forces = []
        for forces in parent_forces:
            flattened_parent_forces.append(forces.flatten())
        self.f_model.fit(parent_f_descriptors, flattened_parent_forces)
        self.e_model.fit(parent_e_descriptors, parent_energies)

    def partial_fit(
        self, new_energies, new_forces, new_e_descriptors, new_f_descriptors
    ):
        raise NotImplementedError
