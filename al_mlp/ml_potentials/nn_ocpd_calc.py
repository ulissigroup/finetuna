from al_mlp.ml_potentials.ocpd_calc import OCPDCalc
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class NNOCPDCalc(OCPDCalc):
    implemented_properties = ["energy", "forces", "stds"]

    def __init__(
        self,
        initial_structure,
        model_path: str,
        checkpoint_path: str,
        nn_params: dict = {},
    ):
        self.initial_structure = initial_structure
        self.n_atoms = len(self.initial_structure)

        self.n_hidden = nn_params.get("n_hidden", 2)
        self.n_hidden2 = nn_params.get("n_hidden2", 2)
        self.dropout_prob = nn_params.get("dropout_prob", 0.9)
        self.n_estimators = nn_params.get("n_estimators", 3)

        super().__init__(model_path, checkpoint_path, mlp_params=nn_params)

        self.stopping_epoch = self.mlp_params.get("stopping_epoch", 100)

        self.loss_func = nn.MSELoss()

    def init_model(self):
        self.ml_model = True
        self.nn_ensemble = []
        self.optimizers = []
        for i in range(self.n_estimators):
            self.nn_ensemble.append(
                Net(
                    len(self.get_descriptor(self.initial_structure)),
                    self.n_hidden,
                    self.n_hidden2,
                    self.n_atoms * 3,
                    self.dropout_prob,
                )
            )
            self.optimizers.append(self.init_optimizer())
        self.mean_energy = 0
        self.std_energy = 0

    def init_optimizer(self):
        optimizer_class = self.mlp_params.get("optimizer", "AdamW")
        if optimizer_class == "AdamW":
            optimizer = torch.optim.AdamW(
                    self.nn_ensemble[-1].parameters(),
                    lr=self.mlp_params.get("lr", 1e-3),
                    betas=self.mlp_params.get("betas", (0.9, 0.999)),
                    eps=self.mlp_params.get("eps", 1e-6),
                    weight_decay=self.mlp_params.get("weight_decay", 0),
                    amsgrad=self.mlp_params.get("amsgrad", True),
                )
        elif optimizer_class == "SGD":
            optimizer = torch.optim.SGD(
                self.nn_ensemble[-1].parameters(),
                lr=self.mlp_params.get("lr", 1e-3),
                momentum=self.mlp_params.get("momentum", 0),
                dampening=self.mlp_params.get("dampening", 0),
                weight_decay=self.mlp_params.get("weight_decay", 0),
                nesterov=self.mlp_params.get("nesterov", False),
            )
        return optimizer
        

    def calculate_ml(self, ocp_descriptor) -> tuple:
        e_mean = self.mean_energy
        e_std = self.std_energy

        predictions = []
        for estimator in self.nn_ensemble:
            predictions.append(
                estimator(torch.tensor(ocp_descriptor))
                .detach()
                .numpy()
            )

        stds = np.std(predictions, axis=0)
        avgs = np.average(predictions, axis=0)

        f_mean = avgs.reshape(self.n_atoms, 3)
        f_std = np.average(stds).item()

        return e_mean, f_mean, e_std, f_std

    def fit(
        self, parent_energies, parent_forces, parent_h_descriptors
    ):
        n_data = len(parent_energies)

        for j in range(len(self.nn_ensemble)):
            estimator = self.nn_ensemble[j]
            self.epoch = 0
            self.epoch_losses = []
            while not self.stopping_criteria(estimator):
                self.epoch_losses.append(0)
                for i in range(n_data):
                    prediction = estimator(torch.tensor(parent_h_descriptors[i]))
                    loss = self.loss_func(
                        prediction, torch.tensor(parent_forces[i].flatten()).to(torch.float32)
                    )

                    self.epoch_losses[-1] += loss.data.item()

                    self.optimizers[j].zero_grad()
                    loss.backward()
                    self.optimizers[j].step()
                self.epoch += 1
        
        self.mean_energy = np.average(parent_energies)
        self.std_energy = np.std(parent_energies)

    def get_data_from_atoms(self, atoms_dataset: "list[Atoms]"):
        energy_data = []
        forces_data = []
        h_data = []
        for atoms in atoms_dataset:
            energy_data.append(atoms.get_potential_energy())
            forces_data.append(atoms.get_forces())
            h_data.append(self.get_descriptor(atoms))
        return energy_data, forces_data, h_data

    def get_descriptor(self, atoms: "Atoms"):
        """ "
        Overwritable method for getting the ocp descriptor from atoms objects
        """
        ocp_descriptor = self.ocp_describer.get_h(atoms)
        h_desc = ocp_descriptor.flatten()
        return h_desc

    def partial_fit(
        self, new_energies, new_forces, new_e_descriptors, new_f_descriptors
    ):
        raise NotImplementedError

    def stopping_criteria(self, estimator):
        return self.epoch > self.stopping_epoch


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output, dropout_prob):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)  # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden2, n_output)  # output layer
        self.dropout = torch.nn.Dropout(dropout_prob)

        torch.nn.init.xavier_uniform_(self.hidden1.weight)
        torch.nn.init.xavier_uniform_(self.hidden2.weight)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.dropout(x)
        x = F.relu(x)  # activation function for hidden layer
        x = self.hidden2(x)
        x = self.dropout(x)
        x = F.relu(x)  # activation function for hidden layer
        x = self.predict(x)  # linear output
        return x
