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

        self.n_hidden = nn_params.get("n_hidden", 10)
        self.n_hidden2 = nn_params.get("n_hidden2", 10)
        self.dropout_prob = nn_params.get("dropout_prob", 0.5)
        self.n_estimators = nn_params.get("n_estimators", 5)

        super().__init__(model_path, checkpoint_path, mlp_params=nn_params)

        self.stopping_epoch = self.mlp_params.get("stopping_epoch", 5)

        self.loss_func = nn.MSELoss()

    def init_model(self):
        self.ml_model = True
        optimizer_params = self.mlp_params.get("optimizer", {})
        self.nn_ensemble = []
        self.optimizers = []
        for i in range(self.n_estimators):
            self.nn_ensemble.append(
                Net(
                    self.n_atoms + len(self.get_descriptor(self.initial_structure)[1]),
                    self.n_hidden,
                    self.n_hidden2,
                    self.n_atoms * 3 + 1,
                    self.dropout_prob,
                )
            )
            self.optimizers.append(
                torch.optim.AdamW(
                    self.nn_ensemble[-1].parameters(),
                    lr=optimizer_params.get("lr", 1e-3),
                    betas=optimizer_params.get("betas", (0.9, 0.999)),
                    eps=optimizer_params.get("eps", 1e-6),
                    weight_decay=optimizer_params.get("weight_decay", 0),
                    amsgrad=optimizer_params.get("amsgrad", True),
                )
            )

    def calculate_ml(self, ocp_descriptor) -> tuple:
        predictions = []
        for estimator in self.nn_ensemble:
            predictions.append(
                estimator(torch.tensor(self.get_unified_descriptor(*ocp_descriptor)))
                .detach()
                .numpy()
            )

        stds = np.std(predictions, axis=0)
        avgs = np.average(predictions, axis=0)

        e_mean = avgs[0]
        f_mean = avgs[1:].reshape(self.n_atoms, 3)
        e_std = stds[0].item()
        f_std = np.average(stds[1:]).item()

        return e_mean, f_mean, e_std, f_std

    def fit(
        self, parent_energies, parent_forces, parent_e_descriptors, parent_f_descriptors
    ):
        n_data = len(parent_energies)

        unified_labels = []
        unified_descriptors = []
        for i in range(n_data):
            unified_descriptors.append(
                self.get_unified_descriptor(
                    parent_e_descriptors[i], parent_f_descriptors[i]
                )
            )
            unified_labels.append(
                self.get_unified_label(parent_energies[i], parent_forces[i])
            )

        for j in range(len(self.nn_ensemble)):
            estimator = self.nn_ensemble[j]
            self.epoch = 0
            while not self.stopping_criteria(estimator):
                for i in range(n_data):
                    prediction = estimator(torch.tensor(unified_descriptors[i]))
                    loss = self.loss_func(
                        prediction, torch.tensor(unified_labels[i]).to(torch.float32)
                    )

                    loss.backward()

                    self.optimizers[j].step()
                    self.optimizers[j].zero_grad()
                self.epoch += 1

    def partial_fit(
        self, new_energies, new_forces, new_e_descriptors, new_f_descriptors
    ):
        raise NotImplementedError

    def get_unified_descriptor(self, e_descriptor, f_descriptor):
        return np.concatenate((e_descriptor, f_descriptor.flatten()))

    def get_unified_label(self, energy, forces):
        return np.concatenate((np.array([energy]), forces.flatten()))

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
