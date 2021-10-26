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

        self.n_hidden = nn_params.get("n_hidden", 20)
        self.n_hidden2 = nn_params.get("n_hidden2", 20)
        self.dropout_prob = nn_params.get("dropout_prob", 0.9)
        self.n_estimators = nn_params.get("n_estimators", 3)

        super().__init__(model_path, checkpoint_path, mlp_params=nn_params)

        self.stopping_epoch = self.mlp_params.get("stopping_epoch", 15)

        self.loss_func = nn.MSELoss()

    def init_model(self):
        self.ml_model = True
        optimizer_params = self.mlp_params.get("optimizer", {})
        self.nn_ensemble = []
        self.optimizers = []
        for i in range(self.n_estimators):
            self.nn_ensemble.append(
                Net(
                    self.n_atoms,
                    self.n_hidden,
                    self.n_hidden2,
                    1,
                    self.dropout_prob,
                )
            )
            self.optimizers.append(
                torch.optim.AdamW(
                    self.nn_ensemble[-1].parameters(),
                    lr=optimizer_params.get("lr", 5e-2),
                    betas=optimizer_params.get("betas", (0.9, 0.999)),
                    eps=optimizer_params.get("eps", 1e-6),
                    weight_decay=optimizer_params.get("weight_decay", 0),
                    amsgrad=optimizer_params.get("amsgrad", True),
                )
            )

        self.f_ensemble = []
        self.f_optimizers = []
        for i in range(self.n_estimators):
            self.f_ensemble.append(
                Net(
                    len(self.get_descriptor(self.initial_structure)[1]),
                    self.n_hidden,
                    self.n_hidden2,
                    self.n_atoms * 3,
                    self.dropout_prob,
                )
            )
            self.f_optimizers.append(
                torch.optim.AdamW(
                    self.nn_ensemble[-1].parameters(),
                    lr=optimizer_params.get("lr", 1e-5),
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
                estimator(torch.tensor(ocp_descriptor[0]))
                .detach()
                .numpy()
            )

        stds = np.std(predictions, axis=0)
        avgs = np.average(predictions, axis=0)

        e_mean = avgs[0]
        e_std = stds[0].item()

        f_predictions = []
        for estimator in self.f_ensemble:
            f_predictions.append(
                estimator(torch.tensor(ocp_descriptor[-1]))
                .detach()
                .numpy()
            )

        stds = np.std(f_predictions, axis=0)
        avgs = np.average(f_predictions, axis=0)

        f_mean = avgs.reshape(self.n_atoms, 3)
        f_std = np.average(stds).item()

        return e_mean, f_mean, e_std, f_std

    def fit(
        self, parent_energies, parent_forces, parent_e_descriptors, parent_f_descriptors
    ):
        n_data = len(parent_energies)

        for j in range(len(self.nn_ensemble)):
            estimator = self.nn_ensemble[j]
            self.epoch = 0
            self.epoch_losses = []
            while not self.stopping_criteria(estimator):
                self.epoch_losses.append(0)
                for i in range(n_data):
                    prediction = estimator(torch.tensor(parent_e_descriptors[i]))
                    loss = self.loss_func(
                        prediction, torch.tensor(np.array([parent_energies[i]])).to(torch.float32)
                    )

                    self.epoch_losses[-1] += loss.data.item()
                    # print(str(self.epoch)+"loss: "+str(loss))

                    self.optimizers[j].zero_grad()
                    loss.backward()
                    self.optimizers[j].step()
                if self.epoch%20 == 0:
                    print(str(self.epoch)+" energy sum_loss: "+str(self.epoch_losses[-1]))
                self.epoch += 1

        for j in range(len(self.f_ensemble)):
            estimator = self.f_ensemble[j]
            self.epoch = 0
            self.epoch_losses = []
            while not self.stopping_criteria(estimator):
                self.epoch_losses.append(0)
                for i in range(n_data):
                    prediction = estimator(torch.tensor(parent_f_descriptors[i]))
                    loss = self.loss_func(
                        prediction, torch.tensor(parent_forces[i].flatten()).to(torch.float32)
                    )

                    self.epoch_losses[-1] += loss.data.item()
                    # print(str(self.epoch)+"loss: "+str(loss))

                    self.optimizers[j].zero_grad()
                    loss.backward()
                    self.optimizers[j].step()
                if self.epoch%20 == 0:
                    print(str(self.epoch)+" forces sum_loss: "+str(self.epoch_losses[-1]))
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
        # return self.epoch > self.stopping_epoch
        if len(self.epoch_losses) > 1000:
            return True
        if len(self.epoch_losses) < 10:
            return False
        if self.epoch_losses[-1] > 1:
            return False
        return np.abs(np.mean(self.epoch_losses[-30:-1])-np.mean(self.epoch_losses[-5:-1])) < 0.03


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
