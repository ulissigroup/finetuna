from finetuna.ml_potentials.ocpd_calc import OCPDCalc
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import copy
from multiprocessing import Pool
from ase.atoms import Atoms


class OCPDNNCalc(OCPDCalc):
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

        self.n_hidden = nn_params.get("n_hidden", 2000)
        self.n_hidden2 = nn_params.get("n_hidden2", 200)
        self.dropout_prob = nn_params.get("dropout_prob", 0)
        self.n_estimators = nn_params.get("n_estimators", 3)
        self.verbose = nn_params.get("verbose", True)

        super().__init__(model_path, checkpoint_path, mlp_params=nn_params)

        self.stopping_epoch = self.mlp_params.get("stopping_epoch", 100)
        self.parallel = self.mlp_params.get("parallel", False)
        if self.parallel:
            self.process_pool = Pool(self.parallel)

        self.loss_func = nn.MSELoss()

    def init_model(self):
        self.ml_model = True
        self.nn_ensemble = []
        self.optimizers = []
        self.schedulers = []
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
            self.init_optimizer()
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
        self.optimizers.append(optimizer)

        scheduler_class = self.mlp_params.get("scheduler", None)
        if scheduler_class == "ReduceLROnPlateau":
            scheduler_dict = self.mlp_params.get("scheduler_dict", {})
            self.schedulers.append(
                torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_dict)
            )

    def calculate_ml(self, ocp_descriptor) -> tuple:
        e_mean = self.mean_energy
        e_std = self.std_energy

        if self.initial_structure.constraints:
            constraints_index = self.initial_structure.constraints[0].index
        else:
            constraints_index = []

        predictions = []
        for estimator in self.nn_ensemble:
            prediction = estimator(torch.tensor(ocp_descriptor)).detach().numpy()
            constraint_array = np.ones((self.n_atoms, 3))
            constraint_array[constraints_index] = np.zeros((3,))
            constraint_array = constraint_array.flatten()
            prediction = np.multiply(constraint_array, prediction)
            predictions.append(prediction)

        stds = np.std(predictions, axis=0)
        avgs = np.average(predictions, axis=0)

        f_mean = avgs.reshape(self.n_atoms, 3)
        f_std = np.average(
            np.delete(
                stds.reshape(self.n_atoms, 3),
                constraints_index,
                axis=0,
            )
        ).item()

        return e_mean, f_mean, e_std, f_std

    def fit(self, parent_energies, parent_forces, parent_h_descriptors):
        args_list = []
        for j in range(self.n_estimators):
            parent_energies_copy = copy.deepcopy(parent_energies)
            parent_forces_copy = copy.deepcopy(parent_forces)
            parent_h_descriptors_copy = copy.deepcopy(parent_h_descriptors)
            estimator = self.nn_ensemble[j]
            optimizer = self.optimizers[j]
            if self.initial_structure.constraints:
                constraints_index = self.initial_structure.constraints[0].index
            else:
                constraints_index = []
            if self.schedulers:
                scheduler = self.schedulers[j]
            else:
                scheduler = None
            args_list.append(
                (
                    j,
                    parent_energies_copy,
                    parent_forces_copy,
                    parent_h_descriptors_copy,
                    estimator,
                    optimizer,
                    scheduler,
                    self.verbose,
                    self.stopping_epoch,
                    self.n_atoms,
                    constraints_index,
                    self.loss_func,
                )
            )

        if self.parallel:
            results_iterator = self.process_pool.starmap(sub_fit, args_list)
            self.nn_ensemble = [model for model in results_iterator]
        else:
            for j in range(self.n_estimators):
                self.nn_ensemble[j] = sub_fit(*args_list[j])

        # energy predictions are irrelevant for active learning so just take the mean
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


def sub_fit(
    j,
    parent_energies,
    parent_forces,
    parent_h_descriptors,
    estimator,
    optimizer,
    scheduler,
    verbose,
    stopping_epoch,
    n_atoms,
    constraints_index,
    loss_function,
):
    n_data = len(parent_energies)

    epoch = 0
    best_loss = np.Inf
    best_model = copy.deepcopy(estimator)
    epoch_losses = []

    if verbose:
        print("*loss(" + str(j) + "," + str(epoch) + "): " + str("Inf"))
        # for param in estimator.hidden1.parameters():
        #     print(param.detach().numpy().sum())
        # for param in estimator.hidden2.parameters():
        #     print(param.detach().numpy().sum())

    while not epoch > stopping_epoch:
        epoch_losses.append(0)
        for i in range(n_data):
            prediction = estimator(torch.tensor(parent_h_descriptors[i]))
            constraint_array = np.ones((n_atoms, 3))
            constraint_array[constraints_index] = np.zeros((3,))
            constraint_tensor = torch.tensor(constraint_array.flatten()).to(
                torch.float32
            )
            loss = loss_function(
                prediction * constraint_tensor,
                torch.tensor(parent_forces[i].flatten()).to(torch.float32),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses[-1] += loss.data.item()
        if scheduler:
            scheduler.step(epoch_losses[-1])
        epoch += 1

        loss_str = " "
        if epoch_losses[-1] < best_loss:
            best_loss = epoch_losses[-1]
            best_model = copy.deepcopy(estimator)
            loss_str = "*"
        loss_str += (
            "loss("
            + str(j)
            + ","
            + str(epoch)
            + "): "
            + str(epoch_losses[-1])
            + ",\tlr: "
            + str(optimizer.param_groups[0]["lr"])
        )

        if verbose and epoch % 100 == 0:
            print(loss_str)
            # for param in estimator.hidden1.parameters():
            #     print(param.detach().numpy().sum())
            # for param in estimator.hidden2.parameters():
            #     print(param.detach().numpy().sum())

    return best_model


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
        x = F.silu(self.hidden1(x))  # activation function for hidden layer
        x = F.silu(self.hidden2(x))  # activation function for hidden layer
        x = self.dropout(self.predict(x))
        return x
