from collections.abc import Iterable
from ase.calculators.calculator import Calculator, all_changes
from ase.atoms import Atoms
from al_mlp.ocp_descriptor import OCPDescriptor
import numpy as np


class OCPDCalc(Calculator):
    """
    Open Catalyst Project Descriptor Calculator.
    This class serves as a parent class for calculators that want to inherit calculate() and train()
    using descriptors from OCP models.

    Parameters
    ----------
    model_path: str
        path to gemnet model config, e.g. '/home/jovyan/working/ocp/configs/s2ef/all/gemnet/gemnet-dT.yml'

    checkpoint_path: str
        path to gemnet model checkpoint, e.g. '/home/jovyan/shared-datasets/OC20/checkpoints/s2ef/gemnet_t_direct_h512_all.pt'

    ml_params: dict
        dictionary of parameters to be passed to the ml potential model in init_model()
    """

    implemented_properties = ["energy", "forces", "stds"]

    def __init__(
        self,
        model_path: str,
        checkpoint_path: str,
        ml_params: dict = {},
    ):
        super().__init__()

        self.ocp_describer = OCPDescriptor(
            config_yml=model_path,
            checkpoint=checkpoint_path,
        )

        self.ml_params = ml_params
        self.init_model()

    def init_model(self):
        """
        initialize a new ml model using the stored parameter dictionary
        """
        raise NotImplementedError

    def calculate_ml(self, ocp_descriptor) -> tuple:
        """
        Give ml model the ocp_descriptor to calculate properties : energy, forces, uncertainties.

        Args:
            ocp_descriptor: list object containing the descriptor of the atoms object

        Returns:
            tuple: (energy, forces, [energy_uncertainty, *force_uncertainties])
        """
        raise NotImplementedError

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Calculate properties including: energy, forces, uncertainties.

        Args:
            atoms: ase Atoms object
        """
        super().calculate(
            atoms=atoms, properties=properties, system_changes=system_changes
        )

        if properties is None:
            properties = self.implemented_properties

        ocp_descriptor = self.ocp_describer.gemnet_forward(atoms)
        energy, forces, uncertainties = self.calculate_ml(ocp_descriptor)

        self.results["energy"] = energy
        self.results["forces"] = forces
        self.results["stds"] = uncertainties
        self.results["force_stds"] = uncertainties[1:]
        self.results["energy_stds"] = uncertainties[0]
        atoms.info["energy_stds"] = self.results["energy_stds"]
        atoms.info["max_force_stds"] = np.nanmax(self.results["force_stds"])
        return

    def fit(self, parent_energies, parent_forces, parent_descriptors):
        """
        fit a new model on the parent dataset,

        Args:
            parent_energies: list of the energies to fit on
            parent_forces: list of the forces to fit on
            parent_descriptors: list of the descriptors to fit on
        """
        raise NotImplementedError

    def partial_fit(self, new_energies, new_forces, new_descriptors):
        """
        partial fit the current model on just the new_dataset

        Args:
            new_energies: list of just the new energies to partially fit on
            new_forces: list of just the new forces to partially fit on
            new_descriptors: list of just the new descriptors to partially fit on
        """
        raise NotImplementedError

    def train(
        self, parent_dataset: Iterable[Atoms], new_dataset: Iterable[Atoms] = None
    ):
        """
        Train the ml model by fitting a new model on the parent dataset,
        or partial fit the current model on just the new_dataset

        Args:
            parent_dataset: list of all the descriptors to be trained on

            new_dataset: list of just the new descriptors to partially fit on
        """
        if not self.ml_model or not new_dataset:
            self.init_model()
            self.fit(self.get_data_from_atoms(parent_dataset))
        else:
            self.partial_fit(self.get_data_from_atoms(new_dataset))

    def get_data_from_atoms(self, atoms_dataset: Iterable[Atoms]):
        energy_data = []
        forces_data = []
        descriptor_data = []
        for atoms in atoms_dataset:
            energy_data.append(atoms.get_potential_energy())
            forces_data.append(atoms.get_forces)
            descriptor_data.append(self.get_descriptor(atoms))
        return energy_data, forces_data, descriptor_data

    def get_descriptor(self, atoms: Atoms):
        """"
        Overwritable method for getting the ocp descriptor from atoms objects
        """
        ocp_descriptor = self.ocp_describer.gemnet_forward(atoms)
        e_desc = ocp_descriptor[0].detach().numpy()
        return e_desc
