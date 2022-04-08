from ase.calculators.calculator import all_changes
from ase.atoms import Atoms
from finetuna.ml_potentials.ml_potential_calc import MLPCalc
from finetuna.ocp_descriptor import OCPDescriptor


class OCPDCalc(MLPCalc):
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

    mlp_params: dict
        dictionary of parameters to be passed to the ml potential model in init_model()
    """

    implemented_properties = ["energy", "forces", "stds"]

    def __init__(
        self,
        model_path: str,
        checkpoint_path: str,
        mlp_params: dict = {},
    ):
        MLPCalc.__init__(self, mlp_params=mlp_params)

        self.ocp_describer = OCPDescriptor(
            model_path=model_path,
            checkpoint_path=checkpoint_path,
        )

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
            tuple: (energy, forces, energy_uncertainty, force_uncertainties)
        """
        raise NotImplementedError

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Calculate properties including: energy, forces, uncertainties.

        Args:
            atoms: ase Atoms object
        """
        MLPCalc.calculate(
            self, atoms=atoms, properties=properties, system_changes=system_changes
        )

        ocp_descriptor = self.get_descriptor(atoms)
        energy, forces, energy_uncertainty, force_uncertainties = self.calculate_ml(
            ocp_descriptor
        )

        self.results["energy"] = energy
        self.results["forces"] = forces
        self.results["stds"] = [energy_uncertainty, force_uncertainties]
        self.results["force_stds"] = force_uncertainties
        self.results["energy_stds"] = energy_uncertainty
        atoms.info["energy_stds"] = self.results["energy_stds"]
        atoms.info["max_force_stds"] = self.results["force_stds"]
        # atoms.info["max_force_stds"] = np.nanmax(self.results["force_stds"])
        return

    def fit(
        self, parent_energies, parent_forces, parent_e_descriptors, parent_f_descriptors
    ):
        """
        fit a new model on the parent dataset,

        Args:
            parent_energies: list of the energies to fit on
            parent_forces: list of the forces to fit on
            parent_e_descriptors: list of the energy descriptors to fit on
            parent_f_descriptors: list of the forces descriptors to fit on
        """
        raise NotImplementedError

    def partial_fit(
        self, new_energies, new_forces, new_e_descriptors, new_f_descriptors
    ):
        """
        partial fit the current model on just the new_dataset

        Args:
            new_energies: list of just the new energies to partially fit on
            new_forces: list of just the new forces to partially fit on
            new_e_descriptors: list of just the new energy descriptors to partially fit on
            new_f_descriptors: list of just the new forces descriptors to partially fit on
        """
        raise NotImplementedError

    def train(self, parent_dataset: "list[Atoms]", new_dataset: "list[Atoms]" = None):
        """
        Train the ml model by fitting a new model on the parent dataset,
        or partial fit the current model on just the new_dataset

        Args:
            parent_dataset: list of all the descriptors to be trained on

            new_dataset: list of just the new descriptors to partially fit on
        """
        if not self.ml_model or not new_dataset:
            self.init_model()
            self.fit(*self.get_data_from_atoms(parent_dataset))
        else:
            self.partial_fit(*self.get_data_from_atoms(new_dataset))

    def get_data_from_atoms(self, atoms_dataset: "list[Atoms]"):
        energy_data = []
        forces_data = []
        e_descriptor_data = []
        f_descriptor_data = []
        for atoms in atoms_dataset:
            energy_data.append(atoms.get_potential_energy())
            forces_data.append(atoms.get_forces())
            ocp_descriptor = self.get_descriptor(atoms)
            e_descriptor_data.append(ocp_descriptor[0])
            f_descriptor_data.append(ocp_descriptor[1])
        return energy_data, forces_data, e_descriptor_data, f_descriptor_data

    def get_descriptor(self, atoms: Atoms):
        """ "
        Overwritable method for getting the ocp descriptor from atoms objects
        """
        ocp_descriptor = self.ocp_describer.gemnet_forward(atoms)
        e_desc = ocp_descriptor[0].detach().numpy().flatten()
        f_desc = ocp_descriptor[1].detach().numpy().flatten()
        return e_desc, f_desc
