from flare.ase.calculator import FLARE_Calculator
from flare.gp import GaussianProcess
from flare.struc import Structure
import numpy as np

from finetuna.ml_potentials.ml_potential_calc import MLPCalc


class FlareCalc(FLARE_Calculator, MLPCalc):

    implemented_properties = ["energy", "forces", "stress", "stds"]

    def __init__(
        self,
        flare_params: dict,
        initial_images,
        mgp_model=None,
        par=False,
        use_mapping=False,
        **kwargs
    ):
        self.initial_images = initial_images
        self.init_species_map()
        MLPCalc.__init__(self, mlp_params=flare_params)
        super().__init__(
            None, mgp_model=mgp_model, par=par, use_mapping=use_mapping, **kwargs
        )

    def init_flare(self):
        self.gp_model = GaussianProcess(**self.mlp_params)

    def init_species_map(self):
        self.species_map = {}
        a_numbers = []
        for image in self.initial_images:
            a_numbers += np.unique(image.numbers).tolist()
        a_numbers = np.unique(a_numbers)
        for i in range(len(a_numbers)):
            self.species_map[a_numbers[i]] = i

    def calculate(self, atoms=None, properties=None, system_changes=...):
        MLPCalc.calculate(
            self, atoms=atoms, properties=properties, system_changes=system_changes
        )
        return super().calculate(
            atoms=atoms, properties=properties, system_changes=system_changes
        )

    def calculate_gp(self, atoms):
        structure = self.get_descriptor_from_atoms(atoms)
        super().calculate_gp(structure)

        self.results["force_stds"] = self.results["stds"]
        self.results["energy_stds"] = self.results["local_energy_stds"]
        atoms.info["energy_stds"] = self.results["local_energy_stds"]
        atoms.info["max_force_stds"] = np.nanmax(self.results["force_stds"])

    def train(self, parent_dataset, new_dataset=None):
        if not self.gp_model or not new_dataset:
            self.init_flare()
            self.train_on_dataset(parent_dataset)
        else:
            self.train_on_dataset(new_dataset)

    def train_on_dataset(self, dataset):
        for atoms in dataset:
            structure = self.get_descriptor_from_atoms(
                atoms, energy=atoms.get_potential_energy(), forces=atoms.get_forces()
            )
            self.gp_model.update_db(
                struc=structure,
                forces=atoms.get_forces(),
                energy=atoms.get_potential_energy(),
            )

    def get_descriptor_from_atoms(self, atoms, energy=None, forces=None):
        structure = Structure(
            cell=atoms.get_cell(),
            species=[self.species_map[x] for x in atoms.get_atomic_numbers()],
            positions=atoms.get_positions(),
            forces=forces,
            energy=energy,
        )
        return structure
