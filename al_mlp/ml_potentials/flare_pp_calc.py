# import flare_pp._C_flare as flare_pp
from flare_pp._C_flare import Structure, NormalizedDotProduct, B2, SquaredExponential

# from flare_pp.sparse_gp_calculator import SGP_Calculator
from flare_pp.sparse_gp import SGP_Wrapper
from ase.calculators.calculator import all_changes
from flare import struc
import numpy as np

from al_mlp.ml_potentials.ml_potential_calc import MLPCalc


class FlarePPCalc(MLPCalc):

    implemented_properties = ["energy", "forces", "stress", "stds"]

    def __init__(self, mlp_params, initial_images):
        MLPCalc.__init__(self, mlp_params=mlp_params)
        self.gp_model = None
        self.results = {}
        self.use_mapping = False
        self.mgp_model = None
        self.initial_images = initial_images
        self.init_species_map()
        self.update_gp_mode = self.mlp_params.get("update_gp_mode", "all")
        self.update_gp_range = self.mlp_params.get("update_gp_range", [])
        self.freeze_hyps = self.mlp_params.get("freeze_hyps", None)
        self.variance_type = self.mlp_params.get("variance_type", "SOR")
        self.opt_method = self.mlp_params.get("opt_method", "BFGS")
        self.kernel_type = self.mlp_params.get("kernel_type", "NormalizedDotProduct")
        self.iteration = 0

    def init_species_map(self):
        self.species_map = {}
        a_numbers = []
        for image in self.initial_images:
            a_numbers += np.unique(image.numbers).tolist()
        a_numbers = np.unique(a_numbers)
        for i in range(len(a_numbers)):
            self.species_map[a_numbers[i]] = i

    def init_flare(self):
        if self.kernel_type == "NormalizedDotProduct":
            self.kernel = NormalizedDotProduct(
                self.mlp_params["sigma"], self.mlp_params["power"]
            )
        elif self.kernel_type == "SquaredExponential":
            self.kernel = SquaredExponential(
                self.mlp_params["sigma"], self.mlp_params["ls"]
            )
        radial_hyps = [0.0, self.mlp_params["cutoff"]]
        settings = [len(self.species_map), 12, 3]
        self.B2calc = B2(
            self.mlp_params["radial_basis"],
            self.mlp_params["cutoff_function"],
            radial_hyps,
            self.mlp_params["cutoff_hyps"],
            settings,
        )
        if self.kernel_type == "SquaredExponential":
            bounds = [
                self.mlp_params.get("bounds", {}).get("sigma", (None, None)),
                self.mlp_params.get("bounds", {}).get(
                    "ls", (self.mlp_params["sigma_e"], None)
                ),
                self.mlp_params.get("bounds", {}).get("sigma_e", (None, None)),
                self.mlp_params.get("bounds", {}).get("sigma_f", (None, None)),
                self.mlp_params.get("bounds", {}).get("sigma_s", (None, None)),
            ]
        else:
            bounds = [
                self.mlp_params.get("bounds", {}).get("sigma", (None, None)),
                self.mlp_params.get("bounds", {}).get("sigma_e", (None, None)),
                self.mlp_params.get("bounds", {}).get("sigma_f", (None, None)),
                self.mlp_params.get("bounds", {}).get("sigma_s", (None, None)),
            ]

        self.gp_model = SGP_Wrapper(
            [self.kernel],
            [self.B2calc],
            self.mlp_params["cutoff"],
            self.mlp_params["sigma_e"],
            self.mlp_params["sigma_f"],
            self.mlp_params["sigma_s"],
            self.species_map,
            variance_type=self.variance_type,
            energy_training=self.mlp_params.get("energy_training", True),
            force_training=self.mlp_params.get("force_training", True),
            stress_training=False,
            max_iterations=self.mlp_params["hpo_max_iterations"],
            opt_method=self.opt_method,
            bounds=bounds,
        )
        self.gp_model.descriptor_calcs = [self.B2calc]
        self.gp_model.kernels = [self.kernel]

    # TODO: Figure out why this is called twice per MD step.
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Calculate properties including: energy, local energies, forces,
            stress, uncertainties.
        """

        MLPCalc.calculate(
            self, atoms=atoms, properties=properties, system_changes=system_changes
        )

        # Create structure descriptor.
        structure_descriptor = self.get_structure_descriptor(atoms)

        #         Predict on structure.
        if self.gp_model.variance_type == "SOR":
            self.gp_model.sparse_gp.predict_SOR(structure_descriptor)
        elif self.gp_model.variance_type == "DTC":
            self.gp_model.sparse_gp.predict_DTC(structure_descriptor)
        elif self.gp_model.variance_type == "local":
            self.gp_model.sparse_gp.predict_local_uncertainties(structure_descriptor)

        self.results["energy"] = structure_descriptor.mean_efs[0]
        self.results["forces"] = structure_descriptor.mean_efs[1:-6].reshape(-1, 3)

        # Convert stress to ASE format.
        flare_stress = structure_descriptor.mean_efs[-6:]
        ase_stress = -np.array(
            [
                flare_stress[0],
                flare_stress[3],
                flare_stress[5],
                flare_stress[4],
                flare_stress[2],
                flare_stress[1],
            ]
        )
        self.results["stress"] = ase_stress

        # Report negative variances, which can arise if there are numerical
        # instabilities.
        if (self.gp_model.variance_type == "SOR") or (
            self.gp_model.variance_type == "DTC"
        ):
            variances = structure_descriptor.variance_efs[1:-6]
            energy_var = structure_descriptor.variance_efs[0]
            energy_std = np.sqrt(np.abs(energy_var))
            stds = np.zeros(len(variances))
            for n in range(len(variances)):
                var = variances[n]
                if var > 0:
                    stds[n] = np.sqrt(var)
                else:
                    stds[n] = -np.sqrt(np.abs(var))
            self.results["force_stds"] = stds.reshape(-1, 3)
            self.results["energy_stds"] = energy_std
            atoms.info["energy_stds"] = energy_std
        # The "local" variance type should be used only if the model has a
        # single atom-centered descriptor.
        # TODO: Generalize this variance type to multiple descriptors.
        elif self.gp_model.variance_type == "local":
            variances = structure_descriptor.local_uncertainties[0]
            sorted_variances = self.sort_variances(structure_descriptor, variances)
            stds = np.zeros(len(sorted_variances))
            for n in range(len(sorted_variances)):
                var = sorted_variances[n]
                if var > 0:
                    stds[n] = np.sqrt(var)
                else:
                    stds[n] = -np.sqrt(np.abs(var))
            stds_full = np.zeros((len(sorted_variances), 3))

            # Divide by the signal std to get a unitless value.
            stds_full[:, 0] = stds / self.gp_model.hyps[0]
            self.results["force_stds"] = stds_full

        atoms.info["max_force_stds"] = np.nanmax(self.results["force_stds"])

    def sort_variances(self, structure_descriptor, variances):
        # Check that the variance length matches the number of atoms.
        assert len(variances) == structure_descriptor.noa
        sorted_variances = np.zeros(len(variances))

        # Sort the variances by atomic order.
        descriptor_values = structure_descriptor.descriptors[0]
        atom_indices = descriptor_values.atom_indices
        n_types = descriptor_values.n_types
        assert n_types == len(atom_indices)

        v_count = 0
        for s in range(n_types):
            for n in range(len(atom_indices[s])):
                atom_index = atom_indices[s][n]
                sorted_variances[atom_index] = variances[v_count]
                v_count += 1

        return sorted_variances

    def get_uncertainties(self, atoms):
        return self.get_property("stds", atoms)

    def calculation_required(self, atoms, quantities):
        return True

    def train(self, parent_dataset, new_dataset=None):
        # # Create sparse GP model.
        if not self.gp_model or not new_dataset:
            self.init_flare()
            self.fit(parent_dataset)
        else:
            self.partial_fit(new_dataset)

        # start_time = time.time()
        if isinstance(self.freeze_hyps, int) and self.iteration < self.freeze_hyps:
            # print("freeze_hyps = ", self.freeze_hyps)
            self.gp_model.train()
            self.iteration += 1
            # print("---training time %s min ---" % ((time.time() - start_time)/60))
            return
        elif self.freeze_hyps == 0:
            return
        elif not self.freeze_hyps:
            # print("freeze hyps not set")
            self.gp_model.train()
            self.iteration += 1
            # print("---training time %s min ---" % ((time.time() - start_time)/60))
        return

    def partial_fit(self, new_dataset):
        for image in new_dataset:
            train_structure = struc.Structure(
                image.get_cell(), image.get_atomic_numbers(), image.get_positions()
            )
            forces = image.get_forces(apply_constraint=False)
            energy = image.get_potential_energy(apply_constraint=False)
            self.gp_model.update_db(
                train_structure,
                forces,
                self.update_gp_range,
                energy,
                mode=self.update_gp_mode,
                update_qr=True,
            )

    def fit(self, parent_data):
        for image in parent_data:
            train_structure = struc.Structure(
                image.get_cell(), image.get_atomic_numbers(), image.get_positions()
            )

            forces = image.get_forces(apply_constraint=False)
            energy = image.get_potential_energy(apply_constraint=False)

            self.gp_model.update_db(
                train_structure, forces, [], energy, mode="all", update_qr=True
            )

    def get_structure_descriptor(self, atoms):
        structure_descriptor = Structure(
            atoms.get_cell(),
            [self.species_map[x] for x in atoms.get_atomic_numbers()],
            atoms.get_positions(),
            self.gp_model.cutoff,
            self.gp_model.descriptor_calculators,
        )
        return structure_descriptor
