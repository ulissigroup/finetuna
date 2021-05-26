# import flare_pp._C_flare as flare_pp
from flare_pp._C_flare import Structure, NormalizedDotProduct, B2

# from flare_pp.sparse_gp_calculator import SGP_Calculator
from flare_pp.sparse_gp import SGP_Wrapper
from ase.calculators.calculator import Calculator, all_changes
from flare import struc
import numpy as np


class FlarePPCalc(Calculator):

    implemented_properties = ["energy", "forces", "stress", "stds"]

    def __init__(self, flare_params, initial_image):
        super().__init__()
        self.gp_model = None
        self.results = {}
        self.use_mapping = False
        self.mgp_model = None
        self.flare_params = flare_params
        self.initial_image = initial_image
        self.init_species_map()
        self.update_gp_mode = self.flare_params.get("update_gp_mode", "all")
        self.update_gp_range = self.flare_params.get("update_gp_range", [])

        # self.init_flare()

    def init_species_map(self):
        self.species_map = {}
        a_numbers = np.unique(self.initial_image[0].numbers)
        for i in range(len(a_numbers)):
            self.species_map[a_numbers[i]] = i

    def init_flare(self):
        self.kernel = NormalizedDotProduct(
            self.flare_params["sigma"], self.flare_params["power"]
        )
        radial_hyps = [0.0, self.flare_params["cutoff"]]
        settings = [len(self.species_map), 12, 3]
        self.B2calc = B2(
            self.flare_params["radial_basis"],
            self.flare_params["cutoff_function"],
            radial_hyps,
            self.flare_params["cutoff_hyps"],
            settings,
        )

        bounds = [
            (None, None),
            (self.flare_params["sigma_e"], None),
            (None, None),
            (None, None),
        ]

        self.gp_model = SGP_Wrapper(
            [self.kernel],
            [self.B2calc],
            self.flare_params["cutoff"],
            self.flare_params["sigma_e"],
            self.flare_params["sigma_f"],
            self.flare_params["sigma_s"],
            self.species_map,
            bounds=bounds,
            stress_training=False,
            variance_type="SOR",
            max_iterations=self.flare_params["max_iterations"],
        )
        self.gp_model.descriptor_calcs = [self.B2calc]
        self.gp_model.kernels = [self.kernel]

    # TODO: Figure out why this is called twice per MD step.
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Calculate properties including: energy, local energies, forces,
            stress, uncertainties.
        """

        super().calculate(
            atoms=atoms, properties=properties, system_changes=system_changes
        )

        if properties is None:
            properties = self.implemented_properties

        # Create structure descriptor.
        structure_descriptor = Structure(
            atoms.get_cell(),
            [self.species_map[x] for x in atoms.get_atomic_numbers()],
            atoms.get_positions(),
            self.gp_model.cutoff,
            self.gp_model.descriptor_calculators,
        )

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
            stds = np.zeros(len(variances))
            for n in range(len(variances)):
                var = variances[n]
                if var > 0:
                    stds[n] = np.sqrt(var)
                else:
                    stds[n] = -np.sqrt(np.abs(var))
            self.results["force_stds"] = stds.reshape(-1, 3)
        # The "local" variance type should be used only if the model has a
        # single atom-centered descriptor.
        # TODO: Generalize this variance type to multiple descriptors.
        #         elif self.gp_model.variance_type == "local":
        #             variances = structure_descriptor.local_uncertainties[0]
        #             sorted_variances = sort_variances(structure_descriptor, variances)
        #             stds = np.zeros(len(sorted_variances))
        #             for n in range(len(sorted_variances)):
        #                 var = sorted_variances[n]
        #                 if var > 0:
        #                     stds[n] = np.sqrt(var)
        #                 else:
        #                     stds[n] = -np.sqrt(np.abs(var))
        #             stds_full = np.zeros((len(sorted_variances), 3))

        #             # Divide by the signal std to get a unitless value.
        #             stds_full[:, 0] = stds / self.gp_model.hyps[0]
        #             self.results["stds"] = stds_full

        atoms.info["max_force_stds"] = np.nanmax(self.results["force_stds"])

    def get_uncertainties(self, atoms):
        return self.get_property("stds", atoms)

    def calculation_required(self, atoms, quantities):
        return True

    def train(self, parent_dataset, new_dataset=None):
        # # Create sparse GP model.
        # sigma = 1.0
        # power = 2
        # kernel = NormalizedDotProduct(sigma, power)
        # cutoff_function = "quadratic"
        # cutoff = 3.0
        # radial_basis = "chebyshev"
        # radial_hyps = [0.0, cutoff]
        # cutoff_hyps = []
        # settings = [len(self.species_map), 12, 3]
        # calc = B2(radial_basis, cutoff_function, radial_hyps, cutoff_hyps, settings)
        # sigma_e = 1.0
        # sigma_f = 0.1
        # sigma_s = 0.0
        # max_iterations = 20
        # bounds = [(None, None), (sigma_e, None), (None, None), (None, None)]
        if not self.gp_model or not new_dataset:
            self.init_flare()
            self.fit(parent_dataset)
        else:
            self.partial_fit(new_dataset)

        # if new_dataset:
        #     print(self.gp_model)
        #     print("partial_fit")
        #     self.partial_fit(new_dataset)
        # else:
        #     self.init_flare()
        #     print(self.gp_model)
        #     self.fit(parent_dataset)
        #     print("fit")

        self.gp_model.train()
        # self.gp_model.descriptor_calcs = [self.B2calc]
        # self.gp_model.kernels = [self.kernel]

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
                update_qr=False,
            )

    def fit(self, parent_data):
        for image in parent_data:
            train_structure = struc.Structure(
                image.get_cell(), image.get_atomic_numbers(), image.get_positions()
            )

            forces = image.get_forces(apply_constraint=False)
            energy = image.get_potential_energy(apply_constraint=False)

            self.gp_model.update_db(
                train_structure, forces, [], energy, mode="all", update_qr=False
            )
