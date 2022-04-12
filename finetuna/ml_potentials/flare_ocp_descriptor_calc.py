from finetuna.ml_potentials.flare_calc import FlareCalc
from finetuna.ocp_descriptor import OCPDescriptor


class FlareOCPDescriptorCalc(FlareCalc):
    implemented_properties = ["energy", "forces", "stress", "stds"]

    def __init__(
        self,
        model_path: str,
        checkpoint_path: str,
        flare_params: dict,
        initial_images,
        mgp_model=None,
        par=False,
        use_mapping=False,
        **kwargs
    ):
        self.ocp_describer = OCPDescriptor(
            model_path=model_path,
            checkpoint_path=checkpoint_path,
        )

        super().__init__(
            flare_params,
            initial_images,
            mgp_model=mgp_model,
            par=par,
            use_mapping=use_mapping,
            **kwargs
        )

    def get_descriptor_from_atoms(self, atoms, energy=None, forces=None):
        ocp_descriptor = self.ocp_describer.gemnet_forward(atoms)
        e_desc = ocp_descriptor[0].detach().numpy()

        atoms_copy = atoms.copy()
        atoms_copy.calc = atoms.calc
        atoms_copy.positions = e_desc
        structure = super().get_descriptor_from_atoms(
            atoms_copy, energy=energy, forces=forces
        )

        return structure
