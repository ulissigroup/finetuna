from flare.ase.calculator import FLARE_Calculator


class FlareOCPDescriptorCalc(FLARE_Calculator):

    implemented_properties = ["energy", "forces", "stress", "stds"]

    def __init__(self, gp_model, mgp_model=None, par=False, use_mapping=False, **kwargs):
        super().__init__(gp_model, mgp_model=mgp_model, par=par, use_mapping=use_mapping, **kwargs)
