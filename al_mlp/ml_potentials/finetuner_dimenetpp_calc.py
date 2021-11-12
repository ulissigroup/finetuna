from ocpmodels.common.relaxation.ase_utils import OCPCalculator
from al_mlp.ml_potentials.finetuner_calc import FinetunerCalc


class DimenetppFinetunerCalc(FinetunerCalc):
    """
    DimenetppFinetunerCalc.
    ML potential calculator class that implements the partially frozen Dimenetpp: freezing some layers and unfreezing some for finetuning.

    Parameters
    ----------
    model_path: str
        path to Dimenetpp model config, e.g. '/home/jovyan/working/ocp/configs/s2ef/all/dimenet_plus_plus/dpp_forceonly.yml'

    checkpoint_path: str
        path to Dimenetpp model checkpoint, e.g. '/home/jovyan/shared-datasets/OC20/checkpoints/s2ef/dimenetpp_all_forceonly.pt'

    mlp_params: dict
        dictionary of parameters to be passed to be used for initialization of the model/calculator
    """

    def __init__(
        self,
        model_path: str,
        checkpoint_path: str,
        mlp_params: dict = {},
    ) -> None:

        self.model_path = model_path
        self.checkpoint_path = checkpoint_path

        FinetunerCalc.__init__(self, mlp_params=mlp_params)

    def init_model(self):
        self.model_class = "Dimenetpp"
        self.ocp_calc = OCPCalculator(
            config_yml=self.model_path,
            checkpoint=self.checkpoint_path,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
        )

        if not self.energy_training:
            self.ocp_calc.trainer.config["optim"]["energy_coefficient"] = 0

        # freeze certain weights within the loaded model
        for name, param in self.ocp_calc.trainer.model.named_parameters():
            if param.requires_grad:
                if "output_blocks.3" not in name and "output_blocks.2" not in name:
                    param.requires_grad = False
