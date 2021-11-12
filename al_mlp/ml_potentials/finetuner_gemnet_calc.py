from ocpmodels.common.relaxation.ase_utils import OCPCalculator
from al_mlp.ml_potentials.finetuner_calc import FinetunerCalc
from ocpmodels.models.gemnet.layers.interaction_block import (
    InteractionBlockTripletsOnly as interaction_block,
)
from ocpmodels.models.gemnet.layers.atom_update_block import OutputBlock
import torch


class GemnetFinetunerCalc(FinetunerCalc):
    """
    GemnetFinetunerCalc.
    ML potential calculator class that implements the partially frozen gemnet: freezing some layers and unfreezing some for finetuning.

    Parameters
    ----------
    model_path: str
        path to gemnet model config, e.g. '/home/jovyan/working/ocp/configs/s2ef/all/gemnet/gemnet-dT.yml'

    checkpoint_path: str
        path to gemnet model checkpoint, e.g. '/home/jovyan/shared-datasets/OC20/checkpoints/s2ef/gemnet_t_direct_h512_all.pt'

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
        self.model_class = "Gemnet"
        self.ocp_calc = OCPCalculator(
            config_yml=self.model_path,
            checkpoint=self.checkpoint_path,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
        )

        # freeze certain weights within the loaded model
        for name, param in self.ocp_calc.trainer.model.named_parameters():
            if param.requires_grad:
                if "out_blocks.3" not in name:
                    param.requires_grad = False

        self.ocp_calc.trainer.model.after_freeze_IB = torch.nn.ModuleList(
            [
                interaction_block(
                    emb_size_atom=self.ocp_calc.trainer.config["model_attributes"].get(
                        "emb_size_atom"
                    ),
                    emb_size_edge=self.ocp_calc.trainer.config["model_attributes"].get(
                        "emb_size_edge"
                    ),
                    emb_size_trip=self.ocp_calc.trainer.config["model_attributes"].get(
                        "emb_size_trip"
                    ),
                    emb_size_rbf=self.ocp_calc.trainer.config["model_attributes"].get(
                        "emb_size_rbf"
                    ),
                    emb_size_cbf=self.ocp_calc.trainer.config["model_attributes"].get(
                        "emb_size_cbf"
                    ),
                    emb_size_bil_trip=self.ocp_calc.trainer.config[
                        "model_attributes"
                    ].get("emb_size_bil_trip"),
                    num_before_skip=self.ocp_calc.trainer.config[
                        "model_attributes"
                    ].get("num_before_skip"),
                    num_after_skip=self.ocp_calc.trainer.config["model_attributes"].get(
                        "num_after_skip"
                    ),
                    num_concat=self.ocp_calc.trainer.config["model_attributes"].get(
                        "num_concat"
                    ),
                    num_atom=self.ocp_calc.trainer.config["model_attributes"].get(
                        "num_atom"
                    ),
                    activation=self.ocp_calc.trainer.config["model_attributes"].get(
                        "activation", "swish"
                    ),
                    scale_file=self.ocp_calc.trainer.config["model_attributes"].get(
                        "scale_file", None
                    ),
                    name=f"AfterFreezeIntBlock_{i+1}",
                )
                for i in range(self.ocp_calc.trainer.model.after_freeze_numblocks)
            ]
        )
        self.ocp_calc.trainer.model.after_freeze_OB = torch.nn.ModuleList(
            [
                OutputBlock(
                    emb_size_atom=self.ocp_calc.trainer.config["model_attributes"].get(
                        "emb_size_atom"
                    ),
                    emb_size_edge=self.ocp_calc.trainer.config["model_attributes"].get(
                        "emb_size_edge"
                    ),
                    emb_size_rbf=self.ocp_calc.trainer.config["model_attributes"].get(
                        "emb_size_rbf"
                    ),
                    nHidden=self.ocp_calc.trainer.config["model_attributes"].get(
                        "num_atom"
                    ),
                    num_targets=self.ocp_calc.trainer.config["model_attributes"].get(
                        "num_targets"
                    ),
                    activation=self.ocp_calc.trainer.config["model_attributes"].get(
                        "activation", "swish"
                    ),
                    output_init=self.ocp_calc.trainer.config["model_attributes"].get(
                        "output_init", "HeOrthogonal"
                    ),
                    direct_forces=self.ocp_calc.trainer.config["model_attributes"].get(
                        "direct_forces", False
                    ),
                    scale_file=self.ocp_calc.trainer.config["model_attributes"].get(
                        "scale_file", None
                    ),
                    name=f"AfterFreezeOutBlock_{i}",
                )
                for i in range(self.ocp_calc.trainer.model.after_freeze_numblocks)
            ]
        )
