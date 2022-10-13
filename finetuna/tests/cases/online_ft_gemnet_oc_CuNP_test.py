import unittest
from finetuna.tests.setup.base_case_online_CuNP import BaseOnlineCuNP


class online_ft_gemnet_oc_CuNP(BaseOnlineCuNP, unittest.TestCase):
    @classmethod
    def get_al_config(cls) -> dict:
        al_config = BaseOnlineCuNP.get_al_config()
        al_config["links"]["ml_potential"] = "ft"
        al_config["finetuner"] = {
            "tuner": {
                "unfreeze_blocks": [
                    "out_blocks.4.seq_forces",
                    "out_blocks.4.dense_rbf_F",
                    "out_blocks.3.seq_forces",
                    "out_blocks.3.dense_rbf_F",
                    "out_blocks.2.seq_forces",
                    "out_blocks.2.dense_rbf_F",
                    "out_blocks.1.seq_forces",
                    "out_blocks.1.dense_rbf_F",
                ],
                "validation_split": [0],
                "num_threads": 4,
            },
            "optim": {
                "batch_size": 1,
                "num_workers": 0,
                "max_epochs": 60,
                "lr_initial": 0.0003,
                "factor": 0.9,
            },
        }
        al_config["ocp"] = {
            "checkpoint_path": "/home/jovyan/shared-scratch/ocp_checkpoints/for_finetuna/public_checkpoints/scaling_attached/gemnet_oc_base_oc20_oc22_attscale.pt",
        }
        return al_config
