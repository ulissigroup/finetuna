import unittest
from al_mlp.tests.cases.base_case_online_CuNP import BaseOnlineCuNP


class online_ft_CuNP(BaseOnlineCuNP, unittest.TestCase):
    @classmethod
    def get_al_config(cls) -> dict:
        al_config = BaseOnlineCuNP.get_al_config()
        al_config["links"]["ml_potential"] = "ft_en"
        al_config["finetuner"] = {
            "tuner": {
                "num_threads": 4,
                "ensemble_method": "mean",
                "validation_split": [-1],
            },
            "optim": {
                "max_epochs": 20,
                "num_workers": 0,
                "eval_every": 1,
            },
        }
        al_config["ocp"] = {
            "checkpoint_path_list": [
                "/home/jovyan/shared-scratch/joe/optim_cleaned_checkpoints/gemnet_t_direct_h512_all.pt",
                "/home/jovyan/shared-scratch/joe/optim_cleaned_checkpoints/spinconv_force_centric_all.pt",
            ],
            "model_path_list": [
                "/home/jovyan/shared-scratch/joe/actions_runners_files/configs/gemnet-dT.yml",
                "/home/jovyan/shared-scratch/joe/actions_runners_files/configs/spinconv_force.yml",
            ],
            "model_class_list": [
                "gemnet",
                "spinconv",
            ],
        }
        return al_config
