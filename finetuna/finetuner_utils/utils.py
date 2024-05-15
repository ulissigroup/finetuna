from torch.utils.data import Dataset


# Create dummy classes with expected functions for loading finetuning trainer and models
class GraphsListDataset(Dataset):
    def __init__(self, graphs_list):
        self.graphs_list = graphs_list

    def __len__(self):
        return len(self.graphs_list)

    def __getitem__(self, idx):
        graph = self.graphs_list[idx]
        return graph


class GenericDB:
    def __init__(self):
        pass

    def close_db(self):
        pass


# Add gemnet_t_uncertainty as GemNetT class for loading homoscedastic model checkpoints
from ocpmodels.common.registry import registry
from ocpmodels.models.gemnet.gemnet import GemNetT


# imported in __init__.py
@registry.register_model("gemnet_t_uncertainty")
class GemNetTUncertainty(GemNetT):
    def __init__(self, *args, **kwargs):
        kwargs.pop("heteroskedastic")
        super().__init__(*args, **kwargs)
