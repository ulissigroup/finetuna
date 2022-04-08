from ase.calculators.calculator import Calculator, all_changes
from ase.atoms import Atoms


class MLPCalc(Calculator):
    """
    Machine Learning Potential Calculator.
    This class serves as a parent class for calculators in finetuna
    Guarantees all calculators will implement train()
    Provides calculate method with boilerplate lines to be called by children calculate methods
    Also guarantees certain universal values will be created during initialization (self.mlp_params)

    Parameters
    ----------
    mlp_params: dict
        dictionary of parameters to be passed to the ml potential model in init_model()
    """

    implemented_properties = ["energy", "forces", "stds"]

    def __init__(
        self,
        mlp_params: dict = {},
    ):
        super().__init__()

        self.mlp_params = mlp_params

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Calculate properties including: energy, forces, uncertainties.

        Args:
            atoms: ase Atoms object
        """
        super().calculate(
            atoms=atoms, properties=properties, system_changes=system_changes
        )

        if properties is None:
            properties = self.implemented_properties

    def train(self, parent_dataset: "list[Atoms]", new_dataset: "list[Atoms]" = None):
        """
        Train the ml model by fitting a new model on the parent dataset,
        or partial fit the current model on just the new_dataset

        Args:
            parent_dataset: list of all the descriptors to be trained on

            new_dataset: list of just the new descriptors to partially fit on
        """
        raise NotImplementedError
