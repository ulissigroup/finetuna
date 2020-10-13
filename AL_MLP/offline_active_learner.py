from amptorch.ase_utils import AMPtorch
from amptorch.trainer import AtomsTrainer
from calcs import DeltaCalc
from utils import convert_to_singlepoint, compute_with_calc

class OfflineActiveLearner:
    """Offline Active Learner

    Parameters
    ----------

    learner_settings: dict
        Dictionary of learner parameters and settings.

    trainer_config: dict
        Dictionary of model settings including the first set training data. 
        The training data must be calculated using the parent_calc.

    parent_calc: ase Calculator object
        Calculator used for querying training data.
        
    base_calc: ase Calculator object.
        Calculator used to calculate delta data for training
     """
    
    def __init__(self, learner_settings, trainer_config, parent_calc, base_calc):
        self.learner_settings = learner_settings
        self.terminator_func = learner_settings["terminator_func"]
        self.query_func = learner_settings["query_func"]
        self.trainer_config = trainer_config
        self.parent_calc = parent_calc
        self.base_calc = base_calc
        self.calcs = [parent_calc, base_calc]
        self.init_training_data()
        
    def init_training_data(self):
        raw_data = self.trainer_config["dataset"]["raw_data"]
        sp_raw_data = convert_to_singlepoint(raw_data)
        parent_ref_image = sp_raw_data[0].copy()
        base_ref_image = compute_with_calc(sp_raw_data[:1],self.base_calc)[0]
        self.refs = [parent_ref_image, base_ref_image]
        self.delta_sub_calc = DeltaCalc(self.calcs, "sub", self.refs)
        self.training_data = compute_with_calc(sp_raw_data, self.delta_sub_calc)
        
    def learn(self, atomistic_method):
        """
        Conduct offline active learning. Returns the trained calculator.
        
        Parameters
        ----------

        atomistic_method: object
            Define relaxation parameters and starting image.
        """
        
        self.iterations = 0
        
        while not terminate:
            if self.iterations > 0:
                self.query_data(sample_candidates)
            self.trainer_config["dataset"]["raw_data"] = self.training_data
                
            trainer = AtomsTrainer(config)
            trainer.train()
            trainer_calc = AMPtorch(trainer)
            trained_calc = DeltaCalc([trainer_calc, self.base_calc], "add", self.refs)
            
            atomistic_method.run(calc=trained_calc, filename="relax")
            sample_candidates = atomistic_method.get_trajectory(filename="relax")
            
            terminate = self.terminator_func(self)
            self.iterations += 1
            
        return trained_calc
            
    def query_data(self, sample_candidates):
        """
        Queries data from a list of images. Calculates the properties and adds them to the training data.
        
        Parameters
        ----------

        sample_candidates: list
            List of ase atoms objects to query from.
        """
        queried_images = self.query_func(sample_candidates)
        for image in queried_images:
            image.calc = None
        self.training_data += compute_with_calc(queried_images, self.delta_sub_calc)
        