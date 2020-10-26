import random
from al_mlp.calcs import DeltaCalc
from al_mlp.utils import convert_to_singlepoint, compute_with_calc


class OfflineActiveLearner:
    """Offline Active Learner

    Parameters
    ----------

    learner_settings: dict
        Dictionary of learner parameters and settings.
        
    trainer: object
        An isntance of a trainer that has a train and predict method.
        
    training_data: list
        A list of ase.Atoms objects that have attached calculators.
        Used as the first set of training data.

    parent_calc: ase Calculator object
        Calculator used for querying training data.
        
    base_calc: ase Calculator object
        Calculator used to calculate delta data for training.
        
     """
    
    def __init__(self, learner_settings, trainer, training_data, parent_calc, base_calc):
        self.learner_settings = learner_settings
        self.trainer = trainer
        self.training_data = training_data
        self.parent_calc = parent_calc
        self.base_calc = base_calc
        self.calcs = [parent_calc, base_calc]
        self.init_training_data()
        
    def init_training_data(self):
        """
        Prepare the training data by attaching delta values for training.
        """
        
        raw_data = self.training_data
        sp_raw_data = convert_to_singlepoint(raw_data)
        parent_ref_image = sp_raw_data[0]
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
                
            self.trainer.train(self.training_data)
            trainer_calc = self.make_trainer_calc()
            trained_calc = DeltaCalc([trainer_calc, self.base_calc], "add", self.refs)
            
            atomistic_method.run(calc=trained_calc, filename="relax_iter_%d"%self.iterations)
            sample_candidates = atomistic_method.get_trajectory(filename="relax_iter_%d"%self.iterations)
            
            terminate = self.check_terminate()
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
    
    def check_terminate(self):
        """
        Default termination function. Teminates after 10 iterations
        """
        if self.iterations >= 10:
            return True
        return False
        
    def query_func(self, sample_candidates):
        """
        Default query strategy. Randomly queries 1 data point.
        """
        queried_images = random.sample(sample_candidates,1)
        return queried_images
    
    def make_trainer_calc(self):
        """
        Default trainer calc after train. Assumes trainer has a 'get_calc' method.
        """
        return self.trainer.get_calc()