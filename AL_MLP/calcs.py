import numpy as np
import copy
from ase.calculators.calculator import Calculator, Parameters, all_changes
from ase.calculators.calculator import PropertyNotImplementedError
from amptorch.trainer import AtomsTrainer

class TrainerCalc(Calculator):
    """Atomistics Machine-Learning Potential (AMP) ASE calculator
        This class is temporary, we might make it so the user has to 
        provide this.
   Parameters
   ----------
    trainer: amptorch AtomsTrainer object with a valid predict method.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, trainer):
        Calculator.__init__(self)
        
        # might remove this to allow any trainer with a predict method.
        if not isinstance(trainer, AtomsTrainer):
            raise ValueError('All the calculators should be inherited form the ase\'s Calculator class')
            
        self.trainer = trainer
        
    def calculate(self, atoms, properties, system_changes):
        calculated_atoms = self.trainer.predict([atoms])[0]
        self.results["energy"] = calculated_atoms.get_potential_energy(apply_constraint=False)
        self.results["forces"] = calculated_atoms.get_forces(apply_constraint=False)
        
class DeltaCalc(Calculator):

    def __init__(self, calcs, mode, refs, atoms=None):
        """Implementation of sum of calculators.

        calcs: list
            List of two :mod:`ase.calculators` objects. First calculator must be the parent Calculator in "sub" mode, or the delta trained Calculator in "add" mode. Second calculator must be the base Calculator.
        mode: string
            "sub" or "add" is used to specify if the difference or the sum of the results are calculated.
        refs: list of atoms
            Same atoms with respective calculators attached. The first atoms must have the parent Calculator. The second atoms must have the base Calculator.
        atoms: Atoms object
            Optional :class:`~ase.Atoms` object to which the calculator will be attached.
        """

        super().__init__(atoms=atoms)

        if len(calcs) == 0:
            raise ValueError('The value of the calcs must be a list of Calculators')
            
        if len(calcs) != 2:
            raise ValueError('Must be a list of two Calculators')

        for calc in calcs:
            if not isinstance(calc, Calculator):
                raise ValueError('All the calculators should be inherited form the ase\'s Calculator class')

        if len(calcs) != len(refs):
            raise ValueError('The length of the weights must be the same as the number of calculators!')

        self.calcs_copy = calcs
        self.mode = mode
        self.refs = refs

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        """ Calculates the desired property.
        Precalculated properties from the first calculator can be attached to atoms.
        "sub" mode: calculates the delta between the two given calculators.
        "add" mode: calculates the predicted value given the predicted delta calculator and the base calc.
        """
        
        self.calcs = [copy.copy(calc) for calc in self.calcs_copy]
        
        if atoms.calc is not None:
            self.calcs[0].results["energy"] = atoms.get_potential_energy(apply_constraint=False)
            self.calcs[0].results["forces"] = atoms.get_forces(apply_constraint=False)
        else:
            self.calcs[0].calculate(atoms, properties, system_changes)
            
        self.calcs[1].calculate(atoms, properties, system_changes)
        
        if self.mode == "sub":
            delta_energies = []
            if "energy" in properties:
                for i in range(len(self.calcs)):
                    delta_energies.append(self.calcs[i].results["energy"] -
                                          self.refs[i].get_potential_energy(apply_constraint=False))
                self.results["energy"] = delta_energies[0] - delta_energies[1]
                
            for k in properties:
                if k not in self.results:
                    self.results[k] = calc.results[k]
                else:
                    self.results[k] -= calc.results[k]
                    
        if self.mode == "add":
            delta_energies = []
            if "energy" in properties:
                delta_energies.append(self.calcs[0].results["energy"])
                delta_energies.append(self.calcs[1].results["energy"] - 
                                      self.refs[1].get_potential_energy(apply_constraint=False))
                delta_energies.append(self.refs[0].get_potential_energy(apply_constraint=False))
                self.results["energy"] = np.sum(delta_energies)
                
            for k in properties:
                if k not in self.results:
                    self.results[k] = calc.results[k]
                else:
                    self.results[k] += calc.results[k]

    def reset(self):
        """Clear all previous results recursively from all fo the calculators."""
        super().reset()

        for calc in self.calcs:
            calc.reset()

    def __str__(self):
        calculators = ', '.join(calc.__class__.__name__ for calc in self.calcs)
        return '{}({})'.format(self.__class__.__name__, calculators)
