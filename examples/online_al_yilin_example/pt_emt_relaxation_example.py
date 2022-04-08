import ase
from finetuna.atomistic_methods import Relaxation
from ase.optimize import BFGS
from ase.calculators.emt import EMT

initial_db = ase.io.read("Pt-init-images.db", ":")
slab = initial_db[1]
true_relax = Relaxation(slab, BFGS)
true_relax.run(EMT(), "true_relax")
