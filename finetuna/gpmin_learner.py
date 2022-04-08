from ase.optimize.gpmin.gpmin import GPMin
import numpy as np
from ase.optimize.gpmin.prior import CalculatorPrior
from finetuna.ml_potentials.finetuner_calc import FinetunerCalc
from finetuna.utils import convert_to_singlepoint
from scipy.optimize import minimize
import wandb


class GPMinLearner(GPMin):
    def __init__(
        self,
        atoms,
        ml_potential,
        finetune=False,
        restart=None,
        logfile="-",
        trajectory=None,
        kernel=None,
        master=None,
        noise=None,
        weight=None,
        scale=None,
        force_consistent=None,
        batch_size=None,
        bounds=None,
        update_prior_strategy="maximum",
        update_hyperparams=False,
    ):
        prior = OCPPrior(
            atoms,
            ml_potential,
        )
        super().__init__(
            atoms,
            restart,
            logfile,
            trajectory,
            prior,
            kernel,
            master,
            noise,
            weight,
            scale,
            force_consistent,
            batch_size,
            bounds,
            update_prior_strategy,
            update_hyperparams,
        )

        self.atoms_list = []
        self.finetune = finetune

    def step(self, f=None):
        if self.finetune and len(self.atoms_list) > len(
            self.prior.atoms.calc.validation_split
        ):
            self.prior.atoms.calc.train(self.atoms_list)
        fc = self.force_consistent
        atoms = self.atoms
        if f is None:
            f = atoms.get_forces()

        r0 = atoms.get_positions().reshape(-1)
        e0 = atoms.get_potential_energy(force_consistent=fc)
        [atoms_copy] = convert_to_singlepoint([atoms])
        self.update(r0, e0, f, atoms_copy)

        r1 = self.relax_model(r0)
        self.atoms.set_positions(r1.reshape(-1, 3))
        e1 = self.atoms.get_potential_energy(force_consistent=fc)
        f1 = self.atoms.get_forces()
        self.log_wandb(e1, f1)
        self.function_calls += 1
        self.force_calls += 1
        count = 0
        while e1 >= e0:
            [atoms_copy] = convert_to_singlepoint([atoms])
            self.update(r1, e1, f1, atoms_copy)
            r1 = self.relax_model(r0)

            self.atoms.set_positions(r1.reshape(-1, 3))
            e1 = self.atoms.get_potential_energy(force_consistent=fc)
            f1 = self.atoms.get_forces()
            self.log_wandb(e1, f1)
            self.function_calls += 1
            self.force_calls += 1
            if self.converged(f1):
                break

            count += 1
            if count == 30:
                raise RuntimeError("A descent model could not be built")
        self.dump()

    def update(self, r, e, f, atoms_copy):
        """Update the PES

        Update the training set, the prior and the hyperparameters.
        Finally, train the model
        """
        # update the atoms list for the prior with the trained calculator
        self.atoms_list.append(atoms_copy)

        # update the training set
        self.x_list.append(r)
        f = f.reshape(-1)
        y = np.append(np.array(e).reshape(-1), -f)
        self.y_list.append(y)

        # update hyperparams
        if (
            self.update_hp
            and self.function_calls % self.nbatch == 0
            and self.function_calls != 0
        ):
            self.fit_to_batch()

        # build the model
        self.train(np.array(self.x_list), np.array(self.y_list))

    def relax_model(self, r0):
        print("Relaxing after function call " + str(self.function_calls))
        self.gtol = 0
        options = {
            "disp": True,
            "gtol": 1e-03,
            # "maxls": 20,
        }
        i = 0
        while i < 10:
            i = i + 1
            self.gtol = options["gtol"]
            print(
                "minimization attempt number " + str(i) + ", for gtol=" + str(self.gtol)
            )
            result = minimize(
                self.acquisition, r0, method="BFGS", jac=True, options=options
            )
            options["gtol"] = options["gtol"] * 3
            if result.success:
                return result.x
        self.dump()
        raise RuntimeError(
            "The minimization of the acquisition function " "has not converged"
        )

    def log_wandb(self, e, f):
        info = {
            "parent_energy": e,
            "parent_fmax": np.sqrt((f**2).sum(axis=1).max()),
            "parent_forces": f,
            "parent_calls": self.function_calls,
        }
        if hasattr(self, "gtol"):
            info["gtol"] = self.gtol
        wandb.log({key: value for key, value in {**info}.items() if value is not None})


class OCPPrior(CalculatorPrior):
    def __init__(self, atoms, calculator):
        super().__init__(atoms, calculator)
        self.parent_initial_energy = atoms.get_potential_energy()
        atoms_copy = atoms.copy()
        atoms_copy.calc = calculator
        self.prior_initial_energy = atoms_copy.get_potential_energy()

    def potential(self, x):
        self.atoms.set_positions(x.reshape(-1, 3))
        V = self.atoms.get_potential_energy(force_consistent=False)
        V -= self.prior_initial_energy
        V += self.parent_initial_energy
        gradV = -self.atoms.get_forces().reshape(-1)
        return np.append(np.array(V).reshape(-1), gradV)
