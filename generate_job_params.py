import yaml
import numpy as np

uncertain_tols = np.linspace(0.5, 2, 6)
hyperparam_allsets = [dict(param_set=dict(uncertain_tol=float(uncertain_tol))) for uncertain_tol in uncertain_tols]

with open('job_params.yml','w') as outfile:
    yaml.dump(hyperparam_allsets, outfile, default_flow_style=False)









