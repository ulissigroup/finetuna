import yaml
import numpy as np

uncertain_tols = np.linspace(0, 0.7, 10)
hyperparam_allsets = [dict(param_set=dict(num_layers=2,
                                          num_nodes=50,
                                          stat_uncertain_tol=float(uncertain_tol),
                                          dyn_uncertain_tol=0.)) for uncertain_tol in uncertain_tols]

with open('job_params.yml','w') as outfile:
    yaml.dump(hyperparam_allsets, outfile, default_flow_style=False)









