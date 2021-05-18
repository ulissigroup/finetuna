import yaml
import pprint


def extract_job_parameters(job_id):
    """This function is used to extract the parameter set from the master job_params.yml
    file."""
    with open("job_params.yml", 'r') as stream:
        hyper_param_set = yaml.safe_load(stream)
    print("\nParameter set for job_id: ",job_id)
    print("------------------------------------")
    pprint.pprint(hyper_param_set[job_id-1]["param_set"])
    print("------------------------------------\n")
    return hyper_param_set[job_id-1]["param_set"]

