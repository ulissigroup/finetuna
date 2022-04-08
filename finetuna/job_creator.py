import os
import yaml
import collections.abc
import subprocess
from ase.db import connect
from ase.calculators.vasp.create_input import GenerateVaspInput


def create_job(
    given_params,
    main_path="/home/finetuna/examples/main/al_main.py",
    sample_job_spec_path="sample_job_spec.yml",
    sample_config_path=None,
    images=None,
    local=False,
):
    # generate the params dictionary from the given config if applicable
    params = given_params
    if sample_config_path is not None:
        with open(sample_config_path, "r") as sample_config:
            base_params = yaml.safe_load(sample_config)
        params = merge_dict(base_params, given_params)

    # get the job name
    job_name = (
        params.get("learner", {}).get("wandb_init", {}).get("name", "default_job_name")
    )  # TODO

    basedir = os.getcwd()
    subdir = "/"
    subdir += job_name
    subdir += "_0"

    # change to new directory created inside the current working directory
    if not os.path.isdir(basedir + subdir):
        os.mkdir(basedir + subdir)
    else:
        while os.path.isdir(basedir + subdir):
            i = int(subdir[subdir.rindex("_") + 1 :])
            subdir = subdir[: subdir.rindex("_") + 1] + str(i + 1)
        os.mkdir(basedir + subdir)
    os.chdir(basedir + subdir)

    # if given some images to pretrain on, make an ase_db and save the link in the config
    if images is not None:
        images_path = basedir + subdir + "/" + "pretrain_images.db"
        params["links"]["images_path"] = images_path
        with connect(images_path) as pretrain_db:
            for image in images:
                pretrain_db.write(image)

    # if given incar in links, change the vasp params to match the incar
    if "incar" in params["links"]:
        vasp_input = GenerateVaspInput()
        vasp_input.atoms = None
        vasp_input.read_incar(params["links"]["incar"])
        if "kpoints" in params["links"]:
            vasp_input.read_kpoints(params["links"]["kpoints"])
        params["vasp"] = vasp_input.todict()
        if "kpts" in params["vasp"]:
            kpts = [float(i) for i in params["vasp"]["kpts"]]
            params["vasp"]["kpts"] = kpts
        else:
            params["vasp"].pop("kpts")
        if "gga" not in params["vasp"]:
            params["vasp"]["gga"] = "PE"  # defaults to PE for oxide
        params["vasp"]["nsw"] = 0
        params["vasp"]["ibrion"] = -1
        params["vasp"]["lreal"] = "Auto"

    # create the new config in the new directory
    config_path = basedir + subdir + "/" + job_name + "_config.yml"
    with open(config_path, "w") as config_file:
        yaml.dump(params, config_file, default_flow_style=False)

    # load the sample_job_spec.yml
    with open(sample_job_spec_path, "r") as sample_job_spec:
        job_spec = yaml.safe_load(sample_job_spec)

    # create the new job_spec dictionary from the sample_job_spec.yml
    job_spec["metadata"]["name"] = "job-" + job_name.replace("_", "-")
    job_spec["spec"]["template"]["spec"]["containers"][0]["name"] = job_name.replace(
        "_", "-"
    )

    if "NAMESPACE" in os.environ:
        namespace = os.environ["NAMESPACE"]
        job_spec["metadata"]["namespace"] = namespace
    if "VOLUME" in os.environ:
        volume = os.environ["VOLUME"]
        job_spec["spec"]["template"]["spec"]["containers"][0]["volumeMounts"][0][
            "name"
        ] = volume
        job_spec["spec"]["template"]["spec"]["volumes"][0]["name"] = volume
        job_spec["spec"]["template"]["spec"]["volumes"][0]["persistentVolumeClaim"][
            "claimName"
        ] = volume

    args_string = job_spec["spec"]["template"]["spec"]["containers"][0]["args"][0]
    args_string = (
        args_string[: args_string.rindex("python")]
        + "python "
        + main_path
        + " --config-yml "
        + config_path
        + " 2>&1 | tee "
        + basedir
        + subdir
        + "/run_logs.txt"
    )
    job_spec["spec"]["template"]["spec"]["containers"][0]["args"][0] = args_string

    # create the new job_spec.yml from the job_spec dictionary
    job_spec_path = basedir + subdir + "/" + job_name + "_spec.yml"
    with open(job_spec_path, "w") as job_spec_file:
        yaml.dump(job_spec, job_spec_file, default_flow_style=False)

    if local is False:
        # call kubectl on the job_spec.yml
        run_result = subprocess.run(["kubectl", "apply", "-f", job_spec_path])
        print(
            "Executed job " + job_name + " with exit code " + str(run_result.returncode)
        )

    # change back to original working directory
    os.chdir(basedir)

    return config_path


def merge_dict(d, u):
    for key, value in u.items():
        if isinstance(value, collections.abc.Mapping):
            d[key] = merge_dict(d.get(key, {}), value)
        else:
            d[key] = value
    return d
