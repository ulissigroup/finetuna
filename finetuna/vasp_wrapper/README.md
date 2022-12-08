# FINETUNA VASP Input Wrapper
This module is for running `FINETUNA` with just VASP input files. All you need is a folder with VASP `INCAR`, `POTCAR`, `POSCAR`, and `KPOINTS` files.

An ASE [VASP Interactive](https://github.com/ulissigroup/vasp-interactive) calculator will be constructed as the parent calculator using the flags in `INCAR` and `KPOINTS` files. VASP Interactive is a more efficient way of running VASP calculations in ASE. Details can be found in the [repo](https://github.com/ulissigroup/vasp-interactive) and the [paper](https://arxiv.org/abs/2205.01223).

- `NSW` should be set as the maximum DFT calls allowed in the relaxation. (Default: 2000)
- `EDIFFG` will be used as the optimization convergence criterion. (Default: 0.03 eV/A)

## How to use
- Install via `pip`

    ```
        cd finetuna
        pip install -e .
    ```
    
- All pre-trained OCP models can be found [here](https://github.com/Open-Catalyst-Project/ocp/blob/main/MODELS.md). We recommend download [GemNet-dT all](https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_08/s2ef/gemnet_t_direct_h512_all.pt), and use it for `FINETUNA`.
    
- Usage

    Required arguments: 

    | Parameter   | Description   |	
    | :---------- | :-------------|
    | -xc         |the exchange correlation function, e.g.: pbe, rpbe, beef-vdw. Check [ASE-VASP](https://wiki.fysik.dtu.dk/ase/ase/calculators/vasp.html#exchange-correlation-functionals) for the full list the keys.
    | -c  	      |the path to the machine learning model checkpoint. (Not required for VASP-Only mode)

    Optional arguments: 

    | Parameter   | Default       | Description   |	
    | :---------- |:-------------:| :-------------|
    | -p  	      |current working directory	|the path to VASP input directory
    | -con        | finetuna.vasp_wrapper.sample_config.yml |the path to the configuration file
    | -vasponly   | False         |add this argument if you want to run pure VASP Interactive relaxation with BFGS optimizer. No FineTuna involved!

For example:

to run a FineTuna relaxation with RPBE functional using the VASP input files in the current directory,

`finetuna_wrap.py -xc rpbe -c /path/to/checkpoint`

to run a VASP Interactive with BEEF-VdW functional using the VASP input files in `/home/vasp_inputs` directory,

`finetuna_wrap.py -xc beef-vdw -vasponly -p /home/vasp_inputs`


