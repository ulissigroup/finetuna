# FINETUNA VASP Input Wrapper
This module is for running `FINETUNA` with just VASP input files. All you need is a folder with VASP `INCAR`, `POTCAR`, `POSCAR`, and `KPOINTS` files.

An ASE [VASP Interactive](https://github.com/ulissigroup/vasp-interactive) calculator will be constructed as the parent calculator using the flags in `INCAR` and `KPOINTS` files. VASP Interactive is a more efficient way of running VASP calculations in ASE. Details can be found in the [repo](https://github.com/ulissigroup/vasp-interactive) and the [paper](https://arxiv.org/abs/2205.01223).

- `NSW` should be set as the maximum DFT calls allowed in the relaxation. (Default: 2000)
- `EDIFFG` will be used as the optimization convergence criterion. (Default: 0.03 eV/A)

## How to use
- Install via `pip`

    ```sh
        pip install git+https://github.com/ulissigroup/finetuna.git
    ```
    
- All pre-trained OCP models can be found [here](https://github.com/Open-Catalyst-Project/ocp/blob/main/MODELS.md). We recommend download [GemNet-dT all](https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_08/s2ef/gemnet_t_direct_h512_all.pt), and use it for `FINETUNA`.
    
- Usage
    - Run `FINETUNA` by
    ```sh
        cd /path/to/vasp/input/folder
        finetuna_wrap.py -c /path/to/checkpoint.pt
    ```
        or
    ```sh
        finetuna_wrap.py -c /path/to/checkpoint.pt -p /path/to/vasp/input/folder
    ```
 
- A subfolder called finetuna_relaxation will be created in the working directory. An [ASE db](https://wiki.fysik.dtu.dk/ase/tutorials/tut06_database/database.html) file (`oal_queried_images.db`) will be generated that stores the structure, energies and forces information.
