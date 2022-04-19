## How to use
This module is for running `FINETUNA` with just VASP input files. All you need is a folder with VASP `INCAR`, `POTCAR`, `POSCAR`, `KPOINTS`.

- Install via `pip`

    ```sh
        pip install git+https://github.com/ulissigroup/finetuna.git
    ```
    
- Example
    
    - To use `FINETUNA` with default learner and finetuner settings, 
    
    ```sh
        cd /path/to/vasp/input/folder
        finetuna.py
    ```
    or
    ```sh
        finetuna.py -p /path/to/vasp/input/folder
    ```
    The default setting is the same as K-steps in the paper and can be found in `sample_config.yml`.
    
    - You can also modify the settings by copying `sample_config.yml` to your own working directory and changing the parameters.
    
      To use your own config, use `-c` to specify the path to your config file.
    
    ```sh
        cd /path/to/vasp/input/folder
        finetuna.py -c /path/to/config/file
    ```
