<!-- [![ulissigroup](https://circleci.com/gh/ulissigroup/finetuna.svg?style=svg)](https://app.circleci.com/pipelines/github/ulissigroup/finetuna) -->
[![Lint](https://github.com/ulissigroup/finetuna/actions/workflows/black.yml/badge.svg)](https://github.com/ulissigroup/finetuna/actions/workflows/black.yml)
[![Test](https://github.com/ulissigroup/finetuna/actions/workflows/unittests.yml/badge.svg)](https://github.com/ulissigroup/finetuna/actions/workflows/unittests.yml)
## *FINETUNA*: Fine-Tuning Accelerated Molecular Simulations
<img align="left" src="https://github.com/ulissigroup/finetuna/blob/main/doc/finetuna_logo.png" width="280">
Are you doing structural optimizations with DFT or other electronic structure codes?? Try :monocle_face::fish: FINETUNA for accurate but 90% faster relaxation!

FINETUNA accelerates atomistic simulations by fine-tuning a pre-trained graph model in an active learning framework.

Installation is easy:
```
conda env create -f env.cpu.yml
git clone https://github.com/ulissigroup/finetuna.git
cd finetuna
pip install -e .
pip install git+https://github.com/ulissigroup/vasp-interactive.git
```

All pre-trained machine learning model checkpoint can be found [here](https://github.com/Open-Catalyst-Project/ocp/blob/main/MODELS.md). We recommend to download the GemNet-dT all model. [click here to download](https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_08/s2ef/gemnet_t_direct_h512_all.pt).

You are all set! Now in your VASP input folder, run the calculation by: `finetuna_wrap.py -c /path/to/the/checkpoint`.


<img src="https://github.com/ulissigroup/finetuna/blob/main/doc/workflow.png" width="700">

### Usage

If you have an ASE atoms object, see example [1](https://github.com/ulissigroup/finetuna/blob/main/examples/online_al_example.py) and [2](https://github.com/ulissigroup/finetuna/blob/main/examples/online_al_beef_example.py).

If you have VASP input files (INCAR, KPOINTS, POTCAR, and POSCAR), see example [3](https://github.com/ulissigroup/finetuna/tree/main/finetuna/vasp_wrapper).
