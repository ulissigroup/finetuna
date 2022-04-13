#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="finetuna",
    version="0.1",
    description="Fine-Tuning Accelerated Molecular Simulations",
    url="https://github.com/ulissigroup/finetuna",
    author="Joseph Musielewicz, Xiaoxiao Wang",
    author_email="al.mlp.package@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["ase"],
    scripts=["examples/vasp_wrapper/finetuna_vasp.py"],
    long_description="""Module for performing active learning by fine-tuning pre-trained machine learning
     potentials to accelerate molecular simulations.""",
)
