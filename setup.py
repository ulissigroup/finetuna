#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="al_mlp",
    version="0.1",
    description="Active Learning for Machine Learning Potentials",
    url="https://github.com/ulissigroup/al_mlp",
    author="Joseph Musielewicz, Lory Wang",
    author_email="al.mlp.package@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["ase"],
    long_description="""Module for performing active learning \
                          with machine learning potentials.""",
)
