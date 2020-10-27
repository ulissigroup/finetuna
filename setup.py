#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='al_mlp',
      version='0.1',
      description='Active Learning for Machine Learning Potentials',
      url='https://github.com/ulissigroup/al_mlp',
      author='Rui Qi Chen, Matt Adams',
      author_email='ruiqic@andrew.cmu.edu, madams2@andrew.cmu.edu',
      packages=find_packages(),
      include_package_data=False,
      install_requires=['ase'],
      long_description='''Module for performing delta active learning \
                          with machine learning potentials.''',)
