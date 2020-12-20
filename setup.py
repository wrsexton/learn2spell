#!/usr/bin/env python3
import pathlib as PL
from setuptools import setup

with open(PL.Path('conf', 'requirements')) as requirements:
    setup(install_requires=requirements.read().splitlines())
