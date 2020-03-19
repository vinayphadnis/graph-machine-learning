# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

try:
    long_description = open("README.rst").read()
except IOError:
    long_description = ""

setup(
    name="graphml",
    version="0.0.1",
    description="A python package for working on a graph based machine learning algorithm used for three dimensional data optimisation and prediction",
    license="MIT",
    author="Vinay Phadnis",
    packages=find_packages(),
    install_requires=[],
    long_description=long_description,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
    ]
)
