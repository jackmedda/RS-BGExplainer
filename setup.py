import os
from setuptools import setup, find_packages

try:
    import torch
except ModuleNotFoundError:
    raise ModuleNotFoundError("torch must be installed first")

try:
    import torch_geometric
except ModuleNotFoundError:
    raise ModuleNotFoundError("torch-geometric must be installed first")

try:
    import torch_scatter
except ModuleNotFoundError:
    raise ModuleNotFoundError("torch-scatter must be installed first")

try:
    import torch_sparse
except ModuleNotFoundError:
    raise ModuleNotFoundError("torch-sparse must be installed first")

install_requires = [
    "recbole>=1.2.0",
    "numba",
    "wandb",
    "igraph",
    "optuna",
    "seaborn>=0.11.2"
]

setup_requires = []

extras_require = {}

classifiers = [
    "Development Status :: 3 - Alpha",
    "Environment :: GPU :: NVIDIA CUDA",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
]

long_description = """
GNNUERS generates explanations in the form of user-item interactions that make
a GNN-based recommender system favor a demographic group over another. GNNUERS learns a perturbation
vector that modifies the adjacency matrix representing the training network. The edges modified by
the perturbation vector are the explanations genereated by the framework. GNNUERS then needs to work on a slight
extended version of a recommender system in order to include the perturbation vector and optimize it.
"""

setup(
    name="gnnuers",
    version="0.0.4.3",  # please remember to edit gnnuers/__init__.py in response, once updating the version
    description="A library to generate explanations of recommendations generated by GNNs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jackmedda/RS-BGExplainer",
    author="jackmedda",
    author_email="jackm.medda@gmail.com",
    packages=[package for package in find_packages() if package.startswith("gnnuers")],
    include_package_data=True,
    install_requires=install_requires,
    setup_requires=setup_requires,
    extras_require=extras_require,
    zip_safe=False,
    classifiers=classifiers,
)
