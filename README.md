# GNNUERS: Explaining Unfairness in GNNs for Recommendation

GNNUERS generates explanations in the form of user-item interactions that make
a GNN-based recommender system favor a demographic group over another. \
GNNUERS learns a perturbation vector that modifies the adjacency matrix representing
the training network. The edges modified by the perturbation vector are the explanations
genereated by the framework. \
GNNUERS then needs to work on a slight extended version of a recommender system
in order to include the perturbation vector. In our study we applied our framework on
GCMC, LightGCN and NGCF, all provided in the [Recbole](https://github.com/RUCAIBox/RecBole)
library, from which GNNUERS depend on for the data handling, the training and evaluation.
Instead, the provided models are independent of the Recbole library.

# Requirements
Our framework was tested on Python 3.9 with the libraries listed in the
[requirements.txt](src/requirements.txt) that can be installed with:
```bash
pip install -r src/requirements.txt
```
Some dependencies related to PyTorch, e.g. torch-scatter, could be hard to retrieve
directly from pip depending on the PyTorch and CUDA version you are using, so you should
specify the PyTorch FTP link storing the right libraries versions.
For instance to install the right version of torch-scatter for PyTorch 1.12.0
you should use the following command:
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+${CUDA}.html
```
where `${CUDA}` should be replaced by either `cpu`, `cu***`, where `***` represents the
CUDA version, e.g. 116, 117.

__NOTE!__ \
The Recbole version (1.0.1) does not support the usage of custom dataset splits like ours,
and we cannot guarantee that, even if provided in new versions, it will match our
modification. To run our experiments the file _recbole/data/dataset/dataset.py_ should
be replaced by the [modified_recbole_dataset.py](modified_recbole_dataset.py) file. In Linux:
```bash
cp modified_recbole_dataset.py /usr/local/lib/python3.9/dist-packages/recbole/data/dataset/dataset.py
```

# Datasets

The datasets used in our datasets are MovieLens 1M, Last.FM 1K, Ta Feng, Insurance and
can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.7602406).
They should be placed in a folder named _dataset_ in the project root folder,
so next to the _config_ and _src_ folders, e.g.:
```
|-- config/*
|-- dataset |-- ml-1m
            |-- lastfm-1k
|-- src/*
```

# Usage

The file [main.py](src/main.py) is the entry point for every step to execute in our pipeline.

## 1. Configuration

GNNUERS scripts are based on similar Recbole config files that can be found in the
[config](config) folder. For each dataset there is a config file for:
- __training__: it is named after the dataset, e.g. _ml-1m.yaml_ for MovieLens-1M,
_tafeng.yaml_ for Ta Feng
- __explaining__: the suffix __explainer_ is added to training config filename, e.g.
_ml-1m_explainer.yaml_ for MovieLens 1M, _tafeng_explainer.yaml_ for Ta Feng

The __training__ config type parameters description can be found in the Recbole repository
and website, except for this part:
```yaml
eval_args:
    split: {'LRS': None}
    order: RO  # not relevant
    group_by: '-'
    mode: 'full'
```
where `LRS` (Load Ready Splits) is not a Recbole split type, but it is added in
our _modified_recbole_dataset.py_ to support custom data splits.

The description of each parameter in the __explaining__ config type can be found in the
relative files. In particular, for the explainer_policies:
- __force_removed_edges__: it should be always True to reproduce our results, it represents
the policy that prevents the restore of a previously deleted edge, such that the edges
deletions follow a monotonic trend
- __group_deletion_constraint__: it is the Connected Nodes (CN) policy
- __random_perturbation__: if True executes the baseline algorithm RND-P

## 2. Train Recommender System

The recommender systems need first to be trained:
```bash
python -m src.main --run train --model MODEL --dataset DATASET --config_file_list config/TRAINING_CONFIG.yaml
```
where __MODEL__ should be one of [GCMC, LightGCN, NGCF], __DATASET__ should match the folder
of dataset, e.g. insurance, ml-1m, __TRAINING_CONFIG__ should be a config file of the
__training__ type.

## 3. Train GNNUERS explainer
```bash
python -m src.main --run explain --model MODEL --dataset DATASET --config_file_list config/TRAINING_CONFIG.yaml --explainer_config_file config/EXPLAINING_CONFIG.yaml --model_file saved/MODEL_FILE
```
where __MODEL__, __DATASET__, __TRAINING_CONFIG__ were already explained above.
__EXPLAINING_CONFIG__ should be the config file relative to the same dataset.

# GNNUERS Output

GNNUERS creates a folder
_src/dp_ndcg_explanations/DATASET/MODEL/FairDP/SENSITIVE_ATTRIBUTE/epochs_EPOCHS/CONF_ID_
where __SENSITIVE_ATTRIBUTE__ can be one of [gender, age], __EPOCHS__ is the number of
epochs used to train GNNUERS, __CONF_ID__ is the configuration/run ID of the just run
experiment. The folder contains the __EXPLAINING_CONFIG__ file in yaml and pkl format used
for the experiment and a file _all_users.pkl_.

_all_users.pkl_ file contains a list of lists where each inner list has 13 values, relative
to the explanations generated at a certain epoch:
1) the user IDS 
2) the __rec__ topk recommendation lists of the non-perturbed model, where __rec__
identifies the set on which these lists are generated, e.g. validation, test
3) the __test__ topk recommendation lists of the non-perturbed model
4) the __rec__ topk recommendation lists of the perturbed model 
5) the __test__ topk recommendation lists of the perturbed model
6) the distance between __rec__ topk lists of the non-perturbed and perturbed model,
with the distance measured as damerau levenshtain distance as default
7) the distance between __test__ topk lists of the non-perturbed and perturbed model
8) GNNUERS total loss
9) GNNUERS distance loss
10) GNNUERS fair loss
11) the deleted edges in a 2xN array, where the first row contains the user ids,
the second the item ids, such that each one of the N columns is a deleted edge
12) epoch relative to the generated explanations
13) GNNUERS fair loss measured on the topk lists of the non-perturbed graph

# Plotting

The script [plot_full_comparison.py](src/plot_full_comparison.py) can be used to plot the
results used in the paper, for two explaining runs, e.g. one with GNNUERS base (1) and one
with GNNUERS+CP (2) we could run:
```bash
python -m src.plot_full_comparison --model_files saved/MODEL_FILE saved/MODEL_FILE --explainer_config_files RUN_1_PATH/config.yaml RUN_2_PATH/config.yaml --utility_metrics [NDCG] --add_plot_table
```

where RUN_1_PATH and RUN_2_PATH are the paths containing the GNNUERS output explanations.
