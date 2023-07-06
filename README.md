# GENIUS-RS: Generalized Explainer of Global Issues in GNN-Based Recommender Systems

GENIUS-RS is a generalized method that leverages counterfactual techniques to explain \
global issues in recommender systems based on Graph Neural Networks. \
Our approach explains a given global issue by detecting a set of user-item interactions, \
represented as edges in a bipartite graph, that alleviates the issue by manipulating the graph \
with edge deletion, addition and re-wiring strategies. Our method can easily accommodate \
any objective aimed at explaining the related global issue. In our study we applied our framework on
GCMC, LightGCN and NGCF, all provided in the [Recbole](https://github.com/RUCAIBox/RecBole)
library, from which GENIUS-RS depend on for the data handling, the training and evaluation.
Instead, the provided models are independent of the Recbole library.

# Requirements
Our framework was tested on Python 3.9 with the libraries listed in the
[requirements.txt](genius/requirements.txt) that can be installed with:
```bash
pip install -r genius/requirements.txt
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

The same version contains a bug related to the NGCF model. A Dropout layer is instantiated inside
the `forward` method, which makes the generation of new embeddings (after the perturbation) not reproducible
even if `eval` is called on the model. To run our experiments the file _recbole/model/general_recommender/ngcf.py_ should
be replaced by the [modified_recbole_ngcf.py](modified_recbole_ngcf.py) file. In Linux:
```bash
cp modified_recbole_ngcf.py /usr/local/lib/python3.9/dist-packages/recbole/model/general_recommender/ngcf.py
```

# Datasets

The datasets used in our datasets are Insurance, Last.FM 1K, MovieLens 1M, TaFeng and
can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.8118988).
They should be placed in a folder named _dataset_ in the project root folder,
so next to the _config_ and _genius_ folders, e.g.:
```
|-- config/*
|-- dataset |-- ml-1m
            |-- lastfm-1k
|-- genius/*
```

# Usage

The file [main.py](genius/main.py) is the entry point for every step to execute in our pipeline.

## 1. Configuration

GENIUS-RS scripts are based on similar Recbole config files that can be found in the
[config](config) folder. The structure is hierarchical, hence the file _base_explainer.yaml_
can be used to set the parameters shared for all the experiments and for each dataset specify
the necessary parameters.
For each dataset there is a config file for:
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
deletions follow a monotonic trend.
- edge_additions: True => edges are added, not removed
- exp_rec_data: "test" => the ground truth lables of the test set are used to measure the approximated NDCG
- __only_adv_group__: "local" => the global issue is measured w.r.t to each batch
- __perturb_adv_group__: the group to be perturbed. False to perturb the disadvantaged group, used when adding nodes.
  True to perturb the advantaged group, used when removing nodes.

## 2. Train Recommender System

The recommender systems need first to be trained:
```bash
python -m genius.main --run train --model MODEL --dataset DATASET --config_file_list config/TRAINING_CONFIG.yaml
```
where __MODEL__ should be one of [GCMC, LightGCN, NGCF], __DATASET__ should match the folder
of dataset, e.g. insurance, ml-1m, __TRAINING_CONFIG__ should be a config file of the
__training__ type.

## 3. Train GENIUS-RS explainer
```bash
python -m genius.main --run explain --model MODEL --dataset DATASET --config_file_list config/TRAINING_CONFIG.yaml --explainer_config_file config/EXPLAINING_CONFIG.yaml --model_file saved/MODEL_FILE
```
where __MODEL__, __DATASET__, __TRAINING_CONFIG__ were already explained above.
__EXPLAINING_CONFIG__ should be the config file relative to the same dataset.

# GENIUS-RS Output

GENIUS-RS creates a folder
_genius/experiments/dp_explanations/DATASET/MODEL/dpbg/LOSS_TYPE/SENSITIVE_ATTRIBUTE/epochs_EPOCHS/CONF_ID_
where __SENSITIVE_ATTRIBUTE__ can be one of [gender, age], __EPOCHS__ is the number of
epochs used to train GENIUS-RS, __CONF_ID__ is the configuration/run ID of the just run
experiment. The folder contains the __EXPLAINING_CONFIG__ file in yaml and pkl format used
for the experiment, a file _cf_data.pkl_ containing the information about the perturbed edges for each epoch,
a file _model_rec_test_preds.pkl_ containing the original recommendations on the rec (perturbation) set and
test set, a file _users_order_.pkl containing the users ids in the order _model_rec_test_preds.pkl_ are sorted,
a file _checkpoint.pth_ containing data used to resume the training if stopped earlier.

_cf_data.pkl_ file contains a list of lists where each inner list has 5 values, relative
to the perturbed edges at a certain epoch:
1) GENIUS-RS total loss
2) GENIUS-RS distance loss
3) GENIUS-RS fair loss
4) global issue of user unfairness of the case study measured with the __fair_metric__ (absolute difference of NDCG)
5) the manipulated edges in a 2xN array, where the first row contains the user ids,
the second the item ids, such that each one of the N columns is a manipulated edge
6) epoch relative to the generated explanations

# Plotting

The script [plot_full_comparison.py](genius/plot_full_comparison.py) can be used to plot the
results used in the paper, for two explaining runs, e.g. one with GENIUS-RS$^-$ (1) and one
with GENIUS-RS$^+$ (2) we could run:
```bash
python -m genius.plot_full_comparison --model_files saved/MODEL_FILE saved/MODEL_FILE --explainer_config_files RUN_1_PATH/config.yaml RUN_2_PATH/config.yaml --utility_metrics [NDCG] --add_plot_table
```

where RUN_1_PATH and RUN_2_PATH are the paths containing the GENIUS-RS output explanations.
