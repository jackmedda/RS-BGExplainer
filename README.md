# CPFairRobust: Robustness in Fairness against Edge-level Perturbations in GNN-based Recommendation

CPFairRobust performs a poisoning-like attack that perturbs the user-item interactions that make
a GNN-based recommender system favor a demographic group over another, disrupting the system fairness. \
CPFairRobust learns a perturbation vector that modifies the adjacency matrix representing
the training network. \
CPFairRobust then needs to work on a slightly extended version of a recommender system
in order to learn the perturbation vector. In our study we applied our framework on
GCMC, LightGCN and NGCF, all provided in the [Recbole](https://github.com/RUCAIBox/RecBole)
library, from which CPFairRobust depend on for the data handling, the training and evaluation.
Instead, the provided models are independent of the Recbole library.

# Requirements
Our framework was tested on Python 3.9 with the libraries listed in the
[requirements.txt](cpfair_robust/requirements.txt) that can be installed with:
```bash
pip install -r cpfair_robust/requirements.txt
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

The datasets used in our experiments are MovieLens 1M, Last.FM 1K, Insurance and
can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.7602406).
They should be placed in a folder named _dataset_ in the project root folder,
so next to the _config_ and _cpfair_robust_ folders, e.g.:
```
|-- config/*
|-- dataset |-- ml-1m
            |-- lastfm-1k
|-- cpfair_robust/*
```

# Usage

The file [main.py](cpfair_robust/main.py) is the entry point for every step to execute in our pipeline.

## 1. Configuration

CPFairRobust scripts are based on similar Recbole config files that can be found in the
-[config](config) folder. The structure is hierarchical, hence the file _base_explainer.yaml_
-can be used to set the parameters shared for all the experiments and for each dataset specify
-the necessary parameters.
-For each dataset there is a config file for:
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
- exp_rec_data: "test" => the ground truth lables of the test set are used to measure the utility metrics
- __only_adv_group__: "local" => the fairness level is measured w.r.t to each batch
- __gradient_deactivation_constraint__: always False
- __perturb_adv_group__: it does not have effect if __gradient_deactivation_constraint__ is False
For each fairness type the parameters to specify are:
- CP
  - __exp_metric__: consumer_DP
  - __metric_loss__: ndcg  # approximated loss to evaluate the fairness level
  - __eval_metric__: ndcg  # metric to evaluate the fairness level
  - __sensitive_attribute__: age  # or "gender"
- CS
  - __exp_metric__: consumer_DP
  - __metric_loss__: softmax  # approximated loss to evaluate the fairness level => the name is inaccurate, but the loss is the one explained in the paper
  - __eval_metric__: precision  # metric to evaluate the fairness level
  - __sensitive_attribute__: age  # or "gender"
- PE
  - __exp_metric__: provider_DP
  - __item_discriminative_attribute__: exposure
- PV
  - __exp_metric__: provider_DP
  - __item_discriminative_attribute__: visibility
The edges addition or deletion is decided by the following parameter:
- edge_additions: True  # to add edges, False to delete them

## 2. Train Recommender System

The recommender systems need first to be trained:
```bash
python -m CPFairRobust.main --run train --model MODEL --dataset DATASET --config_file_list config/TRAINING_CONFIG.yaml
```
where __MODEL__ should be one of [GCMC, LightGCN, NGCF], __DATASET__ should match the folder
of dataset, e.g. insurance, ml-1m, __TRAINING_CONFIG__ should be a config file of the
__training__ type.

## 3. Train CPFairRobust explainer
```bash
python -m cpfair_robust.main --run explain --model MODEL --dataset DATASET --config_file_list config/TRAINING_CONFIG.yaml --explainer_config_file config/EXPLAINING_CONFIG.yaml --model_file saved/MODEL_FILE
```
where __MODEL__, __DATASET__, __TRAINING_CONFIG__ were already explained above.
__EXPLAINING_CONFIG__ should be the config file relative to the same dataset.

# CPFairRobust Output

CPFairRobust creates a folder
_cpfair_robust/experiments/dp_explanations/DATASET/MODEL/dpbg/LOSS_TYPE/SENSITIVE_ATTRIBUTE/epochs_EPOCHS/CONF_ID_
where __SENSITIVE_ATTRIBUTE__ can be one of [gender, age] or not included if __exp_metric__ is provider_DP, __EPOCHS__ is the number of
epochs used to train the attack CPFairRobust, __CONF_ID__ is the configuration/run ID of the just run
experiment. The folder contains the __EXPLAINING_CONFIG__ file in yaml and pkl format used
for the experiment, a file _cf_data.pkl_ containing the information about the perturbed edges for each epoch,
a file _model_rec_test_preds.pkl_ containing the original recommendations on the rec (perturbation) set and
test set, a file _users_order_.pkl containing the users ids in the order _model_rec_test_preds.pkl_ are sorted,
a file _checkpoint.pth_ containing data used to resume the training if stopped earlier.

_cf_data.pkl_ file contains a list of lists where each inner list has 5 values, relative to the perturbed edges at a certain epoch:
1) CPFairRobust total loss
2) CPFairRobust distance loss
3) CPFairRobust fair loss
4) fairness measured with the __fair_metric__ (absolute difference of NDCG)
5) the perturbed edges in a 2xN array, where the first row contains the user ids,
the second the item ids, such that each one of the N columns is a perturbed edge
6) epoch relative to the generated explanations

# Plotting

The scripts inside the folder [scripts](scripts) can be used to plot the
results used in the paper. They should be run from the root folder of this project.
[cpfair_robust_eval.py](scripts/cpfair_robust_eval.py) can be used as follows:
```bash
python scripts/cpfair_robust_eval.py --e cpfair_robust/experiments/dp_explanations/DATASET/MODEL/dpbg/LOSS_TYPE/SENSITIVE_ATTRIBUTE/epochs_EPOCHS/CONF_ID
```
where the argument --e stands for the path of a specific experiment.
The other files starting with __cpfair__ are used to generate specific plots of the paper.
