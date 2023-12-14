# BiGA: Bipartite Graph Augmenter for unfairness mitigation in recommendation

![44804a56-fbf6-4796-b8ad-2727b05c7c13](https://github.com/jackmedda/RS-BGExplainer/assets/26059819/b3d0c73e-f676-4b91-a869-34998675699a)

This repository contains the source code of the paper [Counterfactual Graph Augmentation for Consumer Unfairness Mitigation in Recommender Systems](https://dl.acm.org/doi/10.1145/3583780.3615165).

If you find this repostiory useful for your research or development cite our paper as

```bibtex
@inproceedings{10.1145/3583780.3615165,
author = {Boratto, Ludovico and Fabbri, Francesco and Fenu, Gianni and Marras, Mirko and Medda, Giacomo},
title = {Counterfactual Graph Augmentation for Consumer Unfairness Mitigation in Recommender Systems},
year = {2023},
isbn = {9798400701245},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3583780.3615165},
doi = {10.1145/3583780.3615165},
abstract = {In recommendation literature, explainability and fairness are becoming two prominent perspectives to consider. However, prior works have mostly addressed them separately, for instance by explaining to consumers why a certain item was recommended or mitigating disparate impacts in recommendation utility. None of them has leveraged explainability techniques to inform unfairness mitigation. In this paper, we propose an approach that relies on counterfactual explanations to augment the set of user-item interactions, such that using them while inferring recommendations leads to fairer outcomes. Modeling user-item interactions as a bipartite graph, our approach augments the latter by identifying new user-item edges that not only can explain the original unfairness by design, but can also mitigate it. Experiments on two public data sets show that our approach effectively leads to a better trade-off between fairness and recommendation utility compared with state-of-the-art mitigation procedures. We further analyze the characteristics of added edges to highlight key unfairness patterns. Source code available at https://github.com/jackmedda/RS-BGExplainer/tree/cikm2023.},
booktitle = {Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
pages = {3753–3757},
numpages = {5},
keywords = {recommender systems, mitigation, explainability, gnn, fairness},
location = {Birmingham, United Kingdom},
series = {CIKM '23}
}
```

BiGA uses counterfactual explanations to augment user-item interactions within a bi-partite graph. \
Studying the disparity in the recommendations utility across demographic groups, our method detects
which graph edges should be added to the original data set to mitigate user unfairness on the evaluation data. \
Thus, the added edges work as an explanation of the prior unfairness and of our mitigation procedure underlying process. \
BiGA then needs to work on a slight extended version of a recommender system
in order to include the perturbation vector. In our study we applied our framework on
GCMC, LightGCN and NGCF, all provided in the [Recbole](https://github.com/RUCAIBox/RecBole)
library, from which BiGA depend on for the data handling, the training and evaluation.
Instead, the provided models are independent of the Recbole library.

# Requirements
Our framework was tested on Python 3.9 with the libraries listed in the
[requirements.txt](biga/requirements.txt) that can be installed with:
```bash
pip install -r biga/requirements.txt
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

# Datasets [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8030711.svg)](https://doi.org/10.5281/zenodo.8030711)

The datasets used in our datasets are MovieLens 1M, Last.FM 1K and
can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.8030711).
They should be placed in a folder named _dataset_ in the project root folder,
so next to the _config_ and _biga_ folders, e.g.:
```
|-- config/*
|-- dataset |-- ml-1m
            |-- lastfm-1k
|-- biga/*
```

# Usage

The file [main.py](biga/main.py) is the entry point for every step to execute in our pipeline.

## 1. Configuration

BiGA scripts are based on similar Recbole config files that can be found in the
[config](config) folder. For each dataset there is a config file for:
- __training__: it is named after the dataset, e.g. _ml-1m.yaml_ for MovieLens-1M
- __explaining__: the suffix __explainer_ is added to training config filename, e.g.
_ml-1m_explainer.yaml_ for MovieLens 1M

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
relative files. In particular, the following ones should not be modified:
- edge_additions: True => edges are added, not removed
- exp_rec_data: "valid" => the ground truth lables of the validation set are used to measure the approximated NDCG
- __only_adv_group__: "global" => the perturbation process is optimized to perturb edges such that
  the utility of the disadvantaged (advantaged) group gets closer to the advantaged (disadvantaged) group one.
  In global mode the utility of the second group stays fixed and the algorithm only studies
  the recommendations of the disadvantaged (advantaged) group
- __perturb_adv_group__: False => the group to be perturbed is the disadvantaged one. Using
  __only_adv_group__ = "global" the algorithm is optimized to boost the utility of the disadvantaged group
  which is perturbed.

for the explainer_policies:
- __group_deletion_constraint__: leave it to True to reproduce our experiments. In reality,
  it does not have effect in our experiments, since the algorithm only deals with one of the groups
  when __only_adv_group__ = "global", hence the other group users have already been removed
- __users_zero_constraint__: only perturbs users with `eval_metric` <= `users_zero_constraint_value` (ZN in the paper)
- __users_low_degree__: only perturbs users with an interaction history shorter than `users_low_degree_value` (LD in the paper)
- __users_furthest_constraint__: only perturbs edges connected to the furthest users (perturbed group) from the non perturbed group (F in the paper)
- __sparse_users_constraint__: only perturbs edges connected to users connected with niche items (S in the paper)
- __items_preference_constraint__: only perturbs edges connected to items preferred by the perturbed group (IP in the paper)
- __niche_items_constraint__: only perturbs edges connected to niche items (not used because too similar with S, even though it alters the item set)

## 2. Train Recommender System

The recommender systems need first to be trained:
```bash
python -m biga.main --run train --model MODEL --dataset DATASET --config_file_list config/TRAINING_CONFIG.yaml
```
where __MODEL__ should be one of [GCMC, LightGCN, NGCF], __DATASET__ should match the folder
of dataset, e.g. insurance, ml-1m, __TRAINING_CONFIG__ should be a config file of the
__training__ type.

## 3. Train BiGA mitigation algorithm
```bash
python -m biga.main --run explain --model MODEL --dataset DATASET --config_file_list config/TRAINING_CONFIG.yaml --explainer_config_file config/EXPLAINING_CONFIG.yaml --model_file saved/MODEL_FILE
```
where __MODEL__, __DATASET__, __TRAINING_CONFIG__ were already explained above.
__EXPLAINING_CONFIG__ should be the config file relative to the same dataset.
__MODEL_FILE__ is the path to the .pth model file saved by Recbole in the second step.

# BiGA Output

BiGA creates a folder
_biga/experiments/dp_explanations/DATASET/MODEL/dpbg/LOSS_TYPE/SENSITIVE_ATTRIBUTE/epochs_EPOCHS/CONF_ID_
where __SENSITIVE_ATTRIBUTE__ can be one of [gender, age], __EPOCHS__ is the number of
epochs used to train BiGA, __CONF_ID__ is the configuration/run ID of the just run
experiment. The folder contains the __EXPLAINING_CONFIG__ file in yaml and pkl format used
for the experiment, a file _cf_data.pkl_ containing the information about the perturbed edges for each epoch,
a file _model_rec_test_preds.pkl_ containing the original recommendations on the rec (perturbation) set and
test set, a file _users_order_.pkl containing the users ids in the order _model_rec_test_preds.pkl_ are sorted,
a file _checkpoint.pth_ containing data used to resume the training if stopped earlier.

_cf_data.pkl_ file contains a list of lists where each inner list has 5 values, relative
to the perturbed edges at a certain epoch:
1) BiGA total loss
2) BiGA distance loss
3) BiGA fair loss
4) fairness measured with the __fair_metric__ (function operationalizing demographic parity, i.e. DP)
5) the perturbed edges in a 2xN array, where the first row contains the user ids,
the second the item ids, such that each one of the N columns is a perturbed edge
6) epoch relative to the generated explanations

# Plotting

The scripts inside the folder [scripts](scripts) can be used to plot the
results used in the paper. They should be run from the root folder of this project.
[eval_info.py](scripts/eval_info.py) can be used as follows:
```bash
python scripts/eval_info.py --e biga/experiments/dp_explanations/DATASET/MODEL/dpbg/LOSS_TYPE/SENSITIVE_ATTRIBUTE/epochs_EPOCHS/CONF_ID
```
where the argument --e stands for the path of a specific experiment.

[merge_tables.py](scripts/merge_tables.py) leverages the MD files created by _eval_info.py_ and merges them to create the paper tables.

```bash
python scripts/merge_tables.py --d DATASET --m GCMC LightGCN NGCF --sa gender
```

# RESULTS

![image](https://github.com/jackmedda/RS-BGExplainer/assets/26059819/8fa1bfb0-1a68-4b1d-bfc9-dcefc41683f8)

![image](https://github.com/jackmedda/RS-BGExplainer/assets/26059819/10108998-b447-4850-9dd0-75d7a61bc39a)
