import os
import pickle
import logging
from logging import getLogger
from collections import defaultdict

import yaml
import torch
from torch_geometric.utils import k_hop_subgraph, dense_to_sparse, to_dense_adj, subgraph
from recbole.config import Config
from recbole.data import create_dataset, data_preparation, save_split_dataloaders, load_split_dataloaders
from recbole.utils import init_logger, get_model, get_trainer, init_seed, set_color


EXPS_COLUMNS = [
    "user_id",
    "topk",
    "cf_topk",
    "dist",
    "loss_total",
    "loss_pred",
    "loss_graph_dist",
    "fair_loss",
    "del_edges",
    "nnz_sub_adj",
    "first_fair_loss"
]


def load_data_and_model(model_file, explainer_config_file):
    r"""Load filtered dataset, split dataloaders and saved model.
    Args:
        model_file (str): The path of saved model file.
    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    checkpoint = torch.load(model_file)
    config = checkpoint['config']

    with open(explainer_config_file, 'r', encoding='utf-8') as f:
        explain_config_dict = yaml.load(f.read(), Loader=config.yaml_loader)
    config.final_config_dict.update(explain_config_dict)

    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)

    # specifying tot_item_num to the number of unique items makes the dataloader for evaluation to be batched
    # on interactions of one user at a time
    config['eval_batch_size'] = dataset.item_num
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))

    return config, model, dataset, train_data, valid_data, test_data


def load_exps_file(base_exps_file):
    files = [f for f in os.scandir(base_exps_file)]
    
    exps = {}
    for f in files:
        user_id = int(f.name.split('_')[1].split('.')[0])
        with open(f.path, 'rb') as file:
            exp = pickle.load(file)
        exps[user_id] = exp

    return exps


def get_degree_matrix(adj):
    return torch.diag(adj.sum(dim=1))


def get_neighbourhood(node_idx,
                      edge_index,
                      n_hops,
                      neighbors_hops=False,
                      only_last_level=False,
                      not_user_sub_matrix=False,
                      max_num_nodes=None):
    def hop_difference(_node_idx, _edge_index, edge_s, _n_hops):
        _edge_subset = k_hop_subgraph(_node_idx, _n_hops, _edge_index)
        _edge_subset = subgraph(_edge_subset[0], _edge_index)  # Get subset of edges

        # takes the non-intersection between last level subgraph and lower hop subgraph
        unique, counts = torch.cat((edge_s[0], _edge_subset[0]), dim=1).unique(dim=1, return_counts=True)
        edge_s = [unique[:, counts == 1]]
        return edge_s

    if neighbors_hops:
        n_hops = n_hops * 2

    edge_subset = k_hop_subgraph(node_idx, n_hops, edge_index)  # Get all nodes involved
    edge_subset = subgraph(edge_subset[0], edge_index)  # Get subset of edges
    if n_hops > 1:
        if only_last_level:
            edge_subset = hop_difference(node_idx, edge_index, edge_subset, n_hops - 1)
        if not_user_sub_matrix:
            edge_subset = hop_difference(node_idx, edge_index, edge_subset, 1)

    # sub_adj = to_dense_adj(edge_subset[0], max_num_nodes=max_num_nodes).squeeze().to_sparse()

    # return sub_adj, edge_subset
    return edge_subset


def create_symm_matrix_from_vec(vector, n_rows):
    symm_matrix = torch.zeros(n_rows, n_rows).to(vector.device)
    idx = torch.tril_indices(n_rows, n_rows, -1)
    symm_matrix[idx[0], idx[1]] = vector
    symm_matrix = symm_matrix + symm_matrix.t()

    return symm_matrix


def create_symm_matrix_from_sparse_tril(tril):
    return (tril + tril.t()).to_dense()


def create_vec_from_symm_matrix(matrix):
    idx = torch.tril_indices(matrix.shape[0], matrix.shape[0])
    vector = matrix[idx[0], idx[1]]
    return vector


def get_sparse_eye_mat(num):
    i = torch.LongTensor([range(0, num), range(0, num)])
    val = torch.FloatTensor([1] * num)
    return torch.sparse.FloatTensor(i, val)


def damerau_levenshtein_distance(s1, s2):
    """
    Copyright (c) 2015, James Turk
    https://github.com/jamesturk/jellyfish/blob/main/jellyfish/_jellyfish.py
    """

    # _check_type(s1)
    # _check_type(s2)

    len1 = len(s1)
    len2 = len(s2)
    infinite = len1 + len2

    # character array
    da = defaultdict(int)

    # distance matrix
    score = [[0] * (len2 + 2) for _ in range(len1 + 2)]

    score[0][0] = infinite
    for i in range(0, len1 + 1):
        score[i + 1][0] = infinite
        score[i + 1][1] = i
    for i in range(0, len2 + 1):
        score[0][i + 1] = infinite
        score[1][i + 1] = i

    for i in range(1, len1 + 1):
        db = 0
        for j in range(1, len2 + 1):
            i1 = da[s2[j - 1]]
            j1 = db
            cost = 1
            if s1[i - 1] == s2[j - 1]:
                cost = 0
                db = j

            score[i + 1][j + 1] = min(
                score[i][j] + cost,
                score[i + 1][j] + 1,
                score[i][j + 1] + 1,
                score[i1][j1] + (i - i1 - 1) + 1 + (j - j1 - 1),
            )
        da[s1[i - 1]] = i

    return score[len1 + 1][len2 + 1]


class NDCGApproxLoss(torch.nn.modules.loss._Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', temperature=0.1) -> None:
        super(NDCGApproxLoss, self).__init__(size_average, reduce, reduction)
        self.temperature = temperature

    def forward(self, _input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _input = _input / self.temperature

        def approx_ranks(inp):
            shape = inp.shape[1]

            a = torch.tile(torch.unsqueeze(inp, 2), [1, 1, shape])
            b = torch.tile(torch.unsqueeze(inp, 1), [1, shape, 1])
            return torch.sum(torch.sigmoid(b - a), dim=-1) + .5

        def inverse_max_dcg(_target,
                            gain_fn=lambda _target: torch.pow(2.0, _target) - 1.,
                            rank_discount_fn=lambda rank: 1. / rank.log1p()):
            ideal_sorted_target = torch.topk(_target, _target.shape[1]).values
            rank = (torch.arange(ideal_sorted_target.shape[1]) + 1).to(_target.device)
            discounted_gain = gain_fn(ideal_sorted_target).to(_target.device) * rank_discount_fn(rank)
            discounted_gain = torch.sum(discounted_gain, dim=1, keepdim=True)
            return torch.where(discounted_gain > 0., 1. / discounted_gain, torch.zeros_like(discounted_gain))

        def ndcg(_target, _ranks):
            discounts = 1. / _ranks.log1p()
            gains = torch.pow(2., _target).to(_target.device) - 1.
            dcg = torch.sum(gains * discounts, dim=-1, keepdim=True)
            return dcg * inverse_max_dcg(_target)

        ranks = approx_ranks(_input)

        return -ndcg(target, ranks)
