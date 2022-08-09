import os
import pickle
from logging import getLogger
from collections import defaultdict

import yaml
import torch
import scipy
import numpy as np
import pandas as pd
import networkx as nx
from torch_geometric.utils import k_hop_subgraph, subgraph
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_model, init_seed

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

    # # specifying tot_item_num to the number of unique items makes the dataloader for evaluation to be batched
    # # on interactions of one user at a time
    # if config['explain_scope'] == "group":
    #     config['eval_batch_size'] = dataset.item_num * config['user_batch_exp']
    # elif config['explain_scope'] == "individual":
    #     config['eval_batch_size'] = dataset.item_num
    # else:
    #     raise ValueError(f"`{config['explain_scope']}` is not in {['group', 'individual']}")
    config['explain_scope'] = 'group_explain' if config['group_explain'] else ('group' if config['user_batch_exp'] > 1 else 'individual')

    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))

    return config, model, dataset, train_data, valid_data, test_data


def load_exps_file(base_exps_file):
    files = [f for f in os.scandir(base_exps_file) if 'config' not in f.name]

    group_exp = False
    exps = []
    for f in files:
        if '#' in f.name or 'group_explain' in f.path:
            group_exp = True
            with open(f.path, 'rb') as file:
                exp = pickle.load(file)
            exps.append(exp)
        else:
            user_id = int(f.name.split('_')[1].split('.')[0])
            with open(f.path, 'rb') as file:
                exp = pickle.load(file)
            exps.append((user_id, exp))
    exps = dict(exps) if not group_exp else exps

    return exps


def get_nx_adj_matrix(config, dataset):
    USER_ID = config['USER_ID_FIELD']
    ITEM_ID = config['ITEM_ID_FIELD']
    n_users = dataset.num(USER_ID)
    n_items = dataset.num(ITEM_ID)
    inter_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
    num_all = n_users + n_items

    A = scipy.sparse.dok_matrix((num_all, num_all), dtype=np.float32)
    inter_M = inter_matrix
    inter_M_t = inter_matrix.transpose()
    data_dict = dict(zip(zip(inter_M.row, inter_M.col + n_users), [1] * inter_M.nnz))
    data_dict.update(dict(zip(zip(inter_M_t.row + n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
    A._update(data_dict)
    A = A.tocoo()
    return nx.Graph(A)


def get_nx_biadj_matrix(dataset):
    inter_matrix = dataset.inter_matrix(form='csr').astype(np.float32)

    return nx.bipartite.from_biadjacency_matrix(inter_matrix)


def compute_uniform_categories_prob(_item_df, _item_categories_map, raw=False):
    uni_cat_prob = np.zeros(_item_categories_map.shape)
    for cat_list in _item_df['class']:
        if cat_list:
            uni_cat_prob[cat_list] += 1

    return uni_cat_prob / (_item_df.shape[0] - 1) if not raw else (uni_cat_prob, (_item_df.shape[0] - 1))


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


def compute_bias_disparity(train_bias, rec_bias, train_data):
    bias_disparity = dict.fromkeys(list(train_bias.keys()))
    for attr in train_bias:
        group_map = train_data.dataset.field2id_token[attr]
        bias_disparity[attr] = dict.fromkeys(list(train_bias[attr].keys()))
        for demo_group in train_bias[attr]:
            gr_idx = group_map[demo_group] if demo_group not in group_map else demo_group

            if train_bias[attr][demo_group] is None or rec_bias[attr][demo_group] is None:
                bias_disparity[attr][gr_idx] = None
                continue

            bias_r = rec_bias[attr][demo_group]
            bias_s = train_bias[attr][demo_group]
            bias_disparity[attr][gr_idx] = (bias_r - bias_s) / bias_s

    return bias_disparity


def compute_user_preference(_hist_matrix, _item_df, n_categories=None):
    _item_df = _item_df.set_index('item_id')

    if n_categories is None:
        n_categories = _item_df['class'].to_numpy().flatten().max() + 1

    _pref_matrix = torch.zeros((_hist_matrix.shape[0], n_categories))

    for user_id, user_hist in enumerate(_hist_matrix):
        user_hist = user_hist[user_hist != 0]
        if len(user_hist) > 0:
            for item in user_hist:
                _pref_matrix[user_id, _item_df.loc[item, 'class']] += 1

    return _pref_matrix


def generate_bias_ratio(_train_data,
                        config,
                        sensitive_attrs=None,
                        history_matrix: pd.DataFrame = None,
                        pred_col='cf_topk_pred',
                        mapped_keys=False):
    sensitive_attrs = ['gender', 'age'] if sensitive_attrs is None else sensitive_attrs
    user_df = pd.DataFrame({
        'user_id': _train_data.dataset.user_feat[config['USER_ID_FIELD']].numpy(),
        **{sens_attr: _train_data.dataset.user_feat[sens_attr].numpy() for sens_attr in sensitive_attrs}
    })

    item_df = pd.DataFrame({
        'item_id': _train_data.dataset.item_feat[config['ITEM_ID_FIELD']].numpy(),
        'class': map(lambda x: [el for el in x if el != 0], _train_data.dataset.item_feat['class'].numpy().tolist())
    })

    sensitive_maps = [_train_data.dataset.field2id_token[sens_attr] for sens_attr in sensitive_attrs]
    item_categories_map = _train_data.dataset.field2id_token['class']

    if history_matrix is None:
        history_matrix, _, history_len = _train_data.dataset.history_item_matrix()
        history_matrix, history_len = history_matrix.numpy(), history_len.numpy()
    else:
        clean_history_matrix(history_matrix)
        topk = len(history_matrix.iloc[0][pred_col])
        history_len = np.zeros(user_df.shape[0])
        history_len[history_matrix['user_id']] = topk
        history_df = user_df[['user_id']].join(history_matrix.set_index('user_id'))
        history_topk = history_df[pred_col]
        history_df[pred_col] = history_topk.apply(lambda x: np.zeros(topk, dtype=int) if np.isnan(x).all() else x)

        history_matrix = np.stack(history_df[pred_col].values)

    pref_matrix = compute_user_preference(history_matrix, item_df, n_categories=len(item_categories_map))
    uniform_categories_prob = compute_uniform_categories_prob(item_df, item_categories_map)

    bias_ratio = dict.fromkeys(sensitive_attrs)
    for attr, group_map in zip(sensitive_attrs, sensitive_maps):
        group_keys = [group_map[x] for x in range(len(group_map))] if mapped_keys else range(len(group_map))
        bias_ratio[attr] = dict.fromkeys(group_keys)
        for demo_group, demo_df in user_df.groupby(attr):
            gr_idx = group_map[demo_group] if mapped_keys else demo_group

            if group_map[demo_group] == '[PAD]':
                bias_ratio[attr][gr_idx] = None
                continue

            group_users = demo_df['user_id'].to_numpy()
            # TODO: only consider history len of users with data in pref_matrix
            group_pref = pref_matrix[group_users, :].sum(dim=0)
            group_history_len = history_len[group_users].sum()
            bias_ratio[attr][gr_idx] = group_pref / group_history_len
            bias_ratio[attr][gr_idx] /= uniform_categories_prob

    return bias_ratio


def clean_history_matrix(hist_m):
    for col in ['topk_pred', 'cf_topk_pred']:
        if col in hist_m and isinstance(hist_m.iloc[0][col], str):
            hist_m[col] = hist_m[col].map(lambda x: np.array(x[1:-1].strip().split(), int))


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))
    return unique


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
