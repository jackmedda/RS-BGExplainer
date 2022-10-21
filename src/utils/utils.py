import os
import copy
import pickle
from logging import getLogger
from collections import defaultdict

import yaml
import torch
import scipy
import numpy as np
import pandas as pd
import networkx as nx
import recbole.evaluator.collector as recb_collector
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


def load_data_and_model(model_file, explainer_config_file=None):
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

    if explainer_config_file is not None:
        with open(explainer_config_file, 'r', encoding='utf-8') as f:
            explain_config_dict = yaml.load(f.read(), Loader=config.yaml_loader)
        config.final_config_dict.update(explain_config_dict)

    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)

    if 'group_explain' in config:
        config['explain_scope'] = 'group_explain' if config['group_explain'] else ('group' if config['user_batch_exp'] > 1 else 'individual')

    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))

    return config, model, dataset, train_data, valid_data, test_data


def load_exps_file(base_exps_file):
    files = [f for f in os.scandir(base_exps_file) if 'user' in f.name]

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


def load_dp_exps_file(base_exps_file):
    files = [f for f in os.scandir(base_exps_file) if 'user' in f.name]

    exps = []
    for f in files:
        with open(f.path, 'rb') as file:
            exp = pickle.load(file)
        exps.append(exp)

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


def get_nx_biadj_matrix(dataset, remove_first_row_col=False):
    inter_matrix = dataset.inter_matrix(form='csr').astype(np.float32)
    if remove_first_row_col:
        inter_matrix = inter_matrix[1:, 1:]

    return nx.bipartite.from_biadjacency_matrix(inter_matrix)


def compute_metric(evaluator, dataset, pref_data, pred_col, metric):
    hist_matrix, _, _ = dataset.history_item_matrix()
    dataobject = recb_collector.DataStruct()
    uid_list = pref_data['user_id'].to_numpy()

    pos_matrix = np.zeros((dataset.user_num, dataset.item_num), dtype=int)
    pos_matrix[uid_list[:, None], hist_matrix[uid_list]] = 1
    pos_matrix[:, 0] = 0
    pos_len_list = torch.tensor(pos_matrix.sum(axis=1, keepdims=True))
    pos_idx = torch.tensor(pos_matrix[uid_list[:, None], np.stack(pref_data[pred_col].values)])
    pos_data = torch.cat((pos_idx, pos_len_list[uid_list]), dim=1)

    dataobject.set('rec.topk', pos_data)

    pos_index, pos_len = evaluator.metric_class[metric].used_info(dataobject)
    result = evaluator.metric_class[metric].metric_info(pos_index, pos_len)

    return result


def compute_metric_per_group(evaluator, data, user_df, pref_data, sens_attr, group_idxs, metric="ndcg", raw=False):
    m_idx, f_idx = group_idxs

    _m_ndcg = compute_metric(
        evaluator,
        data.dataset,
        pref_data.set_index('user_id').loc[user_df.loc[user_df[sens_attr] == m_idx, 'user_id']].reset_index(),
        'topk_pred',
        metric
    )[:, -1]
    _f_ndcg = compute_metric(
        evaluator,
        data.dataset,
        pref_data.set_index('user_id').loc[user_df.loc[user_df[sens_attr] == f_idx, 'user_id']].reset_index(),
        'topk_pred',
        metric
    )[:, -1]

    return (_m_ndcg, _f_ndcg) if raw else (_m_ndcg.mean(), _f_ndcg.mean())


def chunk_categorize(array_1d, n_chunks=10):
    array_1d = np.asarray(array_1d)
    a_min, a_max = array_1d.min(), array_1d.max()
    step = np.abs(a_max - a_min) / n_chunks
    if issubclass(array_1d.dtype.type, np.integer):
        step = round(step)

    mapped_a = np.empty_like(array_1d)
    for chunk in range(1, n_chunks + 1):
        if chunk == 1:
            mask = array_1d < step * chunk + a_min
        elif chunk == n_chunks + 1:
            mask = array_1d > step * chunk
        else:
            mask = (array_1d >= step * (chunk - 1) + a_min) & (array_1d <= step * chunk + a_min)

        mapped_a[mask] = step * chunk

    return mapped_a


def compute_uniform_categories_prob(_item_df, n_categories, raw=False):
    uni_cat_prob = np.zeros(n_categories)
    for cat_list in _item_df['class']:
        if cat_list:
            uni_cat_prob[cat_list] += 1

    return uni_cat_prob / (_item_df.shape[0] - 1) if not raw else (uni_cat_prob, (_item_df.shape[0] - 1))


def compute_category_sharing_prob(_item_df, n_categories=None, raw=False):
    if n_categories is None:
        n_categories = _item_df['class'].to_numpy().flatten().max() + 1

    cat_sharing_prob = np.zeros(n_categories)
    for cat_list in _item_df['class']:
        if len(cat_list) > 1:
            cat_sharing_prob[cat_list] += 1

    return cat_sharing_prob / (_item_df.shape[0] - 1) if not raw else (cat_sharing_prob, (_item_df.shape[0] - 1))


def compute_category_intersharing_distribution(_item_df, n_categories=None, raw=False):
    if n_categories is None:
        n_categories = _item_df['class'].to_numpy().flatten().max() + 1

    cat_intersharing = np.zeros((n_categories, n_categories), dtype=float)
    for cat_list in _item_df['class']:
        len_cl = len(cat_list)
        if len_cl > 1:
            for i, cat in enumerate(cat_list):
                cl_non_cat = np.array(cat_list[:i] + cat_list[(i + 1):])
                cat_intersharing[cat, cl_non_cat] += 1

    return cat_intersharing / (_item_df.shape[0] - 1) if not raw else (cat_intersharing, (_item_df.shape[0] - 1))


def get_category_inter_distribution_over_attrs(train_data, sens_attrs, norm=False, item_cats=None):
    inter_num = train_data.dataset.inter_num

    item_cats = item_cats if item_cats is not None else train_data.dataset.item_feat['class']

    n_total_cats = train_data.dataset.item_feat['class'].max() + 1

    item_df = pd.DataFrame({
        'item_id': train_data.dataset.item_feat['item_id'].numpy(),
        'class': map(lambda x: [el for el in x if el != 0], item_cats.numpy().tolist())
    })

    user_df = pd.DataFrame({
        'user_id': train_data.dataset.user_feat['user_id'].numpy(),
        **{sens_attr: train_data.dataset.user_feat[sens_attr].numpy() for sens_attr in sens_attrs}
    })

    train_df = pd.DataFrame(train_data.dataset.inter_feat.numpy())[["user_id", "item_id"]]
    joint_df = train_df.join(item_df.set_index('item_id'), on='item_id').join(user_df.set_index('user_id'), on='user_id')

    cats_inter_counts_attr = {}
    for attr in sens_attrs:
        cats_inter_counts_attr[attr] = {}
        for demo_gr, demo_df in joint_df.groupby(attr):
            class_counts = demo_df.explode('class').value_counts('class')
            class_values = np.zeros(n_total_cats, dtype=float)
            class_values[class_counts.index] = class_counts.values
            class_values[0] = np.nan
            if norm:
                cats_inter_counts_attr[attr][demo_gr] = class_values / inter_num
            else:
                cats_inter_counts_attr[attr][demo_gr] = class_values

    return cats_inter_counts_attr


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


def create_symm_matrix_from_vec(vector, n_rows, idx=None, base_symm='zeros'):
    if isinstance(base_symm, str):
        symm_matrix = getattr(torch, base_symm)(n_rows, n_rows).to(vector.device)
    else:
        symm_matrix = copy.deepcopy(base_symm)
    old_idx = idx
    if old_idx is None:
        idx = torch.tril_indices(n_rows, n_rows, -1)
    symm_matrix[idx[0], idx[1]] = vector
    symm_matrix[idx[1], idx[0]] = vector

    return symm_matrix


def create_sparse_symm_matrix_from_vec(vector, idx, base_symm):
    symm_matrix = copy.deepcopy(base_symm)
    if not symm_matrix.is_coalesced():
        symm_matrix = symm_matrix.coalesce()
    symm_matrix_idxs, symm_matrix_vals = symm_matrix.indices(), symm_matrix.values()

    edge_deletions = False
    if idx.shape[1] == symm_matrix_idxs.shape[1]:
        edge_deletions = (idx.sort(dim=1).values == symm_matrix_idxs.sort(dim=1).values).all()

    if not edge_deletions:  # if pass is edge additions
        idx = torch.cat((symm_matrix_idxs, idx, idx[[1, 0]]), dim=1)
        vector = torch.cat((symm_matrix_vals, vector, vector))
    symm_matrix = torch.sparse.FloatTensor(idx, vector, symm_matrix.shape)

    return symm_matrix


def perturbate_adj_matrix(graph_A, P_symm, mask_sub_adj, num_all, D_indices, pred=False):
    if pred:
        P_hat_symm = (torch.sigmoid(P_symm) >= 0.5).float()
        P = create_sparse_symm_matrix_from_vec(P_hat_symm, mask_sub_adj, graph_A)
        P_loss = P
    else:
        P = create_sparse_symm_matrix_from_vec(torch.sigmoid(P_symm), mask_sub_adj, graph_A)
        P_loss = None

    # Don't need gradient of this if pred is False
    D_tilde = torch.sparse.sum(P, dim=1) if pred else torch.sparse.sum(P, dim=1).detach()
    D_tilde_exp = (D_tilde.to_dense() + 1e-7).pow(-0.5)

    D_tilde_exp = torch.sparse.FloatTensor(D_indices, D_tilde_exp, torch.Size((num_all, num_all)))

    # # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
    return torch.sparse.mm(torch.sparse.mm(D_tilde_exp, P), D_tilde_exp), P_loss


def dense2d_to_sparse_without_nonzero(tensor):
    x, y = tensor.shape
    nonzero = (tensor != 0).detach().cpu()
    x_idxs = torch.repeat_interleave(torch.arange(x), y)[nonzero.flatten()]
    y_idxs = torch.tile(torch.arange(y), [x])[nonzero.flatten()]
    indices = torch.stack((x_idxs, y_idxs)).to(tensor.device)
    values = tensor[nonzero]
    return torch.sparse.FloatTensor(indices, values, torch.Size((x, y)))


def get_adj_matrix(interaction_matrix,
                   num_all,
                   n_users):
    A = scipy.sparse.dok_matrix((num_all, num_all), dtype=np.float32)
    inter_M = interaction_matrix
    inter_M_t = interaction_matrix.transpose()
    data_dict = dict(zip(zip(inter_M.row, inter_M.col + n_users), [1] * inter_M.nnz))
    data_dict.update(dict(zip(zip(inter_M_t.row + n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
    A._update(data_dict)
    A = A.tocoo()
    row = A.row
    col = A.col
    i = torch.LongTensor(np.stack([row, col], axis=0))
    data = torch.FloatTensor(A.data)
    adj = torch.sparse.FloatTensor(i, data, torch.Size(A.shape))
    edge_subset = [torch.LongTensor(i)]

    return adj, edge_subset[0]


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
        bias_disparity[attr] = {}
        for demo_group in train_bias[attr]:
            gr_idx = group_map[demo_group] if demo_group not in group_map else demo_group

            if train_bias[attr][demo_group] is None or rec_bias[attr][demo_group] is None:
                bias_disparity[attr][gr_idx] = None
                continue

            bias_r = rec_bias[attr][demo_group]
            bias_s = train_bias[attr][demo_group]
            bias_disparity[attr][gr_idx] = (bias_r - bias_s) / bias_s

    return bias_disparity


def compute_calibration(train_bias, rec_bias):
    train_bias_mean = train_bias.nanmean(dim=0)
    return (rec_bias.nanmean(dim=0) - train_bias_mean) / train_bias_mean


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


def compute_steck_distribution(_hist_matrix, _item_df, n_categories=None):
    _item_df = _item_df.set_index('item_id')

    if n_categories is None:
        n_categories = _item_df['class'].to_numpy().flatten().max() + 1

    p_gi = torch.zeros((_item_df.shape[0], n_categories))

    for _item_id, _item_data in _item_df.iterrows():
        if len(_item_data['class']) > 0:
            p_gi[_item_id, _item_data['class']] = 1 / len(_item_data['class'])

    p_gu = torch.zeros((_hist_matrix.shape[0], n_categories))

    for user_id, user_hist in enumerate(_hist_matrix):
        user_hist = user_hist[user_hist != 0]
        if len(user_hist) > 0:
            for item in user_hist:
                cats = _item_df.loc[item, 'class']
                p_gu[user_id, cats] += p_gi[item, cats]
            p_gu[user_id] /= len(user_hist)

    return p_gu


def generate_steck_pref_ratio(_train_data,
                              config,
                              sensitive_attr=None,
                              mapped_keys=False,
                              item_cats=None):
    sensitive_attr = 'gender' if sensitive_attr is None else sensitive_attr
    user_df = pd.DataFrame({
        'user_id': _train_data.dataset.user_feat[config['USER_ID_FIELD']].numpy(),
        sensitive_attr: _train_data.dataset.user_feat[sensitive_attr].numpy()
    })

    item_cats = item_cats if item_cats is not None else _train_data.dataset.item_feat['class']

    item_df = pd.DataFrame({
        'item_id': _train_data.dataset.item_feat[config['ITEM_ID_FIELD']].numpy(),
        'class': map(lambda x: [el for el in x if el != 0], item_cats.numpy().tolist())
    })

    sensitive_map = _train_data.dataset.field2id_token[sensitive_attr]
    item_categories_map = _train_data.dataset.field2id_token['class']

    history_matrix, _, history_len = _train_data.dataset.history_item_matrix()
    history_matrix, history_len = history_matrix.numpy(), history_len.numpy()

    p_gu = compute_steck_distribution(history_matrix, item_df, n_categories=len(item_categories_map))

    pref_ratio = dict.fromkeys([sensitive_attr])
    for attr, group_map in zip([sensitive_attr], [sensitive_map]):
        group_keys = [group_map[x] for x in range(len(group_map))] if mapped_keys else range(len(group_map))
        pref_ratio[attr] = dict.fromkeys(group_keys)
        for demo_group, demo_df in user_df.groupby(attr):
            gr_idx = group_map[demo_group] if mapped_keys else demo_group

            if group_map[demo_group] == '[PAD]':
                pref_ratio[attr][gr_idx] = None
                continue

            group_users = demo_df['user_id'].to_numpy()
            # TODO: only consider history len of users with data in pref_matrix
            pref_ratio[attr][gr_idx] = p_gu[group_users, :].mean(dim=0)

    return pref_ratio


def generate_bias_ratio(_train_data,
                        config,
                        sensitive_attr=None,
                        history_matrix: pd.DataFrame = None,
                        pred_col='cf_topk_pred',
                        mapped_keys=False,
                        user_subset=None,
                        item_cats=None):
    sensitive_attr = 'gender' if sensitive_attr is None else sensitive_attr
    user_df = pd.DataFrame({
        'user_id': _train_data.dataset.user_feat[config['USER_ID_FIELD']].numpy(),
        sensitive_attr: _train_data.dataset.user_feat[sensitive_attr].numpy()
    })

    item_cats = item_cats if item_cats is not None else _train_data.dataset.item_feat['class']

    item_df = pd.DataFrame({
        'item_id': _train_data.dataset.item_feat[config['ITEM_ID_FIELD']].numpy(),
        'class': map(lambda x: [el for el in x if el != 0], item_cats.numpy().tolist())
    })

    sensitive_map = _train_data.dataset.field2id_token[sensitive_attr]
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

    if user_subset is not None:
        mask = np.zeros(history_matrix.shape[0], dtype=bool)
        mask[user_subset] = True
        history_matrix[~mask, :] = 0
        history_len[~mask] = 0

    pref_matrix = compute_user_preference(history_matrix, item_df, n_categories=len(item_categories_map))
    uniform_categories_prob = compute_uniform_categories_prob(item_df, len(item_categories_map))

    bias_ratio, pref_ratio = dict.fromkeys([sensitive_attr]), dict.fromkeys([sensitive_attr])
    for attr, group_map in zip([sensitive_attr], [sensitive_map]):
        group_keys = [group_map[x] for x in range(len(group_map))] if mapped_keys else range(len(group_map))
        bias_ratio[attr], pref_ratio[attr] = dict.fromkeys(group_keys), dict.fromkeys(group_keys)
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
            pref_ratio[attr][gr_idx] = bias_ratio[attr][gr_idx]
            bias_ratio[attr][gr_idx] /= uniform_categories_prob

    return bias_ratio, pref_ratio


def generate_individual_bias_ratio(_train_data,
                                   config,
                                   history_matrix: pd.DataFrame = None,
                                   pred_col='cf_topk_pred',
                                   item_cats=None):
    user_df = pd.DataFrame({
        'user_id': _train_data.dataset.user_feat[config['USER_ID_FIELD']].numpy()
    })

    item_cats = item_cats if item_cats is not None else _train_data.dataset.item_feat['class']

    item_df = pd.DataFrame({
        'item_id': _train_data.dataset.item_feat[config['ITEM_ID_FIELD']].numpy(),
        'class': map(lambda x: [el for el in x if el != 0], item_cats.numpy().tolist())
    })

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
    uniform_categories_prob = compute_uniform_categories_prob(item_df, len(item_categories_map))

    pref_ratio = pref_matrix / history_len[:, None]
    bias_ratio = pref_ratio / uniform_categories_prob[None, :]

    return bias_ratio, pref_ratio


def clean_history_matrix(hist_m):
    for col in ['topk_pred', 'cf_topk_pred']:
        if col in hist_m and isinstance(hist_m.iloc[0][col], str):
            hist_m[col] = hist_m[col].map(lambda x: np.array(x[1:-1].strip().split(), int))


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))
    return unique


def unique_cat_recbole_interaction(inter, other, uid_field=None, iid_field=None, return_unique_counts=False):
    uid_field = uid_field or 'user_id'
    iid_field = iid_field or 'item_id'

    _inter = torch.stack((inter[uid_field], inter[iid_field]))
    if isinstance(other, dict):
        _other = torch.stack((other[uid_field], other[iid_field]))
    else:
        _other = torch.as_tensor(other)
    unique, counts = torch.cat((_inter, _other), dim=1).unique(dim=1, return_counts=True)
    new_inter = unique[:, counts == 1]

    if not return_unique_counts:
        return dict(zip([uid_field, iid_field], new_inter))
    else:
        return dict(zip([uid_field, iid_field], new_inter)), unique, counts


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

    __MAX_TOPK_ITEMS__ = 10000

    def __init__(self, size_average=None, reduce=None, topk=None, reduction: str = 'mean', temperature=0.1) -> None:
        super(NDCGApproxLoss, self).__init__(size_average, reduce, reduction)
        self.topk = topk
        self.temperature = temperature

    def forward(self, _input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if _input.shape[1] > self.__MAX_TOPK_ITEMS__:
            topk = self.topk or target.shape[1]
            _, _input_topk = torch.topk(_input, dim=1, k=topk)
            _input = _input[torch.arange(_input.shape[0])[:, None], _input_topk]
            target = target[torch.arange(target.shape[0])[:, None], _input_topk]

        _input_temp = torch.nn.ReLU()(_input) / self.temperature

        def approx_ranks(inp):
            shape = inp.shape[1]

            a = torch.tile(torch.unsqueeze(inp, 2), [1, 1, shape])
            a = torch.transpose(a, 1, 2) - a
            return torch.sum(torch.sigmoid(a), dim=-1) + .5

        def inverse_max_dcg(_target,
                            gain_fn=lambda _target: torch.pow(2.0, _target) - 1.,
                            rank_discount_fn=lambda rank: 1. / rank.log1p()):
            topk = self.topk or _target.shape[1]
            ideal_sorted_target = torch.topk(_target, topk).values
            rank = (torch.arange(ideal_sorted_target.shape[1]) + 1).to(_target.device)
            discounted_gain = gain_fn(ideal_sorted_target).to(_target.device) * rank_discount_fn(rank)
            discounted_gain = torch.sum(discounted_gain, dim=1, keepdim=True)
            return torch.where(discounted_gain > 0., 1. / discounted_gain, torch.zeros_like(discounted_gain))

        def ndcg(_target, _ranks):
            topk = self.topk or _target.shape[1]
            sorted_target, sorted_idxs = torch.topk(_target, topk)
            discounts = 1. / _ranks[torch.arange(_ranks.shape[0])[:, None], sorted_idxs].log1p()
            gains = torch.pow(2., sorted_target).to(_target.device) - 1.
            dcg = torch.sum(gains * discounts, dim=-1, keepdim=True)
            return dcg * inverse_max_dcg(_target)

        ranks = approx_ranks(_input_temp)

        return -ndcg(target, ranks)
