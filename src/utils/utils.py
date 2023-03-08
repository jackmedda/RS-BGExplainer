import os
import copy
import stat
import shutil
import pickle
from logging import getLogger

import yaml
import wandb
import torch
import scipy
import numba
import numpy as np
import pandas as pd
import igraph as ig
import networkx as nx
import recbole.evaluator.collector as recb_collector
from sklearn.decomposition import PCA
# from torch_geometric.utils import k_hop_subgraph, subgraph
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_model, init_seed
from recbole.data.interaction import Interaction


EXPS_COLUMNS = [
    "user_id",
    "rec_topk",
    "test_topk",
    "rec_cf_topk",
    "test_cf_topk",
    "rec_cf_dist",
    "test_cf_dist",
    "loss_total",
    "loss_graph_dist",
    "fair_loss",
    "del_edges",
    "epoch",
    "first_fair_loss"
]


def wandb_init(config, **kwargs):
    config['wandb_tags'] = [k for k in config['explainer_policies'] if config['explainer_policies'][k]]


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


def load_dp_exps_file(base_exps_file):
    files = [f for f in os.scandir(base_exps_file) if 'user' in f.name]

    exps = []
    for f in files:
        with open(f.path, 'rb') as file:
            exp = pickle.load(file)
        exps.append(exp)

    return exps


def get_dataset_with_perturbed_edges(del_edges, train_dataset):
    user_num = train_dataset.user_num
    uid_field, iid_field = train_dataset.uid_field, train_dataset.iid_field

    del_edges = torch.tensor(del_edges)
    del_edges[1] -= user_num  # remap items in range [0, user_num)

    orig_inter_feat = train_dataset.inter_feat
    pert_inter_feat = {}
    for i, col in enumerate([uid_field, iid_field]):
        pert_inter_feat[col] = torch.cat((orig_inter_feat[col], del_edges[i]))

    unique, counts = torch.stack(
        (pert_inter_feat[uid_field], pert_inter_feat[iid_field]),
    ).unique(dim=1, return_counts=True)
    pert_inter_feat[uid_field], pert_inter_feat[iid_field] = unique[:, counts == 1]

    return train_dataset.copy(Interaction(pert_inter_feat))


def get_best_exp_early_stopping(exps, config_dict):
    try:
        patience = config_dict['early_stopping']['patience']
    except TypeError as e:
        patience = config_dict['earlys_patience']

    return max([e[EXPS_COLUMNS.index('epoch')] for e in exps]) - patience


def get_adj_from_inter_matrix(inter_matrix, num_all, n_users):
    A = scipy.sparse.dok_matrix((num_all, num_all), dtype=np.float32)
    inter_M = inter_matrix
    inter_M_t = inter_matrix.transpose()
    data_dict = dict(zip(zip(inter_M.row, inter_M.col + n_users), [1] * inter_M.nnz))
    data_dict.update(dict(zip(zip(inter_M_t.row + n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
    A._update(data_dict)
    return A.tocoo()


def get_adj_matrix(interaction_matrix,
                   num_all,
                   n_users):
    A = get_adj_from_inter_matrix(interaction_matrix, num_all, n_users)
    row = A.row
    col = A.col
    i = torch.LongTensor(np.stack([row, col], axis=0))
    data = torch.FloatTensor(A.data)
    adj = torch.sparse.FloatTensor(i, data, torch.Size(A.shape))
    edge_subset = torch.LongTensor(i)

    return adj, edge_subset


def get_nx_adj_matrix(dataset):
    uid_field = dataset.uid_field
    iid_field = dataset.iid_field
    n_users = dataset.num(uid_field)
    n_items = dataset.num(iid_field)
    inter_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
    num_all = n_users + n_items
    A = get_adj_from_inter_matrix(inter_matrix, num_all, n_users)

    return nx.Graph(A)


def get_nx_biadj_matrix(dataset, remove_first_row_col=False):
    inter_matrix = dataset.inter_matrix(form='csr').astype(np.float32)
    if remove_first_row_col:
        inter_matrix = inter_matrix[1:, 1:]

    return nx.bipartite.from_biadjacency_matrix(inter_matrix)


def get_bipartite_igraph(dataset, remove_first_row_col=False):
    inter_matrix = dataset.inter_matrix(form='csr').astype(np.float32)
    if remove_first_row_col:
        inter_matrix = inter_matrix[1:, 1:]

    incid_adj = ig.Graph.Incidence(inter_matrix.todense().tolist())
    bip_info = np.concatenate([np.zeros(inter_matrix.shape[0], dtype=int), np.ones(inter_matrix.shape[1], dtype=int)])

    return ig.Graph.Bipartite(bip_info, incid_adj.get_edgelist())


def get_node_node_graph_data(history):
    return _get_node_node_graph_data(history)


def get_decomposed_adj_matrix(del_edges, train_dataset, method='PCA'):
    pert_train_dataset = get_dataset_with_perturbed_edges(del_edges, train_dataset)
    train_adj = train_dataset.inter_matrix(form='csr').astype(np.float32)[1:, 1:].todense()
    pert_train_adj = pert_train_dataset.inter_matrix(form='csr').astype(np.float32)[1:, 1:].todense()

    if method.upper() == "PCA":
        decomposer = PCA(n_components=2)
    else:
        raise NotImplementedError("Only PCA is a supported decomposer")

    train_pca = decomposer.fit_transform(train_adj)
    pert_train_pca = decomposer.fit_transform(pert_train_adj)

    return train_pca, pert_train_pca


@numba.jit(nopython=True, parallel=True)
def _get_node_node_graph_data(history):
    hist = [set(h) for h in history]
    n_nodes = history.shape[0] - 1  # removed padding 0
    node_node = np.zeros((n_nodes * (n_nodes - 1) // 2, 3), dtype=np.int32)  # number of combinations

    for n1 in numba.prange(1, n_nodes + 1):
        _inner_combinations(n1, n_nodes, hist, node_node)

    return node_node


@numba.jit(nopython=True, parallel=True)
def _inner_combinations(n1, n_nodes, hist, node_node):
    for n2 in numba.prange(n1 + 1, n_nodes + 1):
        node_node[sum(range(n_nodes - n1 + 1, n_nodes)) + (n2 - n1 - 1)] = [
            n1, n2, len(hist[n1] & hist[n2]) - 1
        ]


def get_user_reachability(graph, last_user_id):
    return get_reachability_per_node(graph, last=last_user_id)


def get_item_reachability(graph, first_item_id):
    return get_reachability_per_node(graph, first=first_item_id)


def get_reachability_per_node(graph, first=None, last=None, nodes=None):
    reach = {}
    dist = np.array(graph.distances())

    if nodes is not None:
        nodes = sorted(nodes)
        dist = dist[nodes][:, nodes]
    else:
        nodes = np.arange(dist.shape[0])

        dist = dist[first:][:, first:] if first is not None else dist
        nodes = nodes[first:] if first is not None else nodes

        dist = dist[:(last + 1)][:, :(last + 1)] if last is not None else dist
        nodes = nodes[:(last + 1)] if last is not None else nodes

    for n, n_dist in zip(nodes, dist):
        n_reach = np.bincount(n_dist[(~np.isinf(n_dist)) & (n_dist > 0)].astype(int))
        n_reach = n_reach[n_reach > 0]
        reach[n] = compute_reachability(n_reach) / len(nodes)

    return reach


def compute_reachability(reach):
    return sum([reach[i] / (i + 1) for i in range(len(reach))])


@numba.jit(nopython=True, parallel=True)
def compute_sharing_potentiality(common_data, hist_len, length=2, depth=2):
    n = common_data.shape[0]
    res = np.zeros((n,), dtype=np.float32)
    for i in numba.prange(n):
        most_sim = np.argsort(common_data[i])[::-1][:length]
        res[i] += _compute_sp_length(i, most_sim, common_data, hist_len)
        for d in range(depth - 1):
            most_d_sim = np.argsort(common_data[most_sim[d]])[::-1][:length]
            res[i] += _compute_sp_length(most_sim[d], most_d_sim, common_data, hist_len) / (2 + d)
    return res


@numba.jit(nopython=True)
def _compute_sp_length(data_i, most_sim, common_data, hist_len):
    sp_length = 0
    for i in numba.prange(most_sim.shape[0]):
        sim = common_data[data_i, most_sim[i]]
        if hist_len[data_i] == 0 and hist_len[most_sim[i]] > 0:
            sp_length += 1  # most_sim[i] can share every item
        elif hist_len[data_i] == 0 or hist_len[most_sim[i]] == 0:
            sp_length += 0
        else:
            sp_length += sim / hist_len[data_i] * (1 - sim / hist_len[most_sim[i]])
    return sp_length / most_sim.shape[0]


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
    if metric in ['hit', 'mrr']:
        result = evaluator.metric_class[metric].metric_info(pos_index)
    else:
        result = evaluator.metric_class[metric].metric_info(pos_index, pos_len)

    return result


def compute_metric_per_group(evaluator, data, user_df, pref_data, sens_attr, group_idxs, col='topk_pred', metric="ndcg", raw=False):
    m_idx, f_idx = group_idxs

    m_group_mask = pref_data.user_id.isin(user_df.loc[user_df[sens_attr] == m_idx, data.dataset.uid_field])
    f_group_mask = pref_data.user_id.isin(user_df.loc[user_df[sens_attr] == f_idx, data.dataset.uid_field])

    metric_result = compute_metric(
        evaluator,
        data.dataset,
        pref_data,
        col,
        metric
    )[:, -1]

    _m_result, _f_result = metric_result[m_group_mask], metric_result[f_group_mask]

    return (_m_result, _f_result) if raw else (_m_result.mean(), _f_result.mean())


def compute_DP_across_random_samples(df, sens_attr, demo_group_field, dataset, metric, iterations=100, batch_size=64, seed=124):
    np.random.seed(seed)

    if not hasattr(compute_DP_across_random_samples, "generated_groups"):
        compute_DP_across_random_samples.generated_groups = {}

    df = df.sort_values(demo_group_field)
    max_user = df['user_id'].max() + 1

    n_users = 0
    demo_groups_order = []
    size_perc = np.zeros((2,), dtype=float)
    groups = np.zeros((2, max_user), dtype=int)
    for i, (dg, gr_df) in enumerate(df.groupby(demo_group_field)):
        gr_users = gr_df['user_id'].unique()
        groups[i, gr_users] = 1
        n_users += gr_users.shape[0]
        size_perc[i] = gr_users.shape[0]
        demo_groups_order.append(dg)
    size_perc /= n_users

    gr_data = np.zeros(max_user)
    for gr_users in groups:
        pos = gr_users.nonzero()[0]
        gr_data[pos] = df.set_index('user_id').loc[pos, metric].to_numpy()

    if (dataset, sens_attr) not in compute_DP_across_random_samples.generated_groups:
        compute_DP_across_random_samples.generated_groups[(dataset, sens_attr)] = np.zeros(
            (iterations, 2, max_user), dtype=np.bool_
        )

    return _compute_DP_random_samples(
        gr_data,
        groups,
        size_perc,
        compute_DP_across_random_samples.generated_groups[(dataset, sens_attr)],
        batch_size=batch_size,
        iterations=iterations
    ), demo_groups_order


@numba.jit(nopython=True, parallel=True)
def _compute_DP_random_samples(group_data, groups, size_perc, out_samples, batch_size=64, iterations=100):
    out = np.empty((iterations, 3), dtype=np.float32)
    check = out_samples.nonzero()[0].shape[0] == 0
    for i in numba.prange(iterations):
        if check:
            samples = np.zeros_like(groups, dtype=np.bool_)
            for gr_i in range(len(groups)):
                sample_size = round(batch_size * size_perc[gr_i])
                samples[gr_i][np.random.choice(groups[gr_i].nonzero()[0], sample_size, replace=False)] = True
            out_samples[i] = samples

        gr1_mean = group_data[out_samples[i, 0]].mean()
        gr2_mean = group_data[out_samples[i, 1]].mean()
        out[i] = [gr1_mean, gr2_mean, np.abs(gr1_mean - gr2_mean)]

    return out


def get_degree_matrix(adj):
    return torch.diag(adj.sum(dim=1))


# def get_neighbourhood(node_idx,
#                       edge_index,
#                       n_hops,
#                       neighbors_hops=False,
#                       only_last_level=False,
#                       not_user_sub_matrix=False,
#                       max_num_nodes=None):
#     def hop_difference(_node_idx, _edge_index, edge_s, _n_hops):
#         _edge_subset = k_hop_subgraph(_node_idx, _n_hops, _edge_index)
#         _edge_subset = subgraph(_edge_subset[0], _edge_index)  # Get subset of edges
#
#         # takes the non-intersection between last level subgraph and lower hop subgraph
#         unique, counts = torch.cat((edge_s[0], _edge_subset[0]), dim=1).unique(dim=1, return_counts=True)
#         edge_s = [unique[:, counts == 1]]
#         return edge_s
#
#     if neighbors_hops:
#         n_hops = n_hops * 2
#
#     edge_subset = k_hop_subgraph(node_idx, n_hops, edge_index)  # Get all nodes involved
#     edge_subset = subgraph(edge_subset[0], edge_index)  # Get subset of edges
#     if n_hops > 1:
#         if only_last_level:
#             edge_subset = hop_difference(node_idx, edge_index, edge_subset, n_hops - 1)
#         if not_user_sub_matrix:
#             edge_subset = hop_difference(node_idx, edge_index, edge_subset, 1)
#
#     # sub_adj = to_dense_adj(edge_subset[0], max_num_nodes=max_num_nodes).squeeze().to_sparse()
#
#     # return sub_adj, edge_subset
#     return edge_subset


def is_symmetrically_sorted(idxs: torch.Tensor):
    assert idxs.shape[0] == 2
    symm_offset = idxs.shape[1] // 2
    return (idxs[:, :symm_offset] == idxs[[1, 0], symm_offset:]).all()


def get_sorter_indices(base_idxs, to_sort_idxs):
    unique, inverse = torch.cat((base_idxs, to_sort_idxs), dim=1).unique(dim=1, return_inverse=True)
    inv_base, inv_to_sort = torch.split(inverse, to_sort_idxs.shape[1])
    sorter = torch.arange(to_sort_idxs.shape[1], device=inv_to_sort.device)[torch.argsort(inv_to_sort)]

    return sorter, inv_base


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


def create_sparse_symm_matrix_from_vec(vector, idx, base_symm, edge_deletions=False, mask_filter=None):
    symm_matrix = copy.deepcopy(base_symm)
    if not symm_matrix.is_coalesced():
        symm_matrix = symm_matrix.coalesce()
    symm_matrix_idxs, symm_matrix_vals = symm_matrix.indices(), symm_matrix.values()

    if not edge_deletions:  # if pass is edge additions
        idx = torch.cat((symm_matrix_idxs, idx, idx[[1, 0]]), dim=1)
        vector = torch.cat((symm_matrix_vals, vector, vector))
    else:
        vector = torch.cat((vector, vector))
        if mask_filter is not None:
            sorter, idx_inverse = get_sorter_indices(idx, symm_matrix_idxs)
            symm_matrix_vals = symm_matrix_vals[sorter][idx_inverse]
            assert is_symmetrically_sorted(symm_matrix_idxs[:, sorter][:, idx_inverse])
            symm_matrix_vals[mask_filter] = vector
            vector = symm_matrix_vals

    symm_matrix = torch.sparse.FloatTensor(idx, vector, symm_matrix.shape)

    return symm_matrix


def perturb_adj_matrix(graph_A, P_symm, mask_sub_adj, num_all, D_indices, pred=False, edge_deletions=False, mask_filter=None):
    if pred:
        P_hat_symm = (torch.sigmoid(P_symm) >= 0.5).float()
        P = create_sparse_symm_matrix_from_vec(P_hat_symm, mask_sub_adj, graph_A, edge_deletions=edge_deletions, mask_filter=mask_filter)
        P_loss = P
    else:
        P = create_sparse_symm_matrix_from_vec(torch.sigmoid(P_symm), mask_sub_adj, graph_A, edge_deletions=edge_deletions, mask_filter=mask_filter)
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


def rolling_window(input_array, size_kernel, stride, op="mean"):
    """Function to get rolling windows.
    https://stackoverflow.com/a/59781066

    Arguments:
        input_array {numpy.array} -- Input, by default it only works with depth equals to 1.
                                      It will be treated as a (height, width) image. If the input have (height, width, channel)
                                      dimensions, it will be rescaled to two-dimension (height, width)
        size_kernel {int} -- size of kernel to be applied. Usually 3,5,7. It means that a kernel of (size_kernel, size_kernel) will be applied
                             to the image.
        stride {int or tuple} -- horizontal and vertical displacement

    Returns:
        [list] -- A list with the resulting numpy.arrays
    """
    # Check right input dimension
    assert len(input_array.shape) in {1, 2}, \
        "input_array must have dimension 2 or 3. Yours have dimension {}".format(len(input_array))

    if input_array.shape == 3:
        input_array = input_array[:, :, 0]

    # Stride: horizontal and vertical displacement
    if isinstance(stride, int):
        sh, sw = stride, stride
    elif isinstance(stride, tuple):
        sh, sw = stride
    else:
        raise NotImplementedError("stride format not supported")

    # Input dimension (height, width)
    n_ah, n_aw = input_array.shape

    # Filter dimension (or window)
    if isinstance(size_kernel, int):
        n_h, n_w = size_kernel, size_kernel
    elif isinstance(size_kernel, tuple):
        n_h, n_w = size_kernel
    else:
        raise NotImplementedError("size_kernel format not supported")

    dim_out_h = int(np.floor((n_ah - n_h) / sh + 1))
    dim_out_w = int(np.floor((n_aw - n_w) / sw + 1))

    return _inner_rolling_window(input_array, dim_out_h, dim_out_w, n_h, n_w, sh, sw, op)


@numba.jit(nopython=True, parallel=True)
def _inner_rolling_window(input_array, dim_out_h, dim_out_w, n_h, n_w, sh, sw, op):
    # List to save output arrays
    rolled_array = np.zeros((dim_out_h, dim_out_w), dtype=np.float32)

    other_args = [dim_out_w, n_h, n_w, sh, sw]

    # Initialize row position
    for i in numba.prange(dim_out_h):
        start_row = sh * i
        _inner_inner_rolling_window(input_array, rolled_array, i, start_row, op, other_args)

    return rolled_array


@numba.jit(nopython=True)
def _inner_inner_rolling_window(input_array, rolled_array, i, start_row, op, other_args):
    dim_out_w, n_h, n_w, sh, sw = other_args
    for j in numba.prange(dim_out_w):
        start_col = sw * j

        # Get one window
        sub_array = input_array[start_row:(start_row + n_h), start_col:(start_col + n_w)]

        if op == "mean":
            agg_window = np.mean(sub_array)
        elif op == "sum":
            agg_window = np.sum(sub_array)
        else:
            raise TypeError("op should be one in ['sum', 'mean']")

        # Append sub_array
        rolled_array[i, j] = agg_window


def damerau_levenshtein_distance(s1, s2):
    s1 = [s1] if np.ndim(s1) == 1 else s1
    s2 = [s2] if np.ndim(s2) == 1 else s2

    out = np.zeros((len(s1, )), dtype=int)
    for i, (_s1, _s2) in enumerate(zip(s1, s2)):
        try:
            numb_s1, numb_s2 = numba.typed.List(_s1), numba.typed.List(_s2)
        except TypeError:
            numb_s1, numb_s2 = numba.typed.List(), numba.typed.List()
            for el in _s1:
                numb_s1.append(el)
            for el in _s2:
                numb_s2.append(el)

        out[i] = _damerau_levenshtein_distance(numb_s1, numb_s2)

    return out.item() if out.shape == (1,) else out


@numba.jit(nopython=True)
def _damerau_levenshtein_distance(s1, s2):
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
    da = {}

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
            i1 = da[s2[j - 1]] if s2[j - 1] in da else 0
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


def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
        shutil.copystat(src, dst)
    lst = os.listdir(src)
    if ignore:
        excl = ignore(src, lst)
        lst = [x for x in lst if x not in excl]
    for item in lst:
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if symlinks and os.path.islink(s):
            if os.path.lexists(d):
                os.remove(d)
            os.symlink(os.readlink(s), d)
            try:
                st = os.lstat(s)
                mode = stat.S_IMODE(st.st_mode)
                os.lchmod(d, mode)
            except:
                pass  # lchmod not available
        elif os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)
