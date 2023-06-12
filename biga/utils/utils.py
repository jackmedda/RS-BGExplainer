import os
import re
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
from sklearn.decomposition import PCA
from recbole.data import create_dataset, data_preparation, create_samplers, get_dataloader
from recbole.utils import init_logger, get_model
from recbole.data.interaction import Interaction

try:
    from torch_geometric.utils import k_hop_subgraph, subgraph
except ModuleNotFoundError:
    pass


_EXPS_COLUMNS = [
    # "user_id",
    # "rec_topk",
    # "test_topk",
    # "rec_cf_topk",
    # "test_cf_topk",
    # "rec_cf_dist",
    # "test_cf_dist",
    "loss_total",
    "loss_graph_dist",
    "fair_loss",
    "fair_metric",
    "del_edges",
    "epoch",
    # "first_fair_loss"
]


_OLD_EXPS_COLUMNS = [
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


def exp_col_index(col):
    try:
        idx = _EXPS_COLUMNS.index(col)
    except ValueError:
        idx = col
    return idx


def old_exp_col_index(col):
    try:
        idx = _OLD_EXPS_COLUMNS.index(col)
    except ValueError:
        idx = col
    return idx


def wandb_init(config, policies=None, **kwargs):
    config = config.final_config_dict if not isinstance(config, dict) else config

    tags = None
    policies = config.get("explainer_policies", policies)
    if policies is not None:
        tags = [k for k in policies if policies[k]]
    config['wandb_tags'] = tags

    return wandb.init(
        **kwargs,
        tags=tags,
        config=config
    )


def load_data_and_model(model_file, explainer_config_file=None, cmd_config_args=None, return_exp_content=False):
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

    exp_file_content = None
    if explainer_config_file is not None:
        with open(explainer_config_file, 'r', encoding='utf-8') as f:
            exp_file_content = f.read()
            explain_config_dict = yaml.load(exp_file_content, Loader=config.yaml_loader)
        config.final_config_dict.update(explain_config_dict)

    if cmd_config_args is not None:
        for arg, val in cmd_config_args.items():
            conf = config
            if '.' in arg:
                subargs = arg.split('.')
                for subarg in subargs[:-1]:
                    conf = conf[subarg]
                arg = subargs[-1]

            if conf[arg] is None:
                try:
                    new_val = float(val)
                    new_val = int(new_val) if new_val.is_integer() else new_val
                except ValueError:
                    new_val = int(val) if val.isdigit() else val
                conf[arg] = new_val
            else:
                try:
                    arg_type = type(conf[arg])
                    if arg_type == bool:
                        new_val = val.title() == 'True'
                    else:
                        new_val = arg_type(val)  # cast to same type in config
                    conf[arg] = new_val
                except (ValueError, TypeError):
                    new_val = None

            if new_val is not None:
                exp_file_content = re.sub(arg + r':.*\n', f"{arg}: {new_val}\n", exp_file_content)

    config['data_path'] = config['data_path'].replace('\\', os.sep)
    # config['device'] = 'cuda'

    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)

    if 'group_explain' in config:
        config['explain_scope'] = 'group_explain' if config['group_explain'] else ('group' if config['user_batch_exp'] > 1 else 'individual')

    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))

    logger.info(model)

    ret = [config, model, dataset, train_data, valid_data, test_data]
    ret = ret + [exp_file_content] if return_exp_content else ret
    return tuple(ret)


def load_dp_exps_file(base_exps_file):
    cf_data_file = os.path.join(base_exps_file, 'cf_data.pkl')
    model_preds_file = os.path.join(base_exps_file, 'model_rec_test_preds.pkl')

    with open(cf_data_file, 'rb') as file:
        exps = [pickle.load(file)]
    with open(model_preds_file, 'rb') as file:
        model_preds = pickle.load(file)

    return (exps, *model_preds)


def load_old_dp_exps_file(base_exps_file):
    cf_data_file = os.path.join(base_exps_file, 'all_users.pkl')

    with open(cf_data_file, 'rb') as file:
        exps = [pickle.load(file)]

    return exps


def get_dataset_with_perturbed_edges(pert_edges, data_set):
    user_num = data_set.user_num
    uid_field, iid_field = data_set.uid_field, data_set.iid_field

    pert_edges = torch.tensor(pert_edges)
    pert_edges[1] -= user_num  # remap items in range [0, item_num)

    orig_inter_feat = data_set.inter_feat
    pert_inter_feat = {}
    for i, col in enumerate([uid_field, iid_field]):
        pert_inter_feat[col] = torch.cat((orig_inter_feat[col], pert_edges[i]))

    unique, counts = torch.stack(
        (pert_inter_feat[uid_field], pert_inter_feat[iid_field]),
    ).unique(dim=1, return_counts=True)
    pert_inter_feat[uid_field], pert_inter_feat[iid_field] = unique[:, counts == 1]

    return data_set.copy(Interaction(pert_inter_feat))


def get_dataloader_with_perturbed_edges(pert_edges, config, dataset, train_data, valid_data, test_data):
    train_dataset = get_dataset_with_perturbed_edges(pert_edges, train_data.dataset)
    valid_dataset = get_dataset_with_perturbed_edges(pert_edges, valid_data.dataset)
    test_dataset = get_dataset_with_perturbed_edges(pert_edges, test_data.dataset)

    built_datasets = [train_dataset, valid_dataset, test_dataset]
    train_sampler, valid_sampler, test_sampler = create_samplers(config, dataset, built_datasets)

    train_data = get_dataloader(config, 'train')(config, train_dataset, train_sampler, shuffle=False)
    valid_data = get_dataloader(config, 'evaluation')(config, valid_dataset, valid_sampler, shuffle=False)
    test_data = get_dataloader(config, 'evaluation')(config, test_dataset, test_sampler, shuffle=False)

    return train_data, valid_data, test_data

def get_best_exp_early_stopping(exps, config_dict):
    best_epoch = get_best_epoch_early_stopping(exps, config_dict)
    epoch_idx = exp_col_index('epoch')
    return [e for e in sorted(exps, key=lambda x: abs(x[epoch_idx] - best_epoch)) if e[epoch_idx] <= best_epoch][0]


def old_get_best_epoch_early_stopping(exps, config_dict):
    try:
        patience = config_dict['early_stopping']['patience']
    except TypeError:
        patience = config_dict['earlys_patience']

    return max([e[old_exp_col_index('epoch')] for e in exps]) - patience


def get_best_epoch_early_stopping(exps, config_dict):
    try:
        patience = config_dict['early_stopping']['patience']
    except TypeError:
        patience = config_dict['earlys_patience']

    return max([e[exp_col_index('epoch')] for e in exps]) - patience


def prepare_batched_data(input_data, data, item_data=None):
    """
    Prepare the batched data according to the "recbole" pipeline
    :param batched_data:
    :param data:
    :param item_data:
    :return:
    """
    data_df = Interaction({k: v[input_data] for k, v in data.dataset.user_feat.interaction.items()})

    if item_data is not None:
        data_df.update(Interaction({data.dataset.iid_field: item_data}))

    if hasattr(data, "uid2history_item"):
        history_item = data.uid2history_item[data_df[data.dataset.uid_field]]

        if len(input_data) > 1:
            history_u = torch.cat([torch.full_like(hist_iid, i) for i, hist_iid in enumerate(history_item)])
            history_i = torch.cat(list(history_item))
        else:
            history_u = torch.full_like(history_item, 0)
            history_i = history_item

        history_index = (history_u, history_i)
    else:
        history_index = None

    return data_df, history_index, None, None


def get_scores(model, batched_data, tot_item_num, test_batch_size, item_tensor, **kwargs):
    interaction, history_index, _, _ = batched_data
    inter_data = interaction.to(model.device)
    try:
        scores = model.full_sort_predict(inter_data, **kwargs)

    except NotImplementedError:
        inter_len = len(interaction)
        new_inter = interaction.to(model.device, **scores_kws).repeat_interleave(tot_item_num)
        batch_size = len(new_inter)
        new_inter.update(item_tensor.repeat(inter_len))
        if batch_size <= test_batch_size:
            scores = model.predict(new_inter)
        else:
            scores = Explainer._spilt_predict(new_inter, batch_size, test_batch_size, test_batch_size)

    scores = scores.view(-1, tot_item_num)
    scores[:, 0] = -np.inf
    if model.ITEM_ID in interaction:
        scores = scores[:, inter_data[model.ITEM_ID]]
    if history_index is not None:
        scores[history_index] = -np.inf

    return scores


def get_top_k(scores_tensor, topk=10):
    scores_top_k, topk_idx = torch.topk(scores_tensor, topk, dim=-1)  # n_users x k

    return scores_top_k, topk_idx


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
        except TypeError:  # python < 3.8
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
