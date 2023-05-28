import scipy
import torch
import numba
import numpy as np


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


def edges_filter_nodes(edges: torch.LongTensor, nodes: torch.LongTensor):
    try:
        adj_filter = torch.isin(edges, nodes).any(dim=0)
    except AttributeError:
        if edges.shape[0] == 1:
            adj_filter = torch.from_numpy(isin_backcomp(edges, nodes))
        else:
            adj_filter = torch.from_numpy(isin_backcomp(edges[0], nodes) | isin_backcomp(edges[1], nodes))
            # adj_filter = (edges[0][:, None] == nodes).any(-1) | (edges[1][:, None] == nodes).any(-1)
    return adj_filter


def isin_backcomp(ar1: torch.Tensor, ar2: torch.Tensor):
    ar1 = ar1.detach().numpy()
    ar2 = ar2.detach().numpy()
    return np.in1d(ar1, ar2)
