import scipy
import torch
import numpy as np
import igraph as ig
import networkx as nx

import gnnuers.utils as utils


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


def perturb_adj_matrix(graph_A, P_symm, mask_sub_adj, num_all, D_indices, pred=False, edge_deletions=False, mask_filter=None):
    if pred:
        P_hat_symm = (torch.sigmoid(P_symm) >= 0.5).float()
        P = utils.create_sparse_symm_matrix_from_vec(P_hat_symm, mask_sub_adj, graph_A, edge_deletions=edge_deletions, mask_filter=mask_filter)
        P_loss = P
    else:
        P = utils.create_sparse_symm_matrix_from_vec(torch.sigmoid(P_symm), mask_sub_adj, graph_A, edge_deletions=edge_deletions, mask_filter=mask_filter)
        P_loss = None

    # Don't need gradient of this if pred is False
    D_tilde = torch.sparse.sum(P, dim=1) if pred else torch.sparse.sum(P, dim=1).detach()
    D_tilde_exp = (D_tilde.to_dense() + 1e-7).pow(-0.5)

    D_tilde_exp = torch.sparse.FloatTensor(D_indices, D_tilde_exp, torch.Size((num_all, num_all)))

    # # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
    return torch.sparse.mm(torch.sparse.mm(D_tilde_exp, P), D_tilde_exp), P_loss
