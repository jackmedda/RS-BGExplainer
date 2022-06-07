from collections import defaultdict

import torch
from torch_geometric.utils import k_hop_subgraph, dense_to_sparse, to_dense_adj, subgraph


def get_degree_matrix(adj):
    return torch.diag(adj.sum(dim=1))


def get_neighbourhood(node_idx,
                      edge_index,
                      n_hops,
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

    edge_subset = k_hop_subgraph(node_idx, n_hops, edge_index)  # Get all nodes involved
    edge_subset = subgraph(edge_subset[0], edge_index)  # Get subset of edges
    if only_last_level and n_hops > 1:
        edge_subset = hop_difference(node_idx, edge_index, edge_subset, n_hops - 1)
    if not_user_sub_matrix and n_hops > 1:
        edge_subset = hop_difference(node_idx, edge_index, edge_subset, 1)

    sub_adj = to_dense_adj(edge_subset[0], max_num_nodes=max_num_nodes).squeeze().to_sparse()

    return sub_adj, edge_subset


def a(_input, target):
    _input = _input / 0.1

    def approx_ranks(inp):
        shape = inp.shape[1]

        a = torch.tile(torch.unsqueeze(inp, 2), [1, 1, shape])
        b = torch.tile(torch.unsqueeze(inp, 1), [1, shape, 1])
        return torch.sum(torch.nn.functional.sigmoid(b - a), dim=-1) + .5

    def inverse_max_dcg(_target,
                        gain_fn=lambda _target: torch.pow(2.0, _target) - 1.,
                        rank_discount_fn=lambda rank: 1. / rank.log1p()):
        ideal_sorted_target = torch.topk(_target, _target.shape[1]).values
        rank = torch.arange(ideal_sorted_target.shape[1]) + 1
        discounted_gain = gain_fn(ideal_sorted_target) * rank_discount_fn(rank.type(torch.FloatTensor))
        discounted_gain = torch.unsqueeze(torch.sum(discounted_gain, dim=1), 1)
        return torch.where(discounted_gain > 0., 1. / discounted_gain, torch.zeros_like(discounted_gain))

    def ndcg(_target, _ranks):
        discounts = 1. / ranks.type(torch.FloatTensor).log1p()
        gains = torch.pow(2., _target.type(torch.FloatTensor)) - 1.
        print(torch.sum(gains * discounts, dim=-1))
        dcg = torch.unsqueeze(torch.sum(gains * discounts, dim=-1), dim=1)
        return dcg * inverse_max_dcg(_target)

    ranks = approx_ranks(_input)

    return -ndcg(target, ranks)


def create_symm_matrix_from_vec(vector, n_rows, device=None):
    matrix = torch.zeros(n_rows, n_rows).to(device)
    idx = torch.tril_indices(n_rows, n_rows).to(device)
    matrix[idx[0], idx[1]] = vector
    symm_matrix = torch.tril(matrix) + torch.tril(matrix, -1).t()
    return symm_matrix


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
