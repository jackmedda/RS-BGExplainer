# -*- coding: utf-8 -*-
# @Time   : 2020/9/1 14:00
# @Author : Changxin Tian
# @Email  : cx.tian@outlook.com

# UPDATE
# @Time   : 2020/10/1
# @Author : Changxin Tian
# @Email  : cx.tian@outlook.com

r"""
GCMC
################################################

Reference:
    van den Berg et al. "Graph Convolutional Matrix Completion." in SIGKDD 2018.

Reference code:
    https://github.com/riannevdberg/gc-mc
"""

import sys
import math

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.layers import SparseDropout
from recbole.utils import InputType

sys.path.append('..')

import src.utils as utils


class GCMCPerturbated(GeneralRecommender):
    r"""GCMC is a model that incorporate graph autoencoders for recommendation.

    Graph autoencoders are comprised of:

    1) a graph encoder model :math:`Z = f(X; A)`, which take as input an :math:`N \times D` feature matrix X and
    a graph adjacency matrix A, and produce an :math:`N \times E` node embedding matrix
    :math:`Z = [z_1^T,..., z_N^T ]^T`;

    2) a pairwise decoder model :math:`\hat A = g(Z)`, which takes pairs of node embeddings :math:`(z_i, z_j)` and
    predicts respective entries :math:`\hat A_{ij}` in the adjacency matrix.

    Note that :math:`N` denotes the number of nodes, :math:`D` the number of input features,
    and :math:`E` the embedding size.

    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset, user_id):
        super(GCMCPerturbated, self).__init__(config, dataset)

        # load dataset info
        self.num_all = self.n_users + self.n_items
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)  # csr

        # load parameters info
        self.dropout_prob = config['dropout_prob']
        self.sparse_feature = config['sparse_feature']
        self.gcn_output_dim = config['gcn_output_dim']
        self.dense_output_dim = config['embedding_size']
        self.n_class = config['class_num']
        self.num_basis_functions = config['num_basis_functions']

        # generate node feature
        if self.sparse_feature:
            features = utils.get_sparse_eye_mat(self.num_all)
            i = features._indices()
            v = features._values()
            self.user_features = torch.sparse.FloatTensor(
                i[:, :self.n_users], v[:self.n_users], torch.Size([self.n_users, self.num_all])
            ).to(self.device)
            item_i = i[:, self.n_users:]
            item_i[0, :] = item_i[0, :] - self.n_users
            self.item_features = torch.sparse.FloatTensor(
                item_i, v[self.n_users:], torch.Size([self.n_items, self.num_all])
            ).to(self.device)
        else:
            features = torch.eye(self.num_all).to(self.device)
            self.user_features, self.item_features = torch.split(features, [self.n_users, self.n_items])
        self.input_dim = self.user_features.shape[1]

        self.n_hops = config['n_hops']
        self.neighbors_hops = config['neighbors_hops']
        self.beta = config['cf_beta']
        self.fair_beta = config['fair_beta']
        self.sub_matrix_only_last_level = config['sub_matrix_only_last_level']
        self.not_user_sub_matrix = config['not_user_sub_matrix']
        self.only_subgraph = config['only_subgraph']
        self.not_pred = config['not_pred']
        self.pred_same = config['pred_same']

        self.edge_additions = config['edge_additions']

        # adj matrices for each relation are stored in self.support
        self.Graph, self.sub_Graph = self.get_adj_matrix(
            user_id,
            self.n_hops,
            neighbors_hops=self.neighbors_hops,
            only_last_level=self.sub_matrix_only_last_level,
            not_user_sub_matrix=self.not_user_sub_matrix
        )
        self.support = [self.Graph]

        self.P_vec_size = int((self.num_all * self.num_all - self.num_all) / 2)  # + self.num_all
        if self.edge_additions:
            self.P_idxs = np.stack((self.interaction_matrix == 0).nonzero())
            self.P_idxs[1] += self.n_users
            self.P_idxs = torch.tensor(self.P_idxs, dtype=int, device=self.device)
            self.P_symm = nn.Parameter(torch.FloatTensor(torch.zeros(self.P_vec_size)) - 5)  # to get sigmoid closer to 0

            self.mask_sub_adj = torch.zeros((self.num_all, self.num_all), dtype=torch.bool).to(self.device)
            self.mask_sub_adj[self.P_idxs[0], self.P_idxs[1]] = True
        else:
            self.P_symm = nn.Parameter(torch.FloatTensor(torch.ones(self.P_vec_size)))
            self.P_idxs = None

            self.mask_sub_adj = torch.zeros((self.num_all, self.num_all), dtype=torch.bool).to(self.device)
            self.mask_sub_adj[tuple(self.sub_Graph)] = True

        # self.reset_parameters()

        # accumulation operation
        self.accum = config['accum']
        if self.accum == 'stack':
            div = self.gcn_output_dim // len(self.support)
            if self.gcn_output_dim % len(self.support) != 0:
                self.logger.warning(
                    "HIDDEN[0] (=%d) of stack layer is adjusted to %d (in %d splits)." %
                    (self.gcn_output_dim, len(self.support) * div, len(self.support))
                )
            self.gcn_output_dim = len(self.support) * div

        # define layers and loss
        self.GcEncoder = GcEncoder(
            accum=self.accum,
            num_user=self.n_users,
            num_item=self.n_items,
            support=self.support,
            input_dim=self.input_dim,
            gcn_output_dim=self.gcn_output_dim,
            dense_output_dim=self.dense_output_dim,
            drop_prob=self.dropout_prob,
            device=self.device,
            perturb_adj=self.P_symm,
            edge_additions=self.edge_additions,
            mask_sub_adj=self.mask_sub_adj,
            only_subgraph=self.only_subgraph,
            sparse_feature=self.sparse_feature,
        ).to(self.device)
        self.BiDecoder = BiDecoder(
            input_dim=self.dense_output_dim,
            output_dim=self.n_class,
            drop_prob=0.,
            device=self.device,
            num_weights=self.num_basis_functions
        ).to(self.device)
        # self.loss_function = nn.CrossEntropyLoss()
        self.loss_function = utils.NDCGApproxLoss()

    def reset_parameters(self, eps=1e-4):
        with torch.no_grad():
            if not self.edge_additions:
                self.P_symm.sub_(eps)

    def get_adj_matrix(self, user_id, n_hops, neighbors_hops=False, only_last_level=False, not_user_sub_matrix=False):
        A = sp.dok_matrix((self.num_all, self.num_all), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        A = A.tocoo()
        row = A.row
        col = A.col
        i = torch.LongTensor(np.stack([row, col], axis=0))
        data = torch.FloatTensor(A.data)
        adj = torch.sparse.FloatTensor(i, data, torch.Size(A.shape))
        if len(user_id) == 1:
            edge_subset = utils.get_neighbourhood(
                user_id[0].item(),
                i,
                n_hops,
                neighbors_hops=neighbors_hops,
                only_last_level=only_last_level,
                not_user_sub_matrix=not_user_sub_matrix,
                max_num_nodes=self.num_all
            )
        else:
            edge_subset = [torch.LongTensor(i)]
        return adj.to_dense().to(self.device), edge_subset[0].to(self.device)

    def forward(self, user_X, item_X, user, item, pred=False):
        # Graph autoencoders are comprised of a graph encoder model and a pairwise decoder model.
        user_embedding, item_embedding = self.GcEncoder(user_X, item_X, pred=pred)
        predict_score = self.BiDecoder(user_embedding, item_embedding, user, item)
        return predict_score

    # def calculate_loss(self, interaction):
    #     user = interaction[self.USER_ID]
    #     pos_item = interaction[self.ITEM_ID]
    #     neg_item = interaction[self.NEG_ITEM_ID]
    #
    #     users = torch.cat((user, user))
    #     items = torch.cat((pos_item, neg_item))
    #
    #     user_X, item_X = self.user_features, self.item_features
    #     predict = self.forward(user_X, item_X, users, items, pred=False)
    #     target = torch.zeros(len(user) * 2, dtype=torch.long).to(self.device)
    #     target[:len(user)] = 1
    #
    #     loss = self.loss_function(predict, target)
    #     return loss

    def loss(self, output, fair_loss_f, fair_loss_target):
        """

        :param output: output of the model with perturbed adj matrix
        :param fair_loss_f: fair loss function
        :param fair_loss_target: fair loss target

        :return:
        """
        adj = self.support[0]

        # non-differentiable adj matrix is taken to compute the graph dist loss
        cf_adj = self.GcEncoder.P_loss
        cf_adj.requires_grad = True  # Need to change this otherwise loss_graph_dist has no gradient

        # compute fairness loss
        fair_loss = fair_loss_f(output, fair_loss_target)

        # compute normalized graph dist loss (logistic sigmoid is not used because reaches too fast 1)
        orig_loss_graph_dist = (cf_adj - adj).abs().sum() / 2  # Number of edges changed (symmetrical)
        loss_graph_dist = orig_loss_graph_dist / (1 + abs(orig_loss_graph_dist))  # sigmoid dist

        loss_total = fair_loss + 0.01 * loss_graph_dist

        return loss_total, orig_loss_graph_dist, loss_graph_dist, fair_loss, cf_adj, adj

    def predict(self, interaction, pred=False):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_X, item_X = self.user_features, self.item_features
        predict = self.forward(user_X, item_X, user, item, pred=pred)

        score = predict[:, 1]
        return score

    def full_sort_predict(self, interaction, pred=False):
        user = interaction[self.USER_ID]

        user_X, item_X = self.user_features, self.item_features
        predict = self.forward(user_X, item_X, user, None, pred=pred)

        score = predict[:, 1]
        return score


class GcEncoder(nn.Module):
    r"""Graph Convolutional Encoder
    GcEncoder take as input an :math:`N \times D` feature matrix :math:`X` and a graph adjacency matrix :math:`A`,
    and produce an :math:`N \times E` node embedding matrix;
    Note that :math:`N` denotes the number of nodes, :math:`D` the number of input features,
    and :math:`E` the embedding size.
    """

    def __init__(
        self,
        accum,
        num_user,
        num_item,
        support,
        input_dim,
        gcn_output_dim,
        dense_output_dim,
        drop_prob,
        device,
        perturb_adj,
        edge_additions,
        mask_sub_adj,
        only_subgraph,
        sparse_feature=True,
        act_dense=lambda x: x,
        share_user_item_weights=True,
        bias=False
    ):
        super(GcEncoder, self).__init__()
        self.num_users = num_user
        self.num_items = num_item
        self.input_dim = input_dim
        self.gcn_output_dim = gcn_output_dim
        self.dense_output_dim = dense_output_dim
        self.accum = accum
        self.sparse_feature = sparse_feature

        self.device = device
        self.dropout_prob = drop_prob
        self.dropout = nn.Dropout(p=self.dropout_prob)
        if self.sparse_feature:
            self.sparse_dropout = SparseDropout(p=self.dropout_prob)
        else:
            self.sparse_dropout = nn.Dropout(p=self.dropout_prob)

        self.dense_activate = act_dense
        self.activate = nn.ReLU()
        self.share_weights = share_user_item_weights
        self.bias = bias

        self.support = support
        self.num_support = len(support)
        self.num_all = self.num_users + self.num_items

        self.P_symm = perturb_adj
        self.edge_additions = edge_additions
        self.P_hat_symm, self.P = None, None

        self.mask_sub_adj = mask_sub_adj

        self.D_indices = torch.arange(self.num_all).tile((2, 1)).to(self.device)
        self.only_subgraph = only_subgraph

        # gcn layer
        if self.accum == 'sum':
            self.weights_u = nn.ParameterList([
                nn.Parameter(
                    torch.FloatTensor(self.input_dim, self.gcn_output_dim).to(self.device), requires_grad=True
                ) for _ in range(self.num_support)
            ])
            if share_user_item_weights:
                self.weights_v = self.weights_u
            else:
                self.weights_v = nn.ParameterList([
                    nn.Parameter(
                        torch.FloatTensor(self.input_dim, self.gcn_output_dim).to(self.device), requires_grad=True
                    ) for _ in range(self.num_support)
                ])
        else:
            assert self.gcn_output_dim % self.num_support == 0, 'output_dim must be multiple of num_support for stackGC'
            self.sub_hidden_dim = self.gcn_output_dim // self.num_support

            self.weights_u = nn.ParameterList([
                nn.Parameter(
                    torch.FloatTensor(self.input_dim, self.sub_hidden_dim).to(self.device), requires_grad=True
                ) for _ in range(self.num_support)
            ])
            if share_user_item_weights:
                self.weights_v = self.weights_u
            else:
                self.weights_v = nn.ParameterList([
                    nn.Parameter(
                        torch.FloatTensor(self.input_dim, self.sub_hidden_dim).to(self.device), requires_grad=True
                    ) for _ in range(self.num_support)
                ])

        # dense layer
        self.dense_layer_u = nn.Linear(self.gcn_output_dim, self.dense_output_dim, bias=self.bias)
        if share_user_item_weights:
            self.dense_layer_v = self.dense_layer_u
        else:
            self.dense_layer_v = nn.Linear(self.gcn_output_dim, self.dense_output_dim, bias=self.bias)

        self._init_weights()

    def _init_weights(self):
        init_range = math.sqrt((self.num_support + 1) / (self.input_dim + self.gcn_output_dim))
        for w in range(self.num_support):
            self.weights_u[w].data.uniform_(-init_range, init_range)
        if not self.share_weights:
            for w in range(self.num_support):
                self.weights_v[w].data.uniform_(-init_range, init_range)

        dense_init_range = math.sqrt((self.num_support + 1) / (self.dense_output_dim + self.gcn_output_dim))
        self.dense_layer_u.weight.data.uniform_(-dense_init_range, dense_init_range)
        if not self.share_weights:
            self.dense_layer_v.weight.data.uniform_(-dense_init_range, dense_init_range)

        if self.bias:
            self.dense_layer_u.bias.data.fill_(0)
            if not self.share_weights:
                self.dense_layer_v.bias.data.fill_(0)

    # @profile
    def perturbate_adj_matrix(self, i, pred=False):
        graph_A = self.support[i]
        num_all = self.num_all

        if pred:
            P_hat_symm = utils.create_symm_matrix_from_vec(self.P_symm, self.num_all)
            P = (torch.sigmoid(P_hat_symm) >= 0.5).float()
            if self.edge_additions:
                self.P_loss = torch.where(self.mask_sub_adj, P, graph_A)
            else:
                self.P_loss = P * graph_A
        else:
            P_hat_symm = utils.create_symm_matrix_from_vec(self.P_symm, self.num_all)

            P = torch.sigmoid(P_hat_symm)

        if not self.only_subgraph:
            if self.edge_additions:
                A_tilde = torch.where(self.mask_sub_adj, P, graph_A)
            else:
                A_tilde = torch.where(self.mask_sub_adj, P * graph_A, graph_A)
        else:
            A_tilde = self.mask_sub_adj.float()

        # Don't need gradient of this if pred is False
        D_tilde = A_tilde.sum(dim=1) if pred else A_tilde.sum(dim=1).detach()
        D_tilde_exp = (D_tilde + 1e-7).pow(-0.5)

        D_tilde_exp = torch.sparse.FloatTensor(self.D_indices, D_tilde_exp, torch.Size((num_all, num_all)))

        # # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
        return torch.mm(torch.sparse.mm(D_tilde_exp, A_tilde), D_tilde_exp.to_dense()).to_sparse()

    def forward(self, user_X, item_X, pred=False):
        # ----------------------------------------GCN layer----------------------------------------

        user_X = self.sparse_dropout(user_X)
        item_X = self.sparse_dropout(item_X)

        embeddings = []
        if self.accum == 'sum':
            wu = 0.
            wv = 0.
            for i in range(self.num_support):
                # weight sharing
                wu = self.weights_u[i] + wu
                wv = self.weights_v[i] + wv

                # multiply feature matrices with weights
                if self.sparse_feature:
                    temp_u = torch.sparse.mm(user_X, wu)
                    temp_v = torch.sparse.mm(item_X, wv)
                else:
                    temp_u = torch.mm(user_X, wu)
                    temp_v = torch.mm(item_X, wv)
                all_embedding = torch.cat([temp_u, temp_v])

                norm_adj = self.perturbate_adj_matrix(i, pred=pred)
                all_emb = torch.sparse.mm(norm_adj, all_embedding)
                embeddings.append(all_emb)

            embeddings = torch.stack(embeddings, dim=1)
            embeddings = torch.sum(embeddings, dim=1)
        else:
            for i in range(self.num_support):
                # multiply feature matrices with weights
                if self.sparse_feature:
                    temp_u = torch.sparse.mm(user_X, self.weights_u[i])
                    temp_v = torch.sparse.mm(item_X, self.weights_v[i])
                else:
                    temp_u = torch.mm(user_X, self.weights_u[i])
                    temp_v = torch.mm(item_X, self.weights_v[i])
                all_embedding = torch.cat([temp_u, temp_v])

                # then multiply with adj matrices
                norm_adj = self.perturbate_adj_matrix(i, pred=pred)
                all_emb = torch.sparse.mm(norm_adj, all_embedding)
                embeddings.append(all_emb)

            embeddings = torch.cat(embeddings, dim=1)

        users, items = torch.split(embeddings, [self.num_users, self.num_items])

        u_hidden = self.activate(users)
        v_hidden = self.activate(items)

        # ----------------------------------------Dense Layer----------------------------------------

        u_hidden = self.dropout(u_hidden)
        v_hidden = self.dropout(v_hidden)

        u_hidden = self.dense_layer_u(u_hidden)
        v_hidden = self.dense_layer_u(v_hidden)

        u_outputs = self.dense_activate(u_hidden)
        v_outputs = self.dense_activate(v_hidden)

        return u_outputs, v_outputs


class BiDecoder(nn.Module):
    """Bi-linear decoder
    BiDecoder takes pairs of node embeddings and predicts respective entries in the adjacency matrix.
    """

    def __init__(self, input_dim, output_dim, drop_prob, device, num_weights=3, act=lambda x: x):
        super(BiDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_weights = num_weights
        self.device = device

        self.activate = act
        self.dropout_prob = drop_prob
        self.dropout = nn.Dropout(p=self.dropout_prob)

        self.weights = nn.ParameterList([
            nn.Parameter(orthogonal([self.input_dim, self.input_dim]).to(self.device)) for _ in range(self.num_weights)
        ])
        self.dense_layer = nn.Linear(self.num_weights, self.output_dim, bias=False)
        self._init_weights()

    def _init_weights(self):
        dense_init_range = math.sqrt(self.output_dim / (self.num_weights + self.output_dim))
        self.dense_layer.weight.data.uniform_(-dense_init_range, dense_init_range)

    def forward(self, u_inputs, i_inputs, users, items=None):
        u_inputs = self.dropout(u_inputs)
        i_inputs = self.dropout(i_inputs)

        if items is not None:
            users_emb = u_inputs[users]
            items_emb = i_inputs[items]

            basis_outputs = []
            for i in range(self.num_weights):
                users_emb_temp = torch.mm(users_emb, self.weights[i])
                scores = torch.mul(users_emb_temp, items_emb)
                scores = torch.sum(scores, dim=1)
                basis_outputs.append(scores)
        else:
            users_emb = u_inputs[users]
            items_emb = i_inputs

            basis_outputs = []
            for i in range(self.num_weights):
                users_emb_temp = torch.mm(users_emb, self.weights[i])
                scores = torch.mm(users_emb_temp, items_emb.transpose(0, 1))
                basis_outputs.append(scores.view(-1))

        basis_outputs = torch.stack(basis_outputs, dim=1)
        basis_outputs = self.dense_layer(basis_outputs)
        output = self.activate(basis_outputs)

        return output


def orthogonal(shape, scale=1.1):
    """
    Initialization function for weights in class GCMC.
    From Lasagne. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    """
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)

    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return torch.tensor(scale * q[:shape[0], :shape[1]], dtype=torch.float32)
