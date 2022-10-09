# -*- coding: utf-8 -*-
# @Time   : 2020/7/16
# @Author : Zihan Lin
# @Email  : linzihan.super@foxmail.com

# UPDATE:
# @Time   : 2020/9/16
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

r"""
NGCF
################################################
Reference:
    Xiang Wang et al. "Neural Graph Collaborative Filtering." in SIGIR 2019.

Reference code:
    https://github.com/xiangwang1223/neural_graph_collaborative_filtering

"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.layers import BiGNNLayer, SparseDropout
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType

import src.utils as utils


class NGCFPerturbated(GeneralRecommender):
    r"""NGCF is a model that incorporate GNN for recommendation.
    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset, user_id):
        super(NGCFPerturbated, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size_list = config['hidden_size_list']
        self.hidden_size_list = [self.embedding_size] + self.hidden_size_list
        self.node_dropout = config['node_dropout']
        self.message_dropout = config['message_dropout']
        self.reg_weight = config['reg_weight']

        # define layers and loss
        self.sparse_dropout = SparseDropout(self.node_dropout)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.GNNlayers = torch.nn.ModuleList()
        for idx, (input_size, output_size) in enumerate(zip(self.hidden_size_list[:-1], self.hidden_size_list[1:])):
            self.GNNlayers.append(BiGNNLayer(input_size, output_size))

        # generate intermediate data
        # self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        self.eye_matrix = self.get_eye_mat().to(self.device)

        self.num_all = self.n_users + self.n_items

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

        if self.edge_additions:
            self.mask_sub_adj = np.stack((self.interaction_matrix == 0).nonzero())
            self.mask_sub_adj = self.mask_sub_adj[:, self.mask_sub_adj[0] != self.mask_sub_adj[1]]
            self.mask_sub_adj[1] += self.n_users
            self.mask_sub_adj = torch.tensor(self.mask_sub_adj, dtype=int, device=self.device)
            # to get sigmoid closer to 0
            self.P_symm = nn.Parameter(torch.FloatTensor(torch.zeros(self.mask_sub_adj.shape[1])) - 5)
        else:
            self.P_symm = nn.Parameter(torch.FloatTensor(torch.ones(self.sub_Graph.shape[1])))
            self.mask_sub_adj = self.sub_Graph

        self.P_loss = None
        self.D_indices = torch.arange(self.num_all).tile((2, 1)).to(self.device)

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

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

    def get_eye_mat(self):
        r"""Construct the identity matrix with the size of  n_items+n_users.

        Returns:
            Sparse tensor of the identity matrix. Shape of (n_items+n_users, n_items+n_users)
        """
        num = self.n_items + self.n_users  # number of column of the square matrix
        i = torch.LongTensor([range(0, num), range(0, num)])
        val = torch.FloatTensor([1] * num)  # identity matrix
        return torch.sparse.FloatTensor(i, val)

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of (n_items+n_users, embedding_dim)
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def perturbate_adj_matrix(self, pred=False):
        perturb_matrix, P_loss = utils.perturbate_adj_matrix(
            self.Graph,
            self.P_symm,
            self.mask_sub_adj,
            self.num_all,
            self.D_indices,
            pred=pred
        )
        if P_loss is not None:
            self.P_loss = P_loss

        return perturb_matrix

    def forward(self, pred=False):
        if self.node_dropout != 0:
            A_hat = self.sparse_dropout(self.perturbate_adj_matrix(pred=pred))
        else:
            A_hat = self.perturbate_adj_matrix(pred=pred)

        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        for gnn in self.GNNlayers:
            all_embeddings = gnn(A_hat, self.eye_matrix, all_embeddings)
            all_embeddings = nn.LeakyReLU(negative_slope=0.2)(all_embeddings)
            all_embeddings = nn.Dropout(self.message_dropout)(all_embeddings)
            all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
            embeddings_list += [all_embeddings]  # storage output embedding of each layer
        ngcf_all_embeddings = torch.cat(embeddings_list, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(ngcf_all_embeddings, [self.n_users, self.n_items])

        return user_all_embeddings, item_all_embeddings

    # def calculate_loss(self, interaction):
    #     # clear the storage variable when training
    #     if self.restore_user_e is not None or self.restore_item_e is not None:
    #         self.restore_user_e, self.restore_item_e = None, None
    #
    #     user = interaction[self.USER_ID]
    #     pos_item = interaction[self.ITEM_ID]
    #     neg_item = interaction[self.NEG_ITEM_ID]
    #
    #     user_all_embeddings, item_all_embeddings = self.forward()
    #     u_embeddings = user_all_embeddings[user]
    #     pos_embeddings = item_all_embeddings[pos_item]
    #     neg_embeddings = item_all_embeddings[neg_item]
    #
    #     pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
    #     neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
    #     mf_loss = self.mf_loss(pos_scores, neg_scores)  # calculate BPR Loss
    #
    #     reg_loss = self.reg_loss(u_embeddings, pos_embeddings, neg_embeddings)  # L2 regularization of embeddings
    #
    #     return mf_loss + self.reg_weight * reg_loss

    def loss(self, output, fair_loss_f, fair_loss_target):
        """

        :param output: output of the model with perturbed adj matrix
        :param fair_loss_f: fair loss function
        :param fair_loss_target: fair loss target

        :return:
        """
        adj = self.Graph

        # non-differentiable adj matrix is taken to compute the graph dist loss
        cf_adj = self.P_loss.to_dense()
        cf_adj.requires_grad = True  # Need to change this otherwise loss_graph_dist has no gradient

        # compute fairness loss
        fair_loss = fair_loss_f(output, fair_loss_target)

        # compute normalized graph dist loss (logistic sigmoid is not used because reaches too fast 1)
        orig_loss_graph_dist = torch.sum((cf_adj - adj).abs()) / 2  # Number of edges changed (symmetrical)
        loss_graph_dist = orig_loss_graph_dist / (1 + abs(orig_loss_graph_dist))  # sigmoid dist

        loss_total = fair_loss + 0.01 * loss_graph_dist

        return loss_total, orig_loss_graph_dist, loss_graph_dist, fair_loss, cf_adj, adj

    def predict(self, interaction, pred=False):

        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward(pred=pred)

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction, pred=False):
        user = interaction[self.USER_ID]
        # if self.restore_user_e is None or self.restore_item_e is None:
        #     self.restore_user_e, self.restore_item_e = self.forward(pred=pred)

        self.restore_user_e, self.restore_item_e = self.forward(pred=pred)

        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)
