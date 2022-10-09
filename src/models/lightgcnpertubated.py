# -*- coding: utf-8 -*-
# @Time   : 2020/8/31
# @Author : Changxin Tian
# @Email  : cx.tian@outlook.com

# UPDATE:
# @Time   : 2020/9/16, 2021/12/22
# @Author : Shanlei Mu, Gaowei Zhang
# @Email  : slmu@ruc.edu.cn, 1462034631@qq.com

r"""
LightGCN
################################################

Reference:
    Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." in SIGIR 2020.

Reference code:
    https://github.com/kuandeng/LightGCN
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType

import src.utils as utils


class LightGCNPerturbated(GeneralRecommender):
    r"""LightGCN is a GCN-based recommender model.

    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.

    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset, user_id):
        super(LightGCNPerturbated, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        # load parameters info
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers = config['n_layers']  # int type:the layer num of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization
        self.require_pow = config['require_pow']

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        # self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

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

        # parameters initialization
        self.apply(xavier_uniform_initialization)
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

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
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
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        adj_matrix = self.perturbate_adj_matrix(pred=pred)
        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
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
    #     # calculate BPR Loss
    #     pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
    #     neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
    #     mf_loss = self.mf_loss(pos_scores, neg_scores)
    #
    #     # calculate BPR Loss
    #     u_ego_embeddings = self.user_embedding(user)
    #     pos_ego_embeddings = self.item_embedding(pos_item)
    #     neg_ego_embeddings = self.item_embedding(neg_item)
    #
    #     reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings, require_pow=self.require_pow)
    #
    #     loss = mf_loss + self.reg_weight * reg_loss
    #
    #     return loss

    def loss(self, output, fair_loss_f, fair_loss_target):
        """

        :param output: output of the model with perturbed adj matrix
        :param fair_loss_f: fair loss function
        :param fair_loss_target: fair loss target

        :return:
        """
        adj = self.Graph

        # non-differentiable adj matrix is taken to compute the graph dist loss
        cf_adj = self.P_loss
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
