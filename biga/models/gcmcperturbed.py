# -*- coding: utf-8 -*-

import sys

import torch
from recbole.model.general_recommender.gcmc import GCMC, GcEncoder

sys.path.append('..')

from biga.models import PerturbedModel


class GCMCPerturbed(PerturbedModel, GCMC):
    def __init__(self, config, dataset, **kwargs):
        GCMC.__init__(self, config, dataset)
        PerturbedModel.__init__(self, config, **kwargs)

        self.support[0] = self.Graph

        self.GcEncoder = GcEncoderPerturbated(
            accum=self.accum,
            num_user=self.n_users,
            num_item=self.n_items,
            support=self.support,
            input_dim=self.input_dim,
            gcn_output_dim=self.gcn_output_dim,
            dense_output_dim=self.dense_output_dim,
            drop_prob=self.dropout_prob,
            device=self.device,
            edge_additions=self.edge_additions,
            mask_sub_adj=self.mask_sub_adj,
            only_subgraph=self.only_subgraph,
            force_removed_edges=self.force_removed_edges,
            perturbate_adj_matrix_func=self.perturbate_adj_matrix,
            sparse_feature=self.sparse_feature
        ).to(self.device)

    def forward(self, user_X, item_X, user, item, pred=False):
        # Graph autoencoders are comprised of a graph encoder model and a pairwise decoder model.
        user_embedding, item_embedding = self.GcEncoder(user_X, item_X, pred=pred)
        predict_score = self.BiDecoder(user_embedding, item_embedding, user, item)
        return predict_score

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


class GcEncoderPerturbated(GcEncoder):

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
        edge_additions,
        mask_sub_adj,
        only_subgraph,
        force_removed_edges,
        perturbate_adj_matrix_func,
        sparse_feature=True,
        act_dense=lambda x: x,
        share_user_item_weights=True,
        bias=False
    ):
        super(GcEncoderPerturbated, self).__init__(
            accum=accum,
            num_user=num_user,
            num_item=num_item,
            support=support,
            input_dim=input_dim,
            gcn_output_dim=gcn_output_dim,
            dense_output_dim=dense_output_dim,
            drop_prob=drop_prob,
            device=device,
            sparse_feature=sparse_feature,
            act_dense=act_dense,
            share_user_item_weights=share_user_item_weights,
            bias=bias
        )
        self.num_all = self.num_users + self.num_items

        self.edge_additions = edge_additions
        self.P_hat_symm, self.P = None, None

        self.mask_sub_adj = mask_sub_adj

        self.D_indices = torch.arange(self.num_all).tile((2, 1)).to(self.device)
        self.only_subgraph = only_subgraph

        self.force_removed_edges = force_removed_edges

        self.perturbate_adj_matrix = perturbate_adj_matrix_func

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

                norm_adj = self.perturbate_adj_matrix(self.support[i], pred=pred)
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
                norm_adj = self.perturbate_adj_matrix(self.support[i], pred=pred)
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
