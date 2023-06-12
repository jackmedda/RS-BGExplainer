# -*- coding: utf-8 -*-

import sys

import torch
from recbole.model.general_recommender import LightGCN

sys.path.append('..')

from biga.models import PerturbedModel


class LightGCNPerturbed(PerturbedModel, LightGCN):
    def __init__(self, config, dataset, **kwargs):
        LightGCN.__init__(self, config, dataset)
        PerturbedModel.__init__(self, config, **kwargs)

    def forward(self, pred=False):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        adj_matrix = self.perturbate_adj_matrix(self.Graph, pred=pred)
        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])

        return user_all_embeddings, item_all_embeddings
