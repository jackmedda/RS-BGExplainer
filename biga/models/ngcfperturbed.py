# -*- coding: utf-8 -*-

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.general_recommender import NGCF

sys.path.append('..')

from biga.models import PerturbedModel


class NGCFPerturbed(PerturbedModel, NGCF):
    def __init__(self, config, dataset, **kwargs):
        NGCF.__init__(self, config, dataset)
        PerturbedModel.__init__(self, config, **kwargs)

        self._drop_layer = nn.Dropout(self.message_dropout)

    def forward(self, pred=False):
        A_hat = self.perturbate_adj_matrix(self.Graph, pred=pred)
        if self.node_dropout != 0:
            A_hat = self.sparse_dropout(A_hat)

        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        for gnn in self.GNNlayers:
            all_embeddings = gnn(A_hat, self.eye_matrix, all_embeddings)
            all_embeddings = nn.LeakyReLU(negative_slope=0.2)(all_embeddings)
            all_embeddings = self._drop_layer(all_embeddings)
            all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
            embeddings_list += [all_embeddings]  # storage output embedding of each layer
        ngcf_all_embeddings = torch.cat(embeddings_list, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(ngcf_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings
