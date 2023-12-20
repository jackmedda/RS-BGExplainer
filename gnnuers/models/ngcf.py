import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.general_recommender import NGCF as Recbole_NGCF


class NGCF(Recbole_NGCF):
    def __init__(self, config, dataset):
        super(NGCF, self).__init__(config, dataset)
        # NOT IN ORIGINAL CODE, ADDED TO PREVENT DROPOUT LAYER TO MODIFY VALUES EVEN IN EVAL MODE
        self.emb_dropout = nn.Dropout(self.message_dropout)

    def forward(self):

        A_hat = self.sparse_dropout(self.norm_adj_matrix) if self.node_dropout != 0 else self.norm_adj_matrix
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        for gnn in self.GNNlayers:
            all_embeddings = gnn(A_hat, self.eye_matrix, all_embeddings)
            all_embeddings = nn.LeakyReLU(negative_slope=0.2)(all_embeddings)
            all_embeddings = self.emb_dropout(all_embeddings)
            all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
            embeddings_list += [all_embeddings]  # storage output embedding of each layer
        ngcf_all_embeddings = torch.cat(embeddings_list, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(ngcf_all_embeddings, [self.n_users, self.n_items])

        return user_all_embeddings, item_all_embeddings
