import torch
import numpy as np
import torch.nn as nn

from . import utils as model_utils


class PerturbedModel(object):
    RANDOM_POLICY = 'RANDOM'

    # missing attributes are obtained by multiple inheritance
    def __init__(self, config, adv_group=None, filtered_users=None):
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
        self.random_perturb_p = config['random_perturbation_p'] or 0.05
        self.initialization = config['perturbation_initialization']
        self.P_symm = None
        self.mask_sub_adj = None
        self.force_removed_edges = None
        self._P_loss = None
        self.D_indices = None

        self.adv_group = adv_group
        self.filtered_users = filtered_users
        if self.filtered_users is not None:
            if isinstance(self.filtered_users, str):
                if self.filtered_users != self.RANDOM_POLICY:
                    raise AttributeError(f'filtered_users can be a tensor of user ids or `{self.RANDOM_POLICY}`')
            else:
                self.filtered_users = self.filtered_users.to(self.device)
        self.mask_filter = None

        self.Graph, self.sub_Graph = model_utils.get_adj_matrix(
            self.interaction_matrix,
            self.num_all,
            self.n_users
        )

        self.Graph, self.sub_Graph = self.Graph.to(self.device), self.sub_Graph.to(self.device)

        self.force_removed_edges = None
        if self.edge_additions:
            self.mask_sub_adj = np.stack((self.interaction_matrix == 0).nonzero())
            self.mask_sub_adj = self.mask_sub_adj[:, self.mask_sub_adj[0] != self.mask_sub_adj[1] & (self.mask_sub_adj[0] != 0)]
            self.mask_sub_adj[1] += self.n_users
            self.mask_sub_adj = torch.tensor(self.mask_sub_adj, dtype=int, device=self.device)

            if self.filtered_users is not None:
                try:
                    self.mask_sub_adj = torch.isin(self.mask_sub_adj[0], self.filtered_users).nonzero()[:, 0]
                except AttributeError:
                    self.mask_sub_adj = self.mask_sub_adj[
                        (self.mask_sub_adj[0][:, None] == self.filtered_users).nonzero()[:, 0]
                    ]

            if self.initialization != 'random':
                P_symm_init = -5  # to get sigmoid closer to 0
                P_symm_func = "zeros"
            else:
                P_symm_init = -6
                P_symm_func = "rand"
            P_symm_size = self.mask_sub_adj.shape[1]
        else:
            self.mask_sub_adj = self.sub_Graph
            self.mask_filter = torch.ones(self.mask_sub_adj.shape[1], dtype=torch.bool, device=self.device)

            if self.filtered_users is not None and self.filtered_users != self.RANDOM_POLICY:
                try:
                    user_filter = torch.isin(self.mask_sub_adj, self.filtered_users).any(dim=0)
                except AttributeError:
                    user_filter = (self.mask_sub_adj[0][:, None] == self.filtered_users).any(-1) | \
                                  (self.mask_sub_adj[1][:, None] == self.filtered_users).any(-1)
                self.mask_filter &= user_filter

            if self.initialization != 'random':
                P_symm_init = 0
                P_symm_func = "ones"
            else:
                P_symm_init = 1
                P_symm_func = "rand"
            P_symm_size = self.mask_filter.nonzero().shape[0] // 2

            if config['explainer_policies']['force_removed_edges']:
                self.force_removed_edges = torch.FloatTensor(torch.ones(P_symm_size)).to(self.device)

        self.P_symm = nn.Parameter(torch.FloatTensor(getattr(torch, P_symm_func)(P_symm_size)) + P_symm_init)

        self._P_loss = None
        self.D_indices = torch.arange(self.num_all).tile((2, 1)).to(self.device)

    @property
    def P_loss(self):
        return self._P_loss

    @P_loss.setter
    def P_loss(self, value):
        self._P_loss = value

    def loss(self, output, fair_loss_f, fair_loss_target):
        """

        :param output: output of the model with perturbed adj matrix
        :param fair_loss_f: fair loss function
        :param fair_loss_target: fair loss target

        :return:
        """
        # compute fairness loss
        fair_loss = fair_loss_f(output, fair_loss_target)

        adj = self.Graph

        # non-differentiable adj matrix is taken to compute the graph dist loss
        cf_adj = self.P_loss
        cf_adj.requires_grad = True  # Need to change this otherwise loss_graph_dist has no gradient

        orig_dist = (cf_adj - adj).coalesce()

        # compute normalized graph dist loss (logistic sigmoid is not used because reaches too fast 1)
        orig_loss_graph_dist = torch.sum(orig_dist.values().abs()) / 2  # Number of edges changed (symmetrical)
        loss_graph_dist = orig_loss_graph_dist / (1 + abs(orig_loss_graph_dist))  # sigmoid dist

        loss_total = fair_loss + 0.01 * loss_graph_dist

        return loss_total, orig_loss_graph_dist, loss_graph_dist, fair_loss, orig_dist

    def perturbate_adj_matrix(self, Graph, pred=False):
        P_symm = self.P_symm
        if not self.edge_additions and self.force_removed_edges is not None:
            if self.filtered_users == self.RANDOM_POLICY:
                if not pred:
                    p = self.random_perturb_p
                    random_perb = torch.FloatTensor(
                        np.random.choice([0, 1], size=self.force_removed_edges.size(0), p=[p, 1 - p])
                    ).to(self.force_removed_edges.device)
                    self.force_removed_edges = self.force_removed_edges * random_perb
                # the minus 1 assigns (0 - 1) = -1 to the already removed edges, such that the sigmoid is < 0.5
                P_symm = self.force_removed_edges - 1
            else:
                self.force_removed_edges = (torch.sigmoid(self.P_symm.detach()) >= 0.5).float() * self.force_removed_edges
                # the minus 1 assigns (0 - 1) = -1 to the already removed edges, such that the sigmoid is < 0.5
                P_symm = torch.where(self.force_removed_edges == 0, self.force_removed_edges - 1, self.P_symm)

        perturb_matrix, P_loss = model_utils.perturb_adj_matrix(
            Graph,
            P_symm,
            self.mask_sub_adj,
            self.num_all,
            self.D_indices,
            pred=pred,
            edge_deletions=not self.edge_additions,
            mask_filter=self.mask_filter
        )
        if P_loss is not None:
            self.P_loss = P_loss

        return perturb_matrix

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

        user_e, item_e = self.forward(pred=pred)

        # get user embedding from storage variable
        u_embeddings = user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, item_e.transpose(0, 1))

        return scores.view(-1)
