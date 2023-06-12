import torch
import numpy as np
import torch.nn as nn

import biga.utils as utils
from . import utils as model_utils


class PerturbedModel(object):
    RANDOM_POLICY = 'RANDOM'

    # missing attributes are obtained by multiple inheritance
    def __init__(self, config, adv_group=None, filtered_users=None, filtered_items=None):
        self.num_all = self.n_users + self.n_items

        self.n_hops = config['n_hops']
        self.neighbors_hops = config['neighbors_hops']
        self.beta = config['cf_beta']
        self.sub_matrix_only_last_level = config['sub_matrix_only_last_level']
        self.not_user_sub_matrix = config['not_user_sub_matrix']
        self.only_subgraph = config['only_subgraph']
        self.not_pred = config['not_pred']
        self.pred_same = config['pred_same']

        self.edge_additions = config['edge_additions']
        self.random_perturb_p = (config['random_perturbation_p'] or 0.05) if filtered_users is not None else None
        self.random_perturb_rs = np.random.RandomState(config['seed'] or 0)
        self.initialization = config['perturbation_initialization']
        self.P_symm = None
        self.mask_sub_adj = None
        self.force_removed_edges = None
        self._P_loss = None
        self.D_indices = None

        self.adv_group = adv_group
        if filtered_users is not None:
            if isinstance(filtered_users, str):
                if filtered_users != self.RANDOM_POLICY:
                    raise AttributeError(f'filtered_users can be a tensor of user ids or `{self.RANDOM_POLICY}`')
        self.mask_filter = None
        self.mask_neighborhood = None

        self.Graph, self.sub_Graph = model_utils.get_adj_matrix(
            self.interaction_matrix,
            self.num_all,
            self.n_users
        )
        self.Graph, self.sub_Graph = self.Graph.to(self.device), self.sub_Graph.to('cpu')

        self.force_removed_edges = None
        if self.edge_additions:
            self.mask_sub_adj = np.stack((self.interaction_matrix == 0).nonzero())
            self.mask_sub_adj = self.mask_sub_adj[
                :, self.mask_sub_adj[0] != self.mask_sub_adj[1] & (self.mask_sub_adj[0] != 0)
            ]
            self.mask_sub_adj[1] += self.n_users
            self.mask_sub_adj = torch.tensor(self.mask_sub_adj, dtype=int, device='cpu')

            if filtered_users is not None:
                filtered_nodes_mask = model_utils.edges_filter_nodes(self.mask_sub_adj[[0]], filtered_users)
                if filtered_items is not None:
                    filtered_nodes_mask &= model_utils.edges_filter_nodes(
                        self.mask_sub_adj[[1]], filtered_items + self.n_users
                    )

                self.mask_sub_adj = self.mask_sub_adj[:, filtered_nodes_mask]
        else:
            self.mask_sub_adj = self.sub_Graph
            self.mask_filter = torch.ones(self.mask_sub_adj.shape[1], dtype=torch.bool, device='cpu')

            if filtered_users is not None and filtered_users != self.RANDOM_POLICY:
                self.mask_filter &= model_utils.edges_filter_nodes(self.mask_sub_adj, filtered_users)
            if filtered_items is not None:
                self.mask_filter &= model_utils.edges_filter_nodes(self.mask_sub_adj, filtered_items + self.n_users)

        self._initialize_P_symm()

        if not self.edge_additions:
            if config['explainer_policies']['force_removed_edges']:
                self.force_removed_edges = torch.FloatTensor(torch.ones(self.P_symm_size)).to('cpu')
            if config['explainer_policies']['neighborhood_perturbation']:
                self.mask_neighborhood = self.mask_filter.clone().detach()

        self._P_loss = None
        self.D_indices = torch.arange(self.num_all).tile((2, 1)).to('cpu')

    def _initialize_P_symm(self):
        if self.edge_additions:
            if self.initialization != 'random':
                self.P_symm_init = -5  # to get sigmoid closer to 0
                self.P_symm_func = "zeros"
            else:
                self.P_symm_init = -6
                self.P_symm_func = "rand"
            self.P_symm_size = self.mask_sub_adj.shape[1]
        else:
            if self.initialization != 'random':
                self.P_symm_init = 0
                self.P_symm_func = "ones"
            else:
                self.P_symm_init = 1
                self.P_symm_func = "rand"
            self.P_symm_size = self.mask_filter.nonzero().shape[0] // 2

        self.P_symm = nn.Parameter(
            torch.FloatTensor(getattr(torch, self.P_symm_func)(self.P_symm_size)) + self.P_symm_init
        )

    def reset_param(self):
        with torch.no_grad():
            self._parameters['P_symm'].copy_(
                torch.FloatTensor(getattr(torch, self.P_symm_func)(self.P_symm_size)) + self.P_symm_init
            )
        if self.force_removed_edges is not None:
            self.force_removed_edges = torch.FloatTensor(torch.ones(self.P_symm_size)).to('cpu')
        if self.mask_neighborhood is not None:
            self.mask_neighborhood = self.mask_filter.clone().detach()

    @property
    def P_loss(self):
        return self._P_loss

    @P_loss.setter
    def P_loss(self, value):
        self._P_loss = value

    def cf_state_dict(self):
        return {
            'P_symm': self.state_dict()['P_symm'],
            'mask_sub_adj': self.mask_sub_adj.detach(),
            'mask_filter': self.mask_filter.detach() if self.mask_filter is not None else None,
            'force_removed_edges': self.force_removed_edges.detach() if self.force_removed_edges is not None else None,
            'mask_neighborhood': self.mask_neighborhood.detach() if self.mask_neighborhood is not None else None
        }

    def load_cf_state_dict(self, ckpt):
        state_dict = self.state_dict()
        state_dict.update({'P_symm': ckpt['P_symm']})
        self.load_state_dict(state_dict)
        self.mask_sub_adj = ckpt['mask_sub_adj']
        self.mask_filter = ckpt['mask_filter']
        self.force_removed_edges = ckpt['force_removed_edges']
        self.mask_neighborhood = ckpt['mask_neighborhood']

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

        loss_total = fair_loss + self.beta * loss_graph_dist

        return loss_total, orig_loss_graph_dist, loss_graph_dist, fair_loss, orig_dist

    def update_neighborhood(self, nodes: torch.Tensor):
        if self.mask_neighborhood is None:
            raise NotImplementedError(
                "neighborhood can be updated only on edge deletion and if the config parameter is set"
            )
        nodes = nodes.flatten().to(self.mask_sub_adj.device)
        nodes_filter = model_utils.edges_filter_nodes(self.mask_sub_adj[:, :self.mask_neighborhood.shape[0]], nodes)
        self._update_neighborhood(nodes_filter)

    def _update_neighborhood(self, nhood: torch.Tensor):
        if not nhood.dtype == torch.bool:
            raise TypeError(f"neighborhood update except a bool Tensor, got {nhood.dtype}")
        self.mask_neighborhood &= nhood

    def _update_P_symm_on_neighborhood(self, P_symm):
        dev = P_symm.device
        if (self.mask_filter != self.mask_neighborhood).any():
            filtered_idxs_asymm = self.mask_filter.nonzero().T.squeeze()[:P_symm.shape[0]]
            P_symm_nhood_mask = self.mask_neighborhood[filtered_idxs_asymm]
            return torch.where(P_symm_nhood_mask.to(dev), P_symm, torch.ones_like(P_symm, device=dev))

    def perturbate_adj_matrix(self, Graph, pred=False):
        P_symm = self.P_symm
        dev = P_symm.device
        if not self.edge_additions:
            if self.force_removed_edges is not None:
                if self.random_perturb_p is not None:  # RANDOM_POLICY is active
                    if not pred:
                        p = self.random_perturb_p
                        random_perb = torch.FloatTensor(
                            (self.random_perturb_rs.rand(self.force_removed_edges.size(0)) > p).astype(int)
                        ).to(self.force_removed_edges.device)
                        self.force_removed_edges = self.force_removed_edges * random_perb
                    # the minus 1 assigns (0 - 1) = -1 to the already removed edges, such that the sigmoid is < 0.5
                    P_symm = self.force_removed_edges.to(dev) - 1
                else:
                    if self.mask_neighborhood is not None:
                        P_symm = self._update_P_symm_on_neighborhood(P_symm)

                    force_removed_edges = self.force_removed_edges.to(dev)
                    force_removed_edges = (torch.sigmoid(P_symm.detach()) >= 0.5).float() * force_removed_edges
                    # the minus 1 assigns (0 - 1) = -1 to the already removed edges, such that the sigmoid is < 0.5
                    P_symm = torch.where(force_removed_edges == 0, force_removed_edges - 1, P_symm)
                    self.force_removed_edges = force_removed_edges.to('cpu')
                    del force_removed_edges

            elif self.mask_neighborhood is not None:
                P_symm = self._update_P_symm_on_neighborhood(P_symm)

        if pred:
            P_hat_symm = (torch.sigmoid(P_symm) >= 0.5).float()
        else:
            P_hat_symm = torch.sigmoid(P_symm)

        kws = dict(
            edge_deletions=not self.edge_additions,
            mask_filter = self.mask_filter.to(dev) if self.mask_filter is not None else None
        )
        P = utils.create_sparse_symm_matrix_from_vec(P_hat_symm, self.mask_sub_adj.to(dev), Graph, **kws)
        if pred:
            self.P_loss = P

        # Don't need gradient of this if pred is False
        D_tilde = torch.sparse.sum(P, dim=1) if pred else torch.sparse.sum(P, dim=1).detach()
        D_tilde_exp = (D_tilde.to_dense() + 1e-7).pow(-0.5)

        D_tilde_exp = torch.sparse.FloatTensor(
            self.D_indices.to(dev), D_tilde_exp, torch.Size((self.num_all, self.num_all))
        )

        # # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
        return torch.sparse.mm(torch.sparse.mm(D_tilde_exp, P), D_tilde_exp)

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
