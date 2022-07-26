# Based on https://github.com/RexYing/gnn-model-explainer/blob/master/explainer/explain.py

import sys
from typing import Iterable

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from recbole.data.interaction import Interaction
from torchviz import make_dot

sys.path.append('..')

import src.utils as utils
from src.models import GCMCPerturbated

import subprocess
import os

from memory_profiler import profile


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    # memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_info


class BGExplainer:

    def __init__(self, config, dataset, model, user_id, dist="damerau_levenshtein", **kwargs):
        super(BGExplainer, self).__init__()
        self.model = model
        self.model.eval()
        self.dataset = dataset

        self.user_id = user_id
        self.beta = config['cf_beta']
        self.device = config['device']
        self.only_subgraph = config['only_subgraph']
        self.force_return = config['explainer_force_return']
        self.keep_history = config['keep_history_if_possible']
        self.explain_fairness_NDCGApprox = config['explain_fairness_NDCGApprox']
        self.unique_graph_dist_loss = config['save_unique_graph_dist_loss']

        self.tot_item_num = dataset.item_num
        self.item_tensor = dataset.get_item_feature().to(model.device)
        self.test_batch_size = self.tot_item_num

        # Instantiate CF model class, load weights from original model
        self.cf_model = GCMCPerturbated(config, dataset, self.user_id)

        self.cf_model.load_state_dict(self.model.state_dict(), strict=False)

        # Freeze weights from original model in cf_model
        for name, param in self.cf_model.named_parameters():
            if name != "P_symm":
                param.requires_grad = False
        # for name, param in self.model.named_parameters():
        #     print("orig model requires_grad: ", name, param.requires_grad)
        # for name, param in self.cf_model.named_parameters():
        #     print("cf model requires_grad: ", name, param.requires_grad)

        lr = config['cf_learning_rate']
        momentum = config["momentum"] or 0.0
        sgd_kwargs = {'momentum': momentum, 'nesterov': True if momentum > 0 else False}
        if config["cf_optimizer"] == "SGD":
            self.cf_optimizer = torch.optim.SGD(self.cf_model.parameters(), lr=lr, **sgd_kwargs)
        elif config["cf_optimizer"] == "Adadelta":
            self.cf_optimizer = torch.optim.Adadelta(self.cf_model.parameters(), lr=lr)
        elif config["cf_optimizer"] == "AdamW":
            self.cf_optimizer = torch.optim.AdamW(self.cf_model.parameters(), lr=lr)
        else:
            raise NotImplementedError("CF Optimizer not implemented")

        if dist == "set":
            self.dist = lambda topk_idx, cf_topk_idx: len(topk_idx) - (len(set(topk_idx) & set(cf_topk_idx)))
        elif dist == "damerau_levenshtein":
            self.dist = utils.damerau_levenshtein_distance

        self.train_bias_ratio = kwargs.get("train_bias_ratio", None)
        self.sensitive_attributes = config['sensitive_attributes']

        self.scores_args, self.topk_args = None, None
        self.model_scores, self.model_scores_topk, self.model_topk_idx = None, None, None

    def explain(self, batched_data, epochs, topk=10, loaded_scores=None, old_field2token_id=None):
        best_cf_example = []
        best_loss = np.inf
        first_fair_loss = None

        _, history_index, positive_u, positive_i = batched_data
        self.scores_args = [batched_data, self.tot_item_num, self.test_batch_size, self.item_tensor]

        self.topk_args = {'topk': topk, 'history_index': history_index if self.keep_history else None}

        if loaded_scores is not None:
            new_field2token_id = self.dataset.field2token_id[self.model.ITEM_ID_FIELD]
            scores_order = [new_field2token_id[k] for k in old_field2token_id]
            self.model_scores = torch.tensor(loaded_scores[self.user_id[0].item()])[scores_order].to(self.device)
        else:
            self.model_scores = self.get_scores(self.model, *self.scores_args, pred=None)

        # topk_idx contains the ids of the topk items
        self.model_scores_topk, self.model_topk_idx = self.get_top_k(self.model_scores, **self.topk_args)

        for epoch in range(epochs):
            new_example, loss_total, fair_loss = self.train(epoch, topk=topk)
            if epoch == 0:
                first_fair_loss = fair_loss
            if new_example is not None and abs(loss_total) < best_loss:
                if self.unique_graph_dist_loss and len(best_cf_example) > 0:
                    old_graph_dist = best_cf_example[-1][-5]
                    new_graph_dist = new_example[-4]
                    if not (old_graph_dist < new_graph_dist):
                        continue

                best_cf_example.append(new_example + [first_fair_loss])
                if not self.unique_graph_dist_loss:
                    best_loss = abs(loss_total)

        print("{} CF examples for user = {}".format(len(best_cf_example), self.user_id))

        return best_cf_example, self.model_scores.detach().cpu().numpy()

    # @profile
    def train(self, epoch, topk=10):
        t = time.time()

        self.cf_model.eval()

        with torch.no_grad():
            cf_scores_pred = self.get_scores(self.cf_model, *self.scores_args, pred=True)
            cf_scores_pred_topk, cf_topk_pred_idx = self.get_top_k(cf_scores_pred, **self.topk_args)

        self.cf_optimizer.zero_grad()
        self.cf_model.train()

        # cf_scores uses differentiable P_hat ==> adjacency matrix not binary, but needed for training
        cf_scores = self.get_scores(self.cf_model, *self.scores_args, pred=False)
        cf_scores_topk, cf_topk_idx = self.get_top_k(cf_scores, **self.topk_args)

        relevance_scores = torch.zeros_like(self.model_scores).float().to(self.device)
        relevance_scores[torch.arange(relevance_scores.shape[0])[:, None], self.model_topk_idx] = 1

        kwargs = {}
        if self.train_bias_ratio is not None:
            user_feat = self.dataset.user_feat
            user_id_mask = self.user_id.unsqueeze(-1) if self.user_id.dim() == 0 else self.user_id
            user_feat = {k: feat[user_id_mask] for k, feat in user_feat.interaction.items()}

            if self.explain_fairness_NDCGApprox:
                kwargs = {
                    "fair_loss_f": utils.NDCGApproxLoss(),
                    "target": get_bias_disparity_target_NDCGApprox(
                        cf_scores,
                        self.train_bias_ratio,
                        self.dataset.item_feat['class'],
                        self.sensitive_attributes,
                        user_feat,
                        topk=topk
                    )
                }
            else:
                kwargs = {
                    "fair_loss_f": BiasDisparityLoss(
                        self.train_bias_ratio,
                        self.dataset.item_feat['class'],
                        self.sensitive_attributes,
                        topk=topk
                    ),
                    "target": user_feat
                }

        loss_total, loss_pred, loss_graph_dist, fair_loss, cf_adj, adj, nnz_sub_adj = self.cf_model.loss(
            cf_scores,
            relevance_scores,
            self.model_topk_idx,
            cf_topk_pred_idx,
            self.dist,
            **kwargs
        )

        # dot = make_dot(loss_total.mean(), params=dict(self.cf_model.named_parameters()), show_attrs=True, show_saved=True)
        # dot.format = 'png'
        # dot.render(f'dots_loss')

        loss_total.backward()
        # for name, param in self.cf_model.named_parameters():
        #     if name == "P_symm":
        #         print(param.grad[param.grad.nonzero()])
        # input()
        nn.utils.clip_grad_norm_(self.cf_model.parameters(), 2.0)
        self.cf_optimizer.step()

        # del self.cf_model.GcEncoder.P
        # del self.cf_model.GcEncoder.P_hat_symm

        fair_loss = fair_loss.mean().item() if fair_loss is not None else torch.nan
        print(f"Explain duration: {time.strftime('%H:%M:%S', time.gmtime(time.time() - t))}",
              'User id: {}'.format(self.user_id),
              'Epoch: {}'.format(epoch + 1),
              'loss: {:.4f}'.format(loss_total.item()),
              'pred loss: {:.4f}'.format(loss_pred.mean().item()),
              ('FairNDCGApprox' if self.explain_fairness_NDCGApprox else 'FairBD') if self.train_bias_ratio is not None else '',
              'fair loss: {:.4f}'.format(fair_loss),
              'graph loss: {:.4f}'.format(loss_graph_dist.item()),
              'nnz sub adj: {}'.format(nnz_sub_adj))
        print('Orig output: {}\n'.format(self.model_scores),
              'Output: {}\n'.format(cf_scores),
              'Output nondiff: {}\n'.format(cf_scores_pred),
              'orig pred: {}, new pred: {}, new pred nondiff: {}'.format(self.model_topk_idx, cf_topk_idx, cf_topk_pred_idx))
        print(" ")

        cf_stats = None
        cf_check = [self.dist(_pred, _topk_idx) > 0 for _pred, _topk_idx in zip(cf_topk_pred_idx, self.model_topk_idx)]
        if any(cf_check) or self.force_return:
            cf_adj, adj = cf_adj.detach().cpu().numpy(), adj.detach().cpu().numpy()
            del_edges = np.sort(np.stack((adj != cf_adj).nonzero(), axis=0).T, axis=1)
            del_edges = pd.DataFrame(del_edges).drop_duplicates().values
            del_edges = del_edges[(del_edges[:, 0] != del_edges[:, 1]) &
                                  (del_edges[:, 0] != 0) &
                                  (del_edges[:, 1] != 0)]
            # del_edges = torch.stack((del_edges[:, 0], del_edges[:, 1]))
            del_edges = del_edges.T

            cf_stats = [self.user_id.detach().numpy(),
                        self.model_topk_idx.detach().cpu().numpy(), cf_topk_pred_idx.detach().cpu().numpy(),
                        [self.dist(_pred, _topk_idx) for _pred, _topk_idx in zip(cf_topk_pred_idx, self.model_topk_idx)],
                        loss_total.item(), loss_pred.mean().item(), loss_graph_dist.item(), fair_loss, del_edges, nnz_sub_adj]

        return cf_stats, loss_total.item(), fair_loss

    @staticmethod
    def _spilt_predict(model, interaction, batch_size, test_batch_size):
        spilt_interaction = dict()
        for key, tensor in interaction.interaction.items():
            spilt_interaction[key] = tensor.split(test_batch_size, dim=0)
        num_block = (batch_size + test_batch_size - 1) // test_batch_size
        result_list = []
        for i in range(num_block):
            current_interaction = dict()
            for key, spilt_tensor in spilt_interaction.items():
                current_interaction[key] = spilt_tensor[i]
            result = model.predict(Interaction(current_interaction).to(model.device))
            if len(result.shape) == 0:
                result = result.unsqueeze(0)
            result_list.append(result)
        return torch.cat(result_list, dim=0)

    def get_scores(self, _model, batched_data, tot_item_num, test_batch_size, item_tensor, pred=False):
        interaction, history_index, _, _ = batched_data
        # assert len(interaction.interaction[_model.USER_ID]) == 1
        try:
            # Note: interaction without item ids
            scores_kws = {'pred': pred} if pred is not None else {}
            scores = _model.full_sort_predict(interaction.to(_model.device), **scores_kws)

        except NotImplementedError:
            inter_len = len(interaction)
            new_inter = interaction.to(_model.device).repeat_interleave(tot_item_num)
            batch_size = len(new_inter)
            new_inter.update(item_tensor.repeat(inter_len))
            if batch_size <= test_batch_size:
                scores = _model.predict(new_inter)
            else:
                scores = self._spilt_predict(new_inter, batch_size, test_batch_size, test_batch_size)

        scores = scores.view(-1, tot_item_num)
        scores[:, 0] = -float("inf")
        if history_index is not None and not self.keep_history:
            scores[history_index] = -float("inf")

        return scores

    @staticmethod
    def get_top_k(scores_tensor, topk=10, history_index=None):
        while True:
            scores_top_k, topk_idx = torch.topk(scores_tensor, topk, dim=-1)  # n_users x k
            if history_index is None:
                break
            inters = np.intersect1d(topk_idx, history_index)
            if inters.shape[0] > 0:
                scores_tensor[inters] = -np.inf
            else:
                break

        # pos_matrix = torch.zeros_like(scores_tensor, dtype=torch.int)
        # pos_matrix[positive_u, positive_i] = 1
        # pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
        # pos_idx = torch.gather(pos_matrix, dim=1, index=topk_idx)
        # result = torch.cat((pos_idx, pos_len_list), dim=1)
        # return scores_top_k, result
        return scores_top_k, topk_idx


class BiasDisparityLoss(torch.nn.modules.loss._Loss):
    __constants__ = ['reduction']

    def __init__(self,
                 train_bias_ratio: dict,
                 item_categories_map: torch.Tensor,
                 sensitive_attributes: Iterable[str],
                 topk=10,
                 size_average=None, reduce=None, reduction: str = 'mean', margin=0.0) -> None:
        super(BiasDisparityLoss, self).__init__(size_average, reduce, reduction)

        self.train_bias_ratio = train_bias_ratio
        self.item_categories_map = item_categories_map
        self.sensitive_attributes = sensitive_attributes
        self.topk = topk

        self.margin = margin

    def forward(self, _input: torch.Tensor, demo_groups: torch.Tensor) -> torch.Tensor:
        sorted_target, sorted_input, _ = get_bias_disparity_sorted_target(_input,
                                                                          self.train_bias_ratio,
                                                                          self.item_categories_map,
                                                                          self.sensitive_attributes,
                                                                          demo_groups,
                                                                          topk=self.topk)

        # return (sorted_target.max() - sorted_target * sorted_input).mean(dim=1)
        return (sorted_target * sorted_input).mean(dim=1)


def get_bias_disparity_target_NDCGApprox(scores,
                                         train_bias_ratio,
                                         item_categories_map,
                                         sensitive_attributes,
                                         demo_groups,
                                         topk=10):
    sorted_target, _, sorted_idxs = get_bias_disparity_sorted_target(scores,
                                                                     train_bias_ratio,
                                                                     item_categories_map,
                                                                     sensitive_attributes,
                                                                     demo_groups,
                                                                     topk=topk)

    return torch.gather(sorted_target, 1, sorted_idxs)


def get_bias_disparity_sorted_target(scores,
                                     train_bias_ratio,
                                     item_categories_map,
                                     sensitive_attributes,
                                     demo_groups,
                                     topk=10):
    sorted_scores, sorted_idxs = torch.topk(scores, scores.shape[1])

    target = torch.zeros_like(sorted_idxs, dtype=torch.float)
    for attr in sensitive_attributes:
        attr_bias_ratio = train_bias_ratio[attr]

        for gr, gr_bias_ratio in attr_bias_ratio.items():
            if gr in demo_groups[attr]:
                gr_idxs = sorted_idxs[demo_groups[attr] == gr, :]
                gr_item_cats = item_categories_map[gr_idxs].view(-1, item_categories_map.shape[-1])
                gr_mean_bias = torch.nanmean(gr_bias_ratio[gr_item_cats], dim=1).nan_to_num(0)
                target[demo_groups[attr] == gr, :] += gr_mean_bias.view(-1, scores.shape[1]).to(target.device)

    target /= len(sensitive_attributes)

    mask_topk = torch.stack([(row < 1).nonzero().squeeze()[:topk] for row in target])
    target[:] = 0
    target[torch.arange(target.shape[0])[:, None], mask_topk] = 1

    # mask_topk = []
    # for row in target:
    #     low_bias_topk = (row < 1).nonzero().squeeze()[:topk]
    #     items_until_last_low = torch.arange(low_bias_topk.max() + 1).to(target.device)
    #     mask_topk.append(items_until_last_low[~items_until_last_low.unsqueeze(1).eq(low_bias_topk).any(-1)])
    # mask_topk = torch.stack(mask_topk)

    target[:] = 0
    target[torch.arange(target.shape[0])[:, None], mask_topk] = 1

    return target, sorted_scores, sorted_idxs
