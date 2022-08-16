# Based on https://github.com/RexYing/gnn-model-explainer/blob/master/explainer/explain.py

import sys
from typing import Iterable

import time

import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from recbole.data.interaction import Interaction
from recbole.utils import set_color
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
        self.train_pref_ratio = kwargs.get("train_pref_ratio", None)
        self.cat_sharing_prob = kwargs.get("cat_sharing_prob", None)
        self.sensitive_attributes = config['sensitive_attributes']

        self.scores_args, self.topk_args = None, None
        self.model_scores, self.model_scores_topk, self.model_topk_idx = None, None, None

        self.verbose = kwargs.get("verbose", False)

        self.group_explain = config['group_explain']
        self.user_batch_exp = config['user_batch_exp']

        self.fair_target_lambda = config['fair_target_lambda']

    def compute_model_predictions(self, batched_data, topk, loaded_scores=None, old_field2token_id=None):
        """
        Compute the predictions of the original model without perturbation
        :param batched_data: current data in batch
        :param topk: integer of topk items
        :param loaded_scores: if not None loads pre-computed prediction of the original model
        :param old_field2token_id: mapping of ids related to pre-computed predictions
        :return:
        """
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

    def update_best_cf_example(self, best_cf_example, new_example, loss_total, best_loss, first_fair_loss):
        """
        Updates the explanations with new explanation (if not None) depending on new loss value
        :param best_cf_example:
        :param new_example:
        :param loss_total:
        :param best_loss:
        :param first_fair_loss:
        :return:
        """
        if new_example is not None and (abs(loss_total) < best_loss or self.unique_graph_dist_loss):
            if self.unique_graph_dist_loss and len(best_cf_example) > 0:
                old_graph_dist = best_cf_example[-1][-5]
                new_graph_dist = new_example[-4]
                if not (old_graph_dist < new_graph_dist):
                    return best_loss

            best_cf_example.append(new_example + [first_fair_loss])
            if not self.unique_graph_dist_loss:
                return abs(loss_total)
        return best_loss

    @staticmethod
    def prepare_batched_data(batched_data, data):
        """
        Prepare the batched data according to the "recbole" pipeline
        :param batched_data:
        :param data:
        :return:
        """
        user_id = batched_data
        user_df_mask = (data.user_df[data.uid_field][..., None] == user_id).any(-1)
        user_df = Interaction({k: v[user_df_mask] for k, v in data.user_df.interaction.items()})
        history_item = data.uid2history_item[user_id]

        if len(user_id) > 1:
            history_u = torch.cat([torch.full_like(hist_iid, i) for i, hist_iid in enumerate(history_item)])
            history_i = torch.cat(list(history_item))
        else:
            history_u = torch.full_like(history_item, 0)
            history_i = history_item

        return user_df, (history_u, history_i), None, None

    def get_iter_data(self, user_data):
        user_data = user_data.split(self.user_batch_exp)

        return (
            tqdm.tqdm(
                user_data,
                total=len(user_data),
                ncols=100,
                desc=set_color(f"Explaining   ", 'pink'),
            )
        )

    def explain(self, batched_data, epochs, topk=10, loaded_scores=None, old_field2token_id=None):
        """
        The method from which starts the perturbation of the graph by optimization of `pred_loss` or `fair_loss`
        :param batched_data:
        :param epochs:
        :param topk:
        :param loaded_scores:
        :param old_field2token_id:
        :return:
        """
        best_cf_example = []
        best_loss = np.inf
        first_fair_loss = None

        if not self.group_explain:
            self.compute_model_predictions(batched_data, topk, loaded_scores=loaded_scores, old_field2token_id=old_field2token_id)
        else:
            batched_data, data = batched_data

        iter_epochs = tqdm.tqdm(
            range(epochs),
            total=epochs,
            ncols=100,
            desc=set_color(f"Epochs   ", 'blue'),
        )

        for epoch in iter_epochs:
            if self.group_explain:
                iter_data = batched_data.split(self.user_batch_exp)
                for batch_idx, batch_user in enumerate(iter_data):
                    self.user_id = batch_user
                    batched_data_epoch = BGExplainer.prepare_batched_data(batch_user, data)
                    self.compute_model_predictions(batched_data_epoch, topk)
                    new_example, loss_total, fair_loss = self.train(epoch, topk=topk)
                    if epoch == 0 and batch_idx == 0:
                        first_fair_loss = fair_loss

                if new_example is not None:
                    all_batch_data = BGExplainer.prepare_batched_data(batched_data, data)
                    self.compute_model_predictions(all_batch_data, topk)
                    model_topk = self.model_topk_idx.detach().cpu().numpy()

                    self.cf_model.eval()
                    with torch.no_grad():
                        cf_scores_pred = self.get_scores(self.cf_model, *self.scores_args, pred=True)
                        _, cf_topk_pred_idx = self.get_top_k(cf_scores_pred, **self.topk_args)
                    cf_topk_pred_idx = cf_topk_pred_idx.detach().cpu().numpy()
                    cf_dist = [self.dist(_pred, _topk_idx) for _pred, _topk_idx in zip(cf_topk_pred_idx, model_topk)]

                    new_example = [batched_data.detach().numpy(), model_topk, cf_topk_pred_idx, cf_dist, *new_example[4:]]

                    best_loss = self.update_best_cf_example(best_cf_example, new_example, loss_total, best_loss, first_fair_loss)
            else:
                new_example, loss_total, fair_loss = self.train(epoch, topk=topk)
                if epoch == 0:
                    first_fair_loss = fair_loss
                # import pdb; pdb.set_trace()
                best_loss = self.update_best_cf_example(best_cf_example, new_example, loss_total, best_loss, first_fair_loss)

            print("{} CF examples for user = {}".format(len(best_cf_example), self.user_id))

        return best_cf_example, self.model_scores.detach().cpu().numpy()

    # @profile
    def train(self, epoch, topk=10):
        """
        Training procedure of explanation
        :param epoch:
        :param topk:
        :return:
        """
        t = time.time()

        self.cf_model.eval()

        # compute non-differentiable permutation of adj matrix
        with torch.no_grad():
            cf_scores_pred = self.get_scores(self.cf_model, *self.scores_args, pred=True)
            cf_scores_pred_topk, cf_topk_pred_idx = self.get_top_k(cf_scores_pred, **self.topk_args)

        self.cf_optimizer.zero_grad()
        self.cf_model.train()

        # compute differentiable permutation of adj matrix
        # cf_scores uses differentiable P_hat ==> adjacency matrix not binary, but needed for training
        cf_scores = self.get_scores(self.cf_model, *self.scores_args, pred=False)
        # remove neginf from output
        cf_scores = torch.nan_to_num(cf_scores, neginf=(torch.min(cf_scores[~torch.isinf(cf_scores)]) - 1).item())
        cf_scores_topk, cf_topk_idx = self.get_top_k(cf_scores, **self.topk_args)

        # target when Silvestri et al. method is used
        relevance_scores = torch.zeros_like(self.model_scores).float().to(self.device)
        relevance_scores[torch.arange(relevance_scores.shape[0])[:, None], self.model_topk_idx] = 1

        kwargs = {}
        if self.train_bias_ratio is not None:  # prepare the loss function for fairness purposes
            user_feat = self.dataset.user_feat
            user_id_mask = self.user_id.unsqueeze(-1) if self.user_id.dim() == 0 else self.user_id
            user_feat = {k: feat[user_id_mask] for k, feat in user_feat.interaction.items()}

            if self.explain_fairness_NDCGApprox:
                kwargs = {
                    "fair_loss_f": utils.NDCGApproxLoss(),
                    "target": get_bias_disparity_target_NDCGApprox(
                        cf_scores,
                        self.train_bias_ratio,
                        self.train_pref_ratio,
                        self.dataset.item_feat['class'],
                        self.sensitive_attributes,
                        user_feat,
                        topk=topk,
                        lmb=self.fair_target_lambda
                    )
                }
            else:
                kwargs = {
                    "fair_loss_f": BiasDisparityLoss(
                        self.train_bias_ratio,
                        self.train_pref_ratio,
                        self.dataset.item_feat['class'],
                        self.sensitive_attributes,
                        topk=topk,
                        lmb=self.fair_target_lambda
                    ),
                    "target": user_feat
                }

        loss_total, loss_pred, loss_graph_dist, fair_loss, cf_adj, adj, nnz_sub_adj, cf_dist = self.cf_model.loss(
            cf_scores,
            relevance_scores,
            self.model_topk_idx,
            cf_topk_pred_idx,
            self.dist,
            **kwargs
        )

        loss_total.backward()

        # Code that saves in a file the computational graph for debug purposes
        # dot = make_dot(loss_total.mean(), params=dict(self.cf_model.named_parameters()), show_attrs=True, show_saved=True)
        # dot.format = 'png'
        # dot.render(f'dots_loss')

        # Debug code that plots the gradient of perturbation matrix
        # for name, param in self.cf_model.named_parameters():
        #     if name == "P_symm":
        #         print(param.grad[param.grad.nonzero()])
        # input()

        nn.utils.clip_grad_norm_(self.cf_model.parameters(), 2.0)
        self.cf_optimizer.step()

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
        if self.verbose:
            print('Orig output: {}\n'.format(self.model_scores),
                  'Output: {}\n'.format(cf_scores),
                  'Output nondiff: {}\n'.format(cf_scores_pred),
                  '{:20}: {},\n {:20}: {},\n {:20}: {}\n'.format(
                      'orig pred', self.model_topk_idx,
                      'new pred', cf_topk_idx,
                      'new pred nondiff', cf_topk_pred_idx)
                  )
        print(" ")

        # Compute distance between original and perturbed list. Explanation maintained only if dist > 0
        cf_stats = None
        if cf_dist is None:
            cf_dist = [self.dist(_pred, _topk_idx) for _pred, _topk_idx in zip(cf_topk_pred_idx, self.model_topk_idx)]
        if (np.array(cf_dist) > 0).any() or self.force_return:
            cf_adj, adj = cf_adj.detach().cpu().numpy(), adj.detach().cpu().numpy()
            del_edges = np.stack((adj != cf_adj).nonzero(), axis=0)
            del_edges = del_edges[:, del_edges[0, :] <= self.dataset.user_num]  # remove duplicated edges

            cf_stats = [self.user_id.detach().numpy(),
                        self.model_topk_idx.detach().cpu().numpy(), cf_topk_pred_idx.detach().cpu().numpy(), cf_dist,
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

        return scores_top_k, topk_idx


class BiasDisparityLoss(torch.nn.modules.loss._Loss):
    __constants__ = ['reduction']

    def __init__(self,
                 train_bias_ratio: dict,
                 item_categories_map: torch.Tensor,
                 sensitive_attributes: Iterable[str],
                 topk=10,
                 lmb=0.5,
                 size_average=None, reduce=None, reduction: str = 'mean', margin=0.1) -> None:
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

        return (sorted_target * sorted_input - sorted_input.max() - self.margin).mean(dim=1)


def get_bias_disparity_target_NDCGApprox(scores,
                                         train_bias_ratio,
                                         train_pref_ratio,
                                         item_categories_map,
                                         sensitive_attributes,
                                         demo_groups,
                                         topk=10,
                                         lmb=0.5,
                                         cat_sharing_prob=None):
    sorted_target, _, sorted_idxs = get_bias_disparity_sorted_target(scores,
                                                                     train_bias_ratio,
                                                                     train_pref_ratio,
                                                                     item_categories_map,
                                                                     sensitive_attributes,
                                                                     demo_groups,
                                                                     topk=topk,
                                                                     lmb=lmb,
                                                                     cat_sharing_prob=cat_sharing_prob)

    return torch.gather(sorted_target, 1, torch.argsort(sorted_idxs))


def get_bias_disparity_sorted_target(scores,
                                     train_bias_ratio,
                                     train_pref_ratio,
                                     item_categories_map,
                                     sensitive_attributes,
                                     demo_groups,
                                     topk=10,
                                     lmb=0.5,
                                     offset=0.05,
                                     cat_sharing_prob=None):
    sorted_scores, sorted_idxs = torch.topk(scores, scores.shape[1])

    target = torch.zeros_like(sorted_idxs, dtype=torch.float)
    for attr in sensitive_attributes:
        attr_pref_ratio = train_pref_ratio[attr]  # dict that maps demographic groups to pref ratio for each category
        assert len(sensitive_attributes) == 1, "Not supported with multiple sensitive attributes at once"
        for gr, gr_pref_ratio in attr_pref_ratio.items():
            if gr in demo_groups[attr]:
                gr_idxs = sorted_idxs[demo_groups[attr] == gr, :]
                gr_target = torch.zeros_like(gr_idxs, dtype=torch.float)

                cat_order = torch.randperm(gr_pref_ratio.shape[0])  # random order of cat to avoid attention on first
                for cat in cat_order:
                    if not torch.isnan(gr_pref_ratio[cat]):
                        # counts how many ones are already in the target of each user
                        one_counts = torch.count_nonzero((gr_target > 0), dim=1).cpu().numpy()
                        count_df = pd.DataFrame(
                            zip(np.arange(one_counts.shape[0]), one_counts),
                            columns=["user", "count"]
                        )

                        # generate dataframe with all items with category cat in the list of scores of the current users
                        rank_df = pd.DataFrame(
                            (item_categories_map[gr_idxs] == cat).nonzero().detach().cpu().numpy(),
                            columns=['user', 'rank', 'cat']
                        )

                        df = rank_df.join(count_df.set_index("user"), on="user")
                        # if some users have the topk full => don't add
                        df_filt = df[df["count"] < topk]

                        if not df_filt.empty:
                            df_filt = df_filt.sort_values("rank")
                            # take the percentage of distribution of this category for each user
                            # if gr_pref_ratio[cat] = 0.3, n_target would be 3 in a top-10 setting
                            # this means that for each user we can choose maximum 3 items of this cat
                            prob = gr_pref_ratio[cat].round(decimals=2)
                            p = np.array([1 - prob, prob])
                            p /= p.sum()
                            # if the probability is too low for n_target to be 1, then n_target becomes 1 depending
                            # on a random choice based on the probability itself
                            n_target = max(int(prob * topk), np.random.choice([0, 1], p=p))
                            # n_target is reduced for a category `cat` if many items share `cat` with other categories
                            if cat_sharing_prob is not None:
                                n_target = int(n_target * (1 - cat_sharing_prob[cat]))
                            target_items = df_filt.iloc[:(n_target * one_counts.shape[0])].groupby("user").apply(
                                lambda _df: _df.take(np.arange(min(_df.shape[0], topk - _df["count"].head(1).values)))
                            )
                            target_items = torch.tensor(target_items[["user", "rank"]].values)

                            gr_target[target_items[:, 0], target_items[:, 1]] = 1
                        else:
                            break

                target[demo_groups[attr] == gr, :] = gr_target

    # target = torch.zeros_like(sorted_idxs, dtype=torch.float)
    # for attr in sensitive_attributes:
    #     attr_bias_ratio = train_bias_ratio[attr]  # dict that maps demographic groups to bias ratio for each category
    #
    #     for gr, gr_bias_ratio in attr_bias_ratio.items():
    #         if gr in demo_groups[attr]:  # if at least one user (row) in `scores` belong to the current group
    #             gr_idxs = sorted_idxs[demo_groups[attr] == gr, :]
    #             # each item id is mapped to its respective categories
    #             gr_item_cats = item_categories_map[gr_idxs].view(-1, item_categories_map.shape[-1])
    #             # each category is mapped to its bias in training and the mean is taken over the categories of each item
    #             gr_mean_bias = torch.nanmean(gr_bias_ratio[gr_item_cats], dim=1).nan_to_num(0)
    #             target[demo_groups[attr] == gr, :] += gr_mean_bias.view(-1, scores.shape[1]).to(target.device)
    #             import pdb; pdb.set_trace()
    #
    # target /= len(sensitive_attributes)
    #
    # # mask_topk = torch.stack([(row < 1).nonzero().squeeze()[:topk] for row in target])
    #
    # import pdb; pdb.set_trace()
    #
    # mask_topk = []
    # for score, row in zip(sorted_scores, target):
    #     # candidated items
    #     low_bias_items = (row < 1).nonzero().squeeze()[:topk]
    #     last_low_bias = low_bias_items.max()
    #     # we take the temporary target with the bias scores until the last topk-th item with low bias
    #     _row_slice, _score_slice = row[:(last_low_bias + 1)], score[:(last_low_bias + 1)]
    #
    #     _score_slice_min, _row_slice_min = _score_slice.min(), _row_slice.min()
    #     # the scores of the model are min-max normalized with min and max of the bias scores
    #     _score_slice_std = (_score_slice - _score_slice_min) / (_score_slice.max() - _score_slice_min)
    #     _score_row_scaled = _score_slice_std * (_row_slice.max() - _row_slice_min) + _row_slice_min
    #
    #     # row slice values are rounded to force the "low bias" and "high bias" in this way:
    #     # case 1: row_slice < (1 - offset) = 0.5
    #     # case 2: row_slice > (1 + offset) = 1.5 .. 2.0 .. (depending if it is grater than 1 + offset or 1.5 etc)
    #     # case 3: row_slice >= (1 - offset) and <= (1 + offset) = 1.0
    #     # case_1 = _row_slice < (1 - offset)
    #     # case_2 = _row_slice > (1 + offset)
    #     # case_3 = ~(case_1 | case_2)
    #     # _row_slice[case_1] = 0.5
    #     # _row_slice[case_2] = torch.ceil_(_row_slice[case_2] * 2) / 2
    #     # _row_slice[case_3] = 1
    #
    #     # then the scaled scores are divided by the bias scores, low bias score and high scale score result in high rank
    #     _ranks = _score_row_scaled / _row_slice
    #     _ranks_min, _ranks_max = _ranks[low_bias_items].min(), _ranks[low_bias_items].max()
    #     # depending on `lmb` the low bias items are "boosted" with a value such that more low bias items are selected
    #     _ranks[low_bias_items] += (_ranks_max - _ranks_min + 0.01) * lmb
    #
    #     mask_topk.append(torch.topk(_ranks, k=topk)[1])
    # mask_topk = torch.stack(mask_topk)
    #
    # # mask_topk = []
    # # for row in target:
    # #     low_bias_topk = (row < 1).nonzero().squeeze()[:topk]
    # #     items_until_last_low = torch.arange(low_bias_topk.max() + 1).to(target.device)
    # #     mask_topk.append(items_until_last_low[~items_until_last_low.unsqueeze(1).eq(low_bias_topk).any(-1)])
    # # mask_topk = torch.stack(mask_topk)
    #
    # target[:] = 0
    # target[torch.arange(target.shape[0])[:, None], mask_topk] = 1

    return target, sorted_scores, sorted_idxs
