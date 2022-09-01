# Based on https://github.com/RexYing/gnn-model-explainer/blob/master/explainer/explain.py

import sys
import time
import math
import itertools
from typing import Iterable

import tqdm
import numpy as np
import torch
import torch.nn as nn
from recbole.data.interaction import Interaction
from recbole.utils import set_color
from torchviz import make_dot

sys.path.append('..')

import src.utils as utils
from src.models import GCMCPerturbated

import subprocess
from memory_profiler import profile


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    # memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_info


class DPBGExplainer:

    def __init__(self, config, dataset, valid_data, model, user_id, dist="damerau_levenshtein", **kwargs):
        super(DPBGExplainer, self).__init__()
        self.model = model
        self.model.eval()
        self.dataset = dataset
        self.valid_data = valid_data

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

        self.sensitive_attributes = config['sensitive_attributes']

        self.scores_args, self.topk_args = None, None
        self.model_scores, self.model_scores_topk, self.model_topk_idx = None, None, None

        self.verbose = kwargs.get("verbose", False)

        self.user_batch_exp = config['user_batch_exp']

        self.old_graph_dist = 0

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
                self.old_graph_dist = best_cf_example[-1][-5]
                new_graph_dist = new_example[-4]
                if not (self.old_graph_dist < new_graph_dist):
                    return best_loss

            best_cf_example.append(new_example + [first_fair_loss])
            self.old_graph_dist = new_example[-4]
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

    def randperm2groups(self, batched_data):
        """
        At least 2 groups are represented in the batch following the distribution in the dataset.
        eps is used to select as an offset with respect to the fixed distribution. If a group has a 70% of distribution
        and the batch size is 32, then 22 +- (22 * eps) items are allocated for that group and the current batch
        :param batched_data:
        :return:
        """
        iter_data = []
        n_samples = batched_data.shape[0]
        n_batch = math.ceil(n_samples / self.user_batch_exp)

        groups = list(itertools.product(
            *[self.dataset.user_feat[attr][self.dataset.user_feat[attr] != 0].unique().numpy()
              for attr in self.sensitive_attributes]
        ))

        masks = []
        for grs in groups:
            masks.append([(self.dataset.user_feat[attr][self.dataset.user_feat[attr] != 0] == gr).numpy()
                          for gr, attr in zip(grs, self.sensitive_attributes)])
        masks = np.stack(masks)
        masks = np.bitwise_and.reduce(masks, axis=1)

        distrib = []
        for mask in masks:
            distrib.append(mask.nonzero()[0].shape[0] / batched_data.shape[0])

        for batch in range(n_batch):
            batch_len = min(n_samples, self.user_batch_exp)  # n_samples is lower than user_batch only for last batch
            batch_counter = batch_len
            batch_data = []
            for mask_i, mask_idx in enumerate(np.random.permutation(np.arange(masks.shape[0]))):
                if mask_i == (masks.shape[0] - 1):
                    n_mask_samples = batch_counter
                else:
                    if batch_counter < batch_len:
                        n_mask_samples = min(round(distrib[mask_idx] * batch_len), batch_counter)
                    else:
                        n_mask_samples = min(round(distrib[mask_idx] * batch_len), batch_counter - 1)
                mask_samples = np.random.permutation(masks[mask_idx].nonzero()[0])
                if batch != (n_batch - 1):
                    mask_samples = mask_samples[:n_mask_samples]
                batch_data.append(batched_data[mask_samples])
                masks[:, mask_samples] = False  # affect groups where these users belong (e.g. gender and age group)
                batch_counter -= mask_samples.shape[0]
                n_samples -= mask_samples.shape[0]

                if batch_counter == 0:
                    break
            iter_data.append(torch.concat(batch_data))

        return iter_data

    def explain(self, batched_data, epochs, topk=10):
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

        batched_data, data = batched_data

        iter_epochs = tqdm.tqdm(
            range(epochs),
            total=epochs,
            ncols=100,
            desc=set_color(f"Epochs   ", 'blue'),
        )

        iter_data = self.randperm2groups(batched_data)
        while any(d.unique().shape[0] < 1 for d in iter_data):  # check if each batch has at least 2 groups
            iter_data = self.randperm2groups(batched_data)
        for epoch in iter_epochs:
            iter_data = [iter_data[i] for i in np.random.permutation(len(iter_data))]
            for batch_idx, batch_user in enumerate(iter_data):
                self.user_id = batch_user
                batched_data_epoch = DPBGExplainer.prepare_batched_data(batch_user, data)
                self.compute_model_predictions(batched_data_epoch, topk)
                new_example, loss_total, fair_loss = self.train(epoch, topk=topk)
                if epoch == 0 and batch_idx == 0:
                    first_fair_loss = fair_loss

            if new_example is not None:
                all_batch_data = DPBGExplainer.prepare_batched_data(batched_data, data)
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

        user_feat = self.dataset.user_feat
        user_id_mask = self.user_id.unsqueeze(-1) if self.user_id.dim() == 0 else self.user_id
        user_feat = {k: feat[user_id_mask] for k, feat in user_feat.interaction.items()}

        loss_total, loss_pred, loss_graph_dist, fair_loss, cf_adj, adj, nnz_sub_adj, cf_dist = self.cf_model.loss(
            cf_scores,
            relevance_scores,
            self.model_topk_idx,
            cf_topk_pred_idx,
            self.dist,
            fair_loss_f=DPNDCGLoss(
                self.sensitive_attributes,
                self.valid_data.history_item_matrix()[0],
                topk=topk
            ),
            target=user_feat
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


class DPNDCGLoss(torch.nn.modules.loss._Loss):
    __constants__ = ['reduction']

    def __init__(self,
                 sensitive_attributes: Iterable[str],
                 history,
                 topk=10,
                 size_average=None, reduce=None, reduction: str = 'mean', temperature=0.1) -> None:
        super(DPNDCGLoss, self).__init__(size_average, reduce, reduction)

        self.ndcg_loss = utils.NDCGApproxLoss(
            size_average=size_average,
            topk=topk,
            reduce=reduce,
            reduction=reduction,
            temperature=temperature
        )
        self.history = history
        self.sensitive_attributes = sensitive_attributes

    def forward(self, _input: torch.Tensor, demo_groups: torch.Tensor) -> torch.Tensor:
        target = torch.zeros_like(_input, dtype=torch.float, device=_input.device)
        target[torch.arange(target.shape[0])[:, None], self.history[demo_groups['user_id']]] = 1
        target[:, 0] = 0

        groups = list(itertools.product(*[demo_groups[attr].unique().numpy() for attr in self.sensitive_attributes]))
        masks = []
        for grs in groups:
            masks.append([(demo_groups[attr] == gr).numpy() for gr, attr in zip(grs, self.sensitive_attributes)])
        masks = np.stack(masks)
        # bitwise and finds the users that belong to all groups simultaneously in the product
        masks = np.bitwise_and.reduce(masks, axis=1)

        ndcg = self.ndcg_loss(_input, target)

        masked_ndcg = []
        for mask in masks:
            masked_ndcg.append(ndcg[mask].mean(dim=0) / mask.nonzero()[0].shape[0])
        masked_ndcg = torch.stack(masked_ndcg)

        loss = None
        for gr_i in range(len(groups)):
            for gr_j in range(gr_i + 1, len(groups)):
                if loss is None:
                    loss = (masked_ndcg[gr_i] - masked_ndcg[gr_j]).abs()
                else:
                    loss += (masked_ndcg[gr_i] - masked_ndcg[gr_j]).abs()
        return loss / math.comb(len(groups), 2)
