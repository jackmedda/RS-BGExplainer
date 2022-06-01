# Based on https://github.com/RexYing/gnn-model-explainer/blob/master/explainer/explain.py

import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
from torchviz import make_dot

sys.path.append('..')

import utils
from models import GCMCPerturbated


class BGExplainer:

    def __init__(self, config, dataset, model, user_id, dist="damerau_levenshtein", **kwargs):
        super(BGExplainer, self).__init__()
        self.model = model
        self.model.eval()

        self.user_id = user_id
        self.beta = config['cf_beta']
        self.device = config['device']

        self.tot_item_num = dataset.item_num
        self.item_tensor = dataset.get_item_feature().to(model.device)
        self.test_batch_size = self.tot_item_num

        # Instantiate CF model class, load weights from original model
        self.cf_model = GCMCPerturbated(config, dataset, self.user_id)

        self.cf_model.load_state_dict(self.model.state_dict(), strict=False)

        # Freeze weights from original model in cf_model
        for name, param in self.cf_model.named_parameters():
            if name != "P_vec":
                param.requires_grad = False
        # for name, param in self.model.named_parameters():
        #     print("orig model requires_grad: ", name, param.requires_grad)
        # for name, param in self.cf_model.named_parameters():
        #     print("cf model requires_grad: ", name, param.requires_grad)

        lr = config['cf_learning_rate']
        momentum = kwargs.get("momentum", 0.0)
        sgd_kwargs = {'momentum': momentum, 'nesterov': True if momentum > 0 else False}
        if config["cf_optimizer"] == "SGD":
            self.cf_optimizer = torch.optim.SGD(self.cf_model.parameters(), lr=lr, **sgd_kwargs)
        elif config["cf_optimizer"] == "Adadelta":
            self.cf_optimizer = torch.optim.Adadelta(self.cf_model.parameters(), lr=lr)
        else:
            raise NotImplementedError("CF Optimizer not implemented")

        if dist == "set":
            self.dist = lambda topk_idx, cf_topk_idx: len(topk_idx) - (len(set(topk_idx) & set(cf_topk_idx)))
        elif dist == "damerau_levenshtein":
            self.dist = utils.damerau_levenshtein_distance

    def explain(self, batched_data, epochs, topk=10):
        best_cf_example = []
        best_loss = np.inf
        for epoch in range(epochs):
            new_example, loss_total = self.train(batched_data, epoch, topk=topk)
            if new_example is not None and loss_total < best_loss:
                best_cf_example.append(new_example)
                best_loss = loss_total
        print("{} CF examples for user = {}".format(len(best_cf_example), self.user_id))

        return best_cf_example

    def train(self, batched_data, epoch, topk=10):
        t = time.time()
        self.cf_model.train()
        self.cf_optimizer.zero_grad()

        _, _, positive_u, positive_i = batched_data
        scores_args = [batched_data, self.tot_item_num, self.test_batch_size, self.item_tensor]

        # topk_idx contains the ids of the topk items
        scores = get_scores(self.model, *scores_args, pred=None)
        scores_topk, topk_idx = get_top_k(scores, positive_u, positive_i, topk=topk)

        # cf_scores uses differentiable P_hat ==> adjacency matrix not binary, but needed for training
        cf_scores = get_scores(self.cf_model, *scores_args, pred=False)
        cf_scores_topk, cf_topk_idx = get_top_k(cf_scores, positive_u, positive_i, topk=topk)

        # cf_scores_pred uses thresholded P ==> binary adjacency matrix ==> gives actual prediction
        cf_scores_pred = get_scores(self.cf_model, *scores_args, pred=True)
        cf_scores_pred_topk, cf_topk_pred_idx = get_top_k(cf_scores_pred, positive_u, positive_i, topk=topk)

        # loss_pred indicator should be based on y_pred_new_actual NOT y_pred_new!
        loss_total, loss_pred, loss_graph_dist, cf_adj, adj = self.cf_model.loss(cf_scores_topk, scores_topk,
                                                                                 cf_scores_pred_topk)
        loss_total.backward()
        nn.utils.clip_grad_norm(self.cf_model.parameters(), 2.0)
        self.cf_optimizer.step()
        print(f"Explain duration: {time.strftime('%H:%M:%S', time.gmtime(time.time() - t))}",
              'User id: {}'.format(self.user_id),
              'Epoch: {}'.format(epoch + 1),
              'loss: {:.4f}'.format(loss_total.item()),
              'pred loss: {:.4f}'.format(loss_pred.item()),
              'graph loss: {:.4f}'.format(loss_graph_dist.item()))
        print('Output: {}\n'.format(cf_scores),
              'Output nondiff: {}\n'.format(cf_scores_pred),
              'orig pred: {}, new pred: {}, new pred nondiff: {}'.format(topk_idx, cf_topk_idx, cf_topk_pred_idx))
        print(" ")

        cf_stats = None
        if self.dist(cf_topk_pred_idx, topk_idx) > 0:
            cf_stats = [self.user_id, cf_adj.detach().cpu().numpy(), adj.detach().cpu().numpy(),
                        scores_topk, cf_scores_topk, cf_scores_pred_topk,
                        loss_total.item(), loss_pred.item(), loss_graph_dist.item()]

        return cf_stats, loss_total.item()


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


def get_scores(_model, batched_data, tot_item_num, test_batch_size, item_tensor, pred=False):
    interaction, history_index, _, _ = batched_data
    assert len(interaction.interaction[_model.USER_ID]) == 1
    try:
        # Note: interaction without item ids
        scores_kws = {'pred': pred} if pred is not None else {}
        scores = _model.full_sort_predict(interaction.to(_model.device), **scores_kws)
        dot = make_dot(scores.mean(), params=dict(_model.named_parameters()), show_attrs=True, show_saved=True)
        dot.format = 'png'
        dot.render(f'dots_{pred}')

    except NotImplementedError:
        inter_len = len(interaction)
        new_inter = interaction.to(_model.device).repeat_interleave(tot_item_num)
        batch_size = len(new_inter)
        new_inter.update(item_tensor.repeat(inter_len))
        if batch_size <= test_batch_size:
            scores = _model.predict(new_inter)
        else:
            scores = _spilt_predict(new_inter, batch_size, test_batch_size, test_batch_size)

    scores = scores.view(-1, tot_item_num)
    scores[:, 0] = -np.inf
    if history_index is not None:
        scores[history_index] = -np.inf

    return scores


def get_top_k(scores_tensor, positive_u, positive_i, topk=10):
    scores_top_k, topk_idx = torch.topk(scores_tensor, topk, dim=-1)  # n_users x k
    pos_matrix = torch.zeros_like(scores_tensor, dtype=torch.int)
    pos_matrix[positive_u, positive_i] = 1
    pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
    pos_idx = torch.gather(pos_matrix, dim=1, index=topk_idx)
    result = torch.cat((pos_idx, pos_len_list), dim=1)
    # return scores_top_k, result
    return scores_top_k.squeeze(), topk_idx.squeeze()
