# Based on https://github.com/RexYing/gnn-model-explainer/blob/master/explainer/explain.py

import os
import sys
import time
import math
import copy
import itertools
from logging import getLogger
from typing import Iterable, Tuple

import tqdm
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from recbole.evaluator import Evaluator
from recbole.utils import get_trainer, set_color
from recbole.data.interaction import Interaction

sys.path.append('..')

import src.utils as utils
import src.models as exp_models
from src.utils.early_stopping import EarlyStopping


class DPBGExplainer:

    def __init__(self, config, dataset, rec_data, model, user_id, dist="damerau_levenshtein", **kwargs):
        super(DPBGExplainer, self).__init__()
        self.model = model
        self.model.eval()
        self.dataset = dataset
        self.rec_data_loader = rec_data
        self.rec_data = rec_data.dataset

        self.user_id = user_id
        self.beta = config['cf_beta']
        self.device = config['device']
        self.only_subgraph = config['only_subgraph']
        self.unique_graph_dist_loss = config['save_unique_graph_dist_loss']

        self.tot_item_num = dataset.item_num
        self.item_tensor = dataset.get_item_feature().to(model.device)
        self.test_batch_size = self.tot_item_num

        # Instantiate CF model class, load weights from original model
        self.cf_model = nn.DataParallel(getattr(exp_models, f"{model.__class__.__name__}Perturbated")(config, dataset, self.user_id)).to(model.device)

        self.cf_model.load_state_dict(self.model.state_dict(), strict=False)

        # Freeze weights from original model in cf_model
        for name, param in self.cf_model.named_parameters():
            if name != "P_symm":
                param.requires_grad = False

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
        self.logger = getLogger()

        self.user_batch_exp = config['user_batch_exp']

        self.old_graph_dist = 0

        self.only_adv_group = config['only_adv_group']

        self.evaluator = Evaluator(config)

        trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

        gender_map = dataset.field2id_token['gender']
        female_idx, male_idx = (gender_map == 'F').nonzero()[0][0], (gender_map == 'M').nonzero()[0][0]

        females = rec_data.dataset.user_feat['gender'] == female_idx
        males = rec_data.dataset.user_feat['gender'] == male_idx
        rec_data_f, rec_data_m = copy.deepcopy(rec_data), copy.deepcopy(rec_data)

        rec_data_f.user_df = Interaction({k: v[females] for k, v in rec_data_f.dataset.user_feat.interaction.items()})
        rec_data_f.uid_list = rec_data_f.user_df['user_id']

        self.females_result = trainer.evaluate(rec_data_f, load_best_model=False, show_progress=config['show_progress'])
        self.logger.info(self.females_result)

        rec_data_m.user_df = Interaction({k: v[males] for k, v in rec_data_m.dataset.user_feat.interaction.items()})
        rec_data_m.uid_list = rec_data_m.user_df['user_id']
        self.males_result = trainer.evaluate(rec_data_m, load_best_model=False, show_progress=config['show_progress'])
        self.logger.info(self.males_result)

        check_func = "__ge__" if config['delete_adv_group'] else "__lt__"

        self.adv_group = male_idx if getattr(self.males_result['ndcg@10'], check_func)(self.females_result['ndcg@10']) else female_idx
        self.disadv_group = female_idx if self.adv_group == male_idx else male_idx
        self.results = dict(zip([male_idx, female_idx], [self.males_result, self.females_result]))

        self.earlys = EarlyStopping(
            config['earlys_patience'],
            config['earlys_ignore'],
            method=config['earlys_method'],
            fn=config['earlys_fn'],
            delta=config['earlys_delta']
        )

        self.earlys_pvalue = config['earlys_pvalue']

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

        self.topk_args = {'topk': topk}

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
                if not (self.old_graph_dist != new_graph_dist):
                    return best_loss

            best_cf_example.append(new_example + [first_fair_loss])
            self.old_graph_dist = new_example[-4]

            pref_data = pd.DataFrame(zip(*new_example[:2], new_example[3]), columns=['user_id', 'topk_pred', 'cf_topk_pred'])
            orig_res = utils.compute_metric(self.evaluator, self.rec_data, pref_data, 'topk_pred', 'ndcg')
            cf_res = utils.compute_metric(self.evaluator, self.rec_data, pref_data, 'cf_topk_pred', 'ndcg')

            user_feat = Interaction({k: v[torch.tensor(new_example[0])] for k, v in self.dataset.user_feat.interaction.items()})
            gender_map = self.dataset.field2id_token['gender']
            female_idx, male_idx = (gender_map == 'F').nonzero()[0][0], (gender_map == 'M').nonzero()[0][0]

            males = user_feat['gender'] == male_idx
            females = user_feat['gender'] == female_idx

            orig_f, orig_m = np.mean(orig_res[females, -1]), np.mean(orig_res[males, -1])
            cf_f, cf_m = np.mean(cf_res[females, -1]), np.mean(cf_res[males, -1])
            self.logger.info(f"Original => NDCG F: {orig_f}, NDCG M: {orig_m}, Diff: {np.abs(orig_f - orig_m)} \n"
                             f"CF       => NDCG F: {cf_f}, NDCG M: {cf_m}, Diff: {np.abs(cf_f - cf_m)}")

            if not self.unique_graph_dist_loss:
                return abs(loss_total)
        return best_loss

    @staticmethod
    def prepare_batched_data(batched_data, data, item_data=None):
        """
        Prepare the batched data according to the "recbole" pipeline
        :param batched_data:
        :param data:
        :param item_data:
        :return:
        """
        data_df = Interaction({k: v[batched_data] for k, v in data.dataset.user_feat.interaction.items()})

        if item_data is not None:
            data_df.update(Interaction({data.dataset.iid_field: item_data}))

        if hasattr(data, "uid2history_item"):
            history_item = data.uid2history_item[data_df['user_id']]
        else:
            history_item = []

        if len(batched_data) > 1:
            history_u = torch.cat([torch.full_like(hist_iid, i) for i, hist_iid in enumerate(history_item)])
            history_i = torch.cat(list(history_item))
        else:
            history_u = torch.full_like(history_item, 0)
            history_i = history_item

        return data_df, (history_u, history_i), None, None

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
            iter_data.append(torch.cat(batch_data))

        return iter_data

    def explain(self, batched_data, test_data, epochs, topk=10):
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
        orig_ndcg_loss = np.abs(self.males_result['ndcg@10'] - self.females_result['ndcg@10'])

        iter_epochs = tqdm.tqdm(
            range(epochs),
            total=epochs,
            ncols=100,
            desc=set_color(f"Epochs   ", 'blue'),
        )

        if self.only_adv_group != "global":
            iter_data = self.randperm2groups(batched_data)
            while any(d.unique().shape[0] < 1 for d in iter_data):  # check if each batch has at least 2 groups
                iter_data = self.randperm2groups(batched_data)
        else:
            batched_gender_data = self.rec_data_loader.dataset.user_feat['gender'][batched_data]
            iter_data = batched_data[batched_gender_data == self.adv_group].split(self.user_batch_exp)

        cwd_files = [f for f in os.listdir() if f.startswith('loss_trend_epoch')]
        if len(cwd_files) > 0 and os.path.isfile(cwd_files[0]):
            os.remove(cwd_files[0])

        fair_losses = []
        new_example = None
        for epoch in iter_epochs:
            iter_data = [iter_data[i] for i in np.random.permutation(len(iter_data))]
            epoch_fair_loss = []
            for batch_idx, batch_user in enumerate(iter_data):
                self.user_id = batch_user
                batched_data_epoch = DPBGExplainer.prepare_batched_data(batch_user, self.rec_data_loader)
                self.compute_model_predictions(batched_data_epoch, topk)

                new_example, loss_total, fair_loss = self.train(epoch, topk=topk)
                epoch_fair_loss.append(fair_loss)

            epoch_fair_loss = np.mean(epoch_fair_loss)
            fair_losses.append(epoch_fair_loss)

            if self.verbose:
                if os.path.isfile(f'loss_trend_epoch{epoch}.png'):
                    os.remove(f'loss_trend_epoch{epoch}.png')
                sns.lineplot(
                    x='epoch',
                    y='fair loss',
                    data=pd.DataFrame(zip(np.arange(epoch + 1), fair_losses), columns=['epoch', 'fair loss'])
                )
                plt.savefig(f'loss_trend_epoch{epoch + 1}.png')
                plt.close()

            if new_example is not None:
                new_example[6] = epoch_fair_loss

                # Recommendations generated passing test set (items in train and validation are considered watched)
                test_batch_data = DPBGExplainer.prepare_batched_data(batched_data, test_data)
                self.compute_model_predictions(test_batch_data, topk)
                test_model_topk = self.model_topk_idx.detach().cpu().numpy()

                self.cf_model.eval()
                with torch.no_grad():
                    test_cf_scores_pred = self.get_scores(self.cf_model, *self.scores_args, pred=True)
                    _, test_cf_topk_pred_idx = self.get_top_k(test_cf_scores_pred, **self.topk_args)
                test_cf_topk_pred_idx = test_cf_topk_pred_idx.detach().cpu().numpy()
                test_cf_dist = [
                    self.dist(_pred, _topk_idx) for _pred, _topk_idx in zip(test_cf_topk_pred_idx, test_model_topk)
                ]

                rec_batch_data = DPBGExplainer.prepare_batched_data(batched_data, self.rec_data_loader)
                self.compute_model_predictions(rec_batch_data, topk)
                rec_model_topk = self.model_topk_idx.detach().cpu().numpy()

                self.cf_model.eval()
                with torch.no_grad():
                    rec_cf_scores_pred = self.get_scores(self.cf_model, *self.scores_args, pred=True)
                    _, rec_cf_topk_pred_idx = self.get_top_k(rec_cf_scores_pred, **self.topk_args)
                rec_cf_topk_pred_idx = rec_cf_topk_pred_idx.detach().cpu().numpy()
                rec_cf_dist = [
                    self.dist(_pred, _topk_idx) for _pred, _topk_idx in zip(rec_cf_topk_pred_idx, rec_model_topk)
                ]

                new_example = [
                    batched_data.detach().numpy(),
                    rec_model_topk,
                    test_model_topk,
                    rec_cf_topk_pred_idx,
                    test_cf_topk_pred_idx,
                    rec_cf_dist,
                    test_cf_dist,
                    *new_example[4:]
                ]

                pref_data = pd.DataFrame(zip(new_example[0], new_example[3]), columns=['user_id', 'cf_topk_pred'])
                cf_res = utils.compute_metric(self.evaluator, self.rec_data, pref_data, 'cf_topk_pred', 'ndcg')

                user_feat = Interaction({k: v[torch.tensor(new_example[0])] for k, v in self.dataset.user_feat.interaction.items()})
                gender_map = self.dataset.field2id_token['gender']
                female_idx, male_idx = (gender_map == 'F').nonzero()[0][0], (gender_map == 'M').nonzero()[0][0]

                males = user_feat['gender'] == male_idx
                females = user_feat['gender'] == female_idx

                cf_f, cf_m = cf_res[females, -1], cf_res[males, -1]

                if self.earlys.check(epoch_fair_loss):
                    self.logger.info(self.earlys)
                    self.logger.info(f"Early Stopping: best epoch {epoch + 1 - len(self.earlys.history) + self.earlys.best_loss}")
                    if self.earlys_pvalue is None:
                        break

                if self.earlys_pvalue is not None:
                    stat = stats.f_oneway(cf_f, cf_m)
                    if stat.pvalue > self.earlys_pvalue and len(self.earlys.history) > self.earlys.ignore:
                        self.logger.info(f"Early Stopping with ANOVA test result: {stat}")
                        break

                best_loss = self.update_best_cf_example(best_cf_example, new_example, loss_total, best_loss, orig_ndcg_loss)

            self.logger.info("{} CF examples for user = {}".format(len(best_cf_example), self.user_id))

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

        user_feat = self.dataset.user_feat
        user_id_mask = self.user_id.unsqueeze(-1) if self.user_id.dim() == 0 else self.user_id
        user_feat = {k: feat[user_id_mask] for k, feat in user_feat.interaction.items()}

        target = torch.zeros_like(cf_scores, dtype=torch.float, device=cf_scores.device)
        target[torch.arange(target.shape[0])[:, None], self.rec_data.history_item_matrix()[0][user_feat['user_id']]] = 1
        target[:, 0] = 0

        loss_total, orig_loss_graph_dist, loss_graph_dist, fair_loss, cf_adj, adj = self.cf_model.loss(
            cf_scores,
            DPNDCGLoss(
                self.sensitive_attributes,
                user_feat,
                topk=topk,
                adv_group_data=(self.only_adv_group, self.disadv_group, self.results[self.disadv_group]['ndcg@10'])
            ),
            target
        )

        loss_total.backward()

        # import pdb; pdb.set_trace()
        # for name, param in self.cf_model.named_parameters():
        #     if name == "P_symm":
        #         print(param.grad[param.grad])

        nn.utils.clip_grad_norm_(self.cf_model.parameters(), 2.0)
        self.cf_optimizer.step()

        fair_loss = fair_loss.mean().item() if fair_loss is not None else torch.nan
        self.logger.info(f"Explain duration: {time.strftime('%H:%M:%S', time.gmtime(time.time() - t))}, " +
                         # 'User id: {}, '.format(str(self.user_id)) +
                         'Epoch: {}, '.format(epoch + 1) +
                         'loss: {:.4f}, '.format(loss_total.item()) +
                         'fair loss: {:.4f}, '.format(fair_loss) +
                         'graph loss: {:.4f}, '.format(loss_graph_dist.item()) +
                         'del edges: {:.4f}, '.format(int(orig_loss_graph_dist.item())))
        if self.verbose:
            self.logger.info('Orig output: {}\n'.format(self.model_scores) +
                             'Output: {}\n'.format(cf_scores) +
                             'Output nondiff: {}\n'.format(cf_scores_pred) +
                             '{:20}: {},\n {:20}: {},\n {:20}: {}\n'.format(
                                 'orig pred', self.model_topk_idx,
                                 'new pred', cf_topk_idx,
                                 'new pred nondiff', cf_topk_pred_idx)
                             )
        self.logger.info(" ")

        cf_stats = None
        if orig_loss_graph_dist.item() > 0:
            # Compute distance between original and perturbed list. Explanation maintained only if dist > 0
            # cf_dist = [self.dist(_pred, _topk_idx) for _pred, _topk_idx in zip(cf_topk_pred_idx, self.model_topk_idx)]
            cf_dist = None

            cf_adj, adj = cf_adj.detach().cpu().numpy(), adj.detach().cpu().numpy()
            del_edges = np.stack((adj != cf_adj).nonzero(), axis=0)
            del_edges = del_edges[:, del_edges[0, :] < self.dataset.user_num]  # remove duplicated edges

            cf_stats = [self.user_id.detach().numpy(),
                        self.model_topk_idx.detach().cpu().numpy(), cf_topk_pred_idx.detach().cpu().numpy(),
                        cf_dist, loss_total.item(), loss_graph_dist.item(), fair_loss, del_edges, epoch + 1]

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

    @staticmethod
    def get_scores(_model, batched_data, tot_item_num, test_batch_size, item_tensor, pred=False):
        device = _model.module.device if isinstance(_model, nn.DataParallel) else _model.device
        interaction, history_index, _, _ = batched_data
        inter_data = interaction.to(device)
        try:
            scores_kws = {'pred': pred} if pred is not None else {}
            scores = _model(inter_data, **scores_kws)

        except NotImplementedError:
            inter_len = len(interaction)
            new_inter = interaction.to(device, **scores_kws).repeat_interleave(tot_item_num)
            batch_size = len(new_inter)
            new_inter.update(item_tensor.repeat(inter_len))
            if batch_size <= test_batch_size:
                scores = _model.predict(new_inter)
            else:
                scores = DPBGExplainer._spilt_predict(new_inter, batch_size, test_batch_size, test_batch_size)

        scores = scores.view(-1, tot_item_num)
        scores[:, 0] = -np.inf
        if _model.ITEM_ID in interaction:
            scores = scores[:, inter_data[_model.ITEM_ID]]
        if history_index is not None:
            scores[history_index] = -np.inf

        return scores

    @staticmethod
    def get_top_k(scores_tensor, topk=10):
        scores_top_k, topk_idx = torch.topk(scores_tensor, topk, dim=-1)  # n_users x k

        return scores_top_k, topk_idx


class DPNDCGLoss(torch.nn.modules.loss._Loss):
    __constants__ = ['reduction']

    def __init__(self,
                 sensitive_attributes: Iterable[str],
                 user_feat,
                 topk=10,
                 adv_group_data: Tuple[str, int, float] = None,
                 size_average=None, reduce=None, reduction: str = 'mean', temperature=0.1) -> None:
        super(DPNDCGLoss, self).__init__(size_average, reduce, reduction)

        self.ndcg_loss = utils.NDCGApproxLoss(
            size_average=size_average,
            topk=topk,
            reduce=reduce,
            reduction=reduction,
            temperature=temperature
        )
        self.sensitive_attributes = sensitive_attributes
        self.user_feat = user_feat
        self.adv_group_data = adv_group_data

    def forward(self, _input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        groups = list(itertools.product(*[self.user_feat[attr].unique().numpy() for attr in self.sensitive_attributes]))
        masks = []
        for grs in groups:
            masks.append([(self.user_feat[attr] == gr).numpy() for gr, attr in zip(grs, self.sensitive_attributes)])
        masks = np.stack(masks)
        # bitwise and finds the users that belong simultaneously to the groups in the product
        masks = np.bitwise_and.reduce(masks, axis=1)

        ndcg_values = self.ndcg_loss(_input, target)

        masked_ndcg = []
        for mask in masks:
            masked_ndcg.append(ndcg_values[mask].mean(dim=0))
        masked_ndcg = torch.stack(masked_ndcg)

        loss = None
        for gr_i_idx in range(len(groups)):
            gr_i = groups[gr_i_idx]
            if self.adv_group_data[0] == "global":
                # the loss works to optimize NDCG towards -1, the global NDCG is however positive
                loss = (masked_ndcg[gr_i_idx] - (-self.adv_group_data[2])).abs()
            else:
                for gr_j_idx in range(gr_i_idx + 1, len(groups)):
                    l_val = masked_ndcg[gr_i_idx]
                    r_val = masked_ndcg[gr_j_idx]

                    if self.adv_group_data[0] == "local":
                        if self.adv_group_data[1] == gr_i:
                            l_val = l_val.detach()
                        else:
                            r_val = r_val.detach()

                    if loss is None:
                        loss = (l_val - r_val).abs()
                    else:
                        loss += (l_val - r_val).abs()

        return loss / max(math.comb(len(groups), 2), 1)
