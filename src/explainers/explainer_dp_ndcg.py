# Based on https://github.com/RexYing/gnn-model-explainer/blob/master/explainer/explain.py

import os
import sys
import time
import math
from logging import getLogger
from typing import Tuple, Callable, Dict

import tqdm
import wandb
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as sp_stats
import scipy.signal as sp_signal
import matplotlib.pyplot as plt
from recbole.evaluator import Evaluator
from recbole.utils import set_color
from recbole.data.interaction import Interaction

sys.path.append('..')

import src.utils as utils
import src.models as exp_models
from src.utils.early_stopping import EarlyStopping


class DPBGExplainer:

    def __init__(self, config, dataset, rec_data, model, dist="damerau_levenshtein", **kwargs):
        super(DPBGExplainer, self).__init__()
        self.config = config

        # self.cf_model = None
        self.model = model
        self.model.eval()
        self._cf_model = None

        self.dataset = dataset
        self.rec_data_loader = rec_data
        self.rec_data = rec_data.dataset

        self.cf_optimizer = None

        self.beta = config['cf_beta']
        self.device = config['device']
        self.only_subgraph = config['only_subgraph']
        self.unique_graph_dist_loss = config['save_unique_graph_dist_loss']

        self.tot_item_num = dataset.item_num
        self.item_tensor = dataset.get_item_feature().to(model.device)
        self.test_batch_size = self.tot_item_num

        if dist == "set":
            self.dist = lambda topk_idx, cf_topk_idx: len(topk_idx) - (len(set(topk_idx) & set(cf_topk_idx)))
        elif dist == "damerau_levenshtein":
            self.dist = utils.damerau_levenshtein_distance

        self.sensitive_attribute = config['sensitive_attribute']

        self.topk_args = None
        self.model_scores, self.model_scores_topk, self.model_topk_idx = None, None, None

        self.verbose = kwargs.get("verbose", False)
        self.logger = getLogger()

        self.user_batch_exp = config['user_batch_exp']

        self.old_graph_dist = 0

        self.only_adv_group = config['only_adv_group']

        self.evaluator = Evaluator(config)

        attr_map = dataset.field2id_token[self.sensitive_attribute]
        self.f_idx, self.m_idx = (attr_map == 'F').nonzero()[0][0], (attr_map == 'M').nonzero()[0][0]

        self.results = None
        self.adv_group, self.disadv_group = None, None

        self.earlys = EarlyStopping(
            config['early_stopping']['patience'],
            config['early_stopping']['ignore'],
            method=config['early_stopping']['method'],
            fn=config['early_stopping']['mode'],
            delta=config['early_stopping']['delta']
        )

        self.earlys_pvalue = config['early_stopping']['pvalue']

        self.previous_ndcg = config['previous_ndcg']
        self.previous_batch_LR_scaling = config['previous_batch_LR_scaling']

        self.increase_disparity = config['explainer_policies']['increase_disparity']
        self.group_deletion_constraint = config['explainer_policies']['group_deletion_constraint']

        wandb.config.update(config.final_config_dict)
        # wandb.watch(self.cf_model)

    @property
    def cf_model(self):
        if self._cf_model is None:
            print("Counterfactual Model Explainer is not initalized yet. Execute 'explain' to initialize it.")
        else:
            return self._cf_model

    @cf_model.setter
    def cf_model(self, value):
        self._cf_model = value

    def initialize_cf_model(self, **kwargs):
        # Instantiate CF model class, load weights from original model
        self.cf_model = getattr(
            exp_models,
            f"{self.model.__class__.__name__}Perturbed"
        )(self.config, self.dataset, **kwargs).to(self.model.device)

        self.cf_model.load_state_dict(self.model.state_dict(), strict=False)

        # Freeze weights from original model in cf_model
        for name, param in self.cf_model.named_parameters():
            if name != "P_symm":
                param.requires_grad = False

        lr = self.config['cf_learning_rate']
        momentum = self.config["momentum"] or 0.0
        sgd_kwargs = {'momentum': momentum, 'nesterov': True if momentum > 0 else False}
        if self.config["cf_optimizer"] == "SGD":
            self.cf_optimizer = torch.optim.SGD(self.cf_model.parameters(), lr=lr, **sgd_kwargs)
        elif self.config["cf_optimizer"] == "Adadelta":
            self.cf_optimizer = torch.optim.Adadelta(self.cf_model.parameters(), lr=lr)
        elif self.config["cf_optimizer"] == "AdamW":
            self.cf_optimizer = torch.optim.AdamW(self.cf_model.parameters(), lr=lr)
        else:
            raise NotImplementedError("CF Optimizer not implemented")

    def compute_model_predictions(self, scores_args, topk):
        """
        Compute the predictions of the original model without perturbation
        :param topk: integer of topk items
        :param scores_args: arguments needed by recbole to compute scores
        :return:
        """
        self.topk_args = {'topk': topk}

        self.model_scores = self.get_scores(self.model, *scores_args, pred=None)

        # topk_idx contains the ids of the topk items
        self.model_scores_topk, self.model_topk_idx = self.get_top_k(self.model_scores, **self.topk_args)

    def logging_exp_ndcg_per_group(self, new_example):
        pref_data = pd.DataFrame(zip(*new_example[:2], new_example[3]),
                                 columns=['user_id', 'topk_pred', 'cf_topk_pred'])
        orig_res = utils.compute_metric(self.evaluator, self.rec_data, pref_data, 'topk_pred', 'ndcg')
        cf_res = utils.compute_metric(self.evaluator, self.rec_data, pref_data, 'cf_topk_pred', 'ndcg')

        user_feat = Interaction({k: v[torch.tensor(new_example[0])] for k, v in self.dataset.user_feat.interaction.items()})

        m_users = user_feat[self.sensitive_attribute] == self.m_idx
        f_users = user_feat[self.sensitive_attribute] == self.f_idx

        orig_f, orig_m = np.mean(orig_res[f_users, -1]), np.mean(orig_res[m_users, -1])
        cf_f, cf_m = np.mean(cf_res[f_users, -1]), np.mean(cf_res[m_users, -1])
        self.logger.info(f"Original => NDCG F: {orig_f}, NDCG M: {orig_m}, Diff: {np.abs(orig_f - orig_m)} \n"
                         f"CF       => NDCG F: {cf_f}, NDCG M: {cf_m}, Diff: {np.abs(cf_f - cf_m)}")

    def update_best_cf_example(self, best_cf_example, new_example, loss_total, best_loss, first_fair_loss, force_update=False):
        """
        Updates the explanations with new explanation (if not None) depending on new loss value
        :param best_cf_example:
        :param new_example:
        :param loss_total:
        :param best_loss:
        :param first_fair_loss:
        :return:
        """
        if force_update or (new_example is not None and (abs(loss_total) < best_loss or self.unique_graph_dist_loss)):
            if self.unique_graph_dist_loss and len(best_cf_example) > 0:
                self.old_graph_dist = best_cf_example[-1][-5]
                new_graph_dist = new_example[-4]
                if not force_update and not (self.old_graph_dist != new_graph_dist):
                    return best_loss

            best_cf_example.append(new_example + [first_fair_loss])
            self.old_graph_dist = new_example[-4]

            if self.verbose:
                self.logging_exp_ndcg_per_group(new_example)

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

        attr = self.sensitive_attribute
        user_feat = self.dataset.user_feat[attr][batched_data]
        groups = user_feat.unique().numpy()

        masks = []
        for gr in groups:
            masks.append((user_feat == gr).numpy())
        masks = np.stack(masks)

        distrib = []
        for mask in masks:
            distrib.append(mask.nonzero()[0].shape[0] / batched_data.shape[0])

        for batch in range(n_batch):
            distrib = []
            for mask in masks:
                distrib.append(mask.nonzero()[0].shape[0] / n_samples)

            batch_len = min(n_samples, self.user_batch_exp)  # n_samples is lower than user_batch only for last batch
            batch_counter = batch_len
            batch_data = []
            for mask_i, mask_idx in enumerate(np.random.permutation(np.arange(masks.shape[0]))):
                if mask_i == (masks.shape[0] - 1):
                    n_mask_samples = batch_counter
                else:
                    if batch_counter < batch_len:
                        n_mask_samples = max(min(round(distrib[mask_idx] * batch_len), batch_counter), 1)
                    else:
                        n_mask_samples = max(min(round(distrib[mask_idx] * batch_len), batch_counter - 1), 1)
                mask_samples = np.random.permutation(masks[mask_idx].nonzero()[0])
                if batch != (n_batch - 1):
                    if mask_samples.shape[0] == n_mask_samples:
                        n_mask_samples = max(n_mask_samples - (n_batch - 1) - batch, 1)

                    mask_samples = mask_samples[:n_mask_samples]
                batch_data.append(batched_data[mask_samples])
                masks[mask_idx, mask_samples] = False  # affect groups where these users belong (e.g. gender and age group)
                batch_counter -= mask_samples.shape[0]
                n_samples -= mask_samples.shape[0]

                if batch_counter == 0:
                    break
            iter_data.append(torch.cat(batch_data))

        return iter_data

    def prepare_iter_batched_data(self, batched_data):
        if self.only_adv_group != "global":
            iter_data = self.randperm2groups(batched_data)
            # check if each batch has at least 2 groups
            while any(self.dataset.user_feat[self.sensitive_attribute][d].unique().shape[0] < 2 for d in iter_data):
                iter_data = self.randperm2groups(batched_data)
        else:
            batched_attr_data = self.rec_data_loader.dataset.user_feat[self.sensitive_attribute][batched_data]
            iter_data = batched_data[batched_attr_data == self.adv_group].split(self.user_batch_exp)

        return iter_data

    @staticmethod
    def _verbose_plot(fair_losses, epoch):
        if os.path.isfile(f'loss_trend_epoch{epoch}.png'):
            os.remove(f'loss_trend_epoch{epoch}.png')
        ax = sns.lineplot(
            x='epoch',
            y='fair loss',
            data=pd.DataFrame(zip(np.arange(1, epoch + 2), fair_losses), columns=['epoch', 'fair loss'])
        )
        if len(fair_losses) > 20:
            sns.lineplot(
                x=np.arange(1, epoch + 2),
                y=sp_signal.savgol_filter(fair_losses, window_length=len(fair_losses) // 2, polyorder=2),
                ax=ax
            )
        plt.savefig(f'loss_trend_epoch{epoch + 1}.png')
        plt.close()

    def _earlys_pvalue_check(self, new_example):
        pref_data = pd.DataFrame(zip(new_example[0], new_example[3]), columns=['user_id', 'cf_topk_pred'])
        cf_res = utils.compute_metric(self.evaluator, self.rec_data, pref_data, 'cf_topk_pred', 'ndcg')

        user_feat = Interaction(
            {k: v[torch.tensor(new_example[0])] for k, v in self.dataset.user_feat.interaction.items()})

        m_users = user_feat[self.sensitive_attribute] == self.m_idx
        f_users = user_feat[self.sensitive_attribute] == self.f_idx

        cf_f, cf_m = cf_res[f_users, -1], cf_res[m_users, -1]

        stat = sp_stats.f_oneway(cf_f, cf_m)
        check = stat.pvalue > self.earlys_pvalue and len(self.earlys.history) > self.earlys.ignore
        if check:
            self.logger.info(f"Early Stopping with ANOVA test result: {stat}")

        return check

    def determine_adv_group(self, batched_data, rec_model_topk):
        pref_users = batched_data.numpy()
        pref_data = pd.DataFrame(zip(pref_users, rec_model_topk.tolist()), columns=['user_id', 'topk_pred'])

        orig_res = utils.compute_metric(self.evaluator, self.rec_data, pref_data, 'topk_pred', 'ndcg')

        pref_users_sens_attr = self.dataset.user_feat[self.sensitive_attribute][pref_users].numpy()

        f_result = orig_res[pref_users_sens_attr == self.f_idx, -1].mean()
        m_result = orig_res[pref_users_sens_attr == self.m_idx, -1].mean()

        check_func = "__ge__" if self.config['delete_adv_group'] else "__lt__"

        self.adv_group = self.m_idx if getattr(m_result, check_func)(f_result) else self.f_idx
        self.disadv_group = self.f_idx if self.adv_group == self.m_idx else self.m_idx
        self.results = dict(zip([self.m_idx, self.f_idx], [m_result, f_result]))

    def increase_dataset_unfairness(self, batched_data, test_data, rec_model_topk, topk=10):
        pref_users = batched_data.numpy()
        pref_data = pd.DataFrame(zip(pref_users, rec_model_topk.tolist()), columns=['user_id', 'topk_pred'])

        orig_res = utils.compute_metric(self.evaluator, self.rec_data, pref_data, 'topk_pred', 'ndcg')

        m_users_mask = self.dataset.user_feat[self.sensitive_attribute][pref_users] == self.m_idx
        f_users_mask = self.dataset.user_feat[self.sensitive_attribute][pref_users] == self.f_idx

        m_size, f_size = m_users_mask.nonzero().shape[0], f_users_mask.nonzero().shape[0]
        if m_size >= f_size:
            gr_to_reduce, gr_fixed = self.m_idx, self.f_idx
            steps = np.linspace(f_size, m_size, 10, dtype=int)[::-1]
        else:
            gr_to_reduce, gr_fixed = self.f_idx, self.m_idx
            steps = np.linspace(m_size, f_size, 10, dtype=int)[::-1]

        df_res = pd.DataFrame(zip(
            pref_users,
            orig_res[:, - 1],
            self.dataset.user_feat[self.sensitive_attribute][pref_users].numpy()
        ), columns=['user_id', 'result', self.sensitive_attribute])

        if self.dataset.dataset_name == "lastfm-1k":
            ascending = True
            def check_func(gr_red_res, gr_fix_res): return gr_red_res <= gr_fix_res / 2
        else:
            ascending = False
            def check_func(gr_red_res, gr_fix_res): return gr_fix_res <= gr_red_res / 2

        for step in steps:
            step_df = df_res.groupby(self.sensitive_attribute).apply(
                lambda x: x.sort_values('result', ascending=ascending)[:step]
            ).reset_index(drop=True)
            mean_metric = step_df.groupby(self.sensitive_attribute).mean()

            batched_data = torch.tensor(step_df['user_id'].to_numpy())

            if check_func(mean_metric.loc[gr_to_reduce, 'result'], mean_metric.loc[gr_fixed, 'result']):
                print(mean_metric)
                break

        # recompute recommendations of original model
        test_batch_data = DPBGExplainer.prepare_batched_data(batched_data, test_data)
        test_scores_args = [test_batch_data, self.tot_item_num, self.test_batch_size, self.item_tensor]
        self.compute_model_predictions(test_scores_args, topk)
        test_model_topk = self.model_topk_idx.detach().cpu().numpy()

        rec_batch_data = DPBGExplainer.prepare_batched_data(batched_data, self.rec_data_loader)
        rec_scores_args = [rec_batch_data, self.tot_item_num, self.test_batch_size, self.item_tensor]
        self.compute_model_predictions(rec_scores_args, topk)
        rec_model_topk = self.model_topk_idx.detach().cpu().numpy()

        return batched_data, test_model_topk, test_scores_args, rec_model_topk, rec_scores_args

    def explain(self, batched_data, test_data, epochs, topk=10):
        """
        The method from which starts the perturbation of the graph by optimization of `pred_loss` or `fair_loss`
        :param batched_data:
        :param test_data:
        :param epochs:
        :param topk:
        :return:
        """
        best_cf_example = []
        best_loss = np.inf

        cwd_files = [f for f in os.listdir() if f.startswith('loss_trend_epoch')]
        if len(cwd_files) > 0 and os.path.isfile(cwd_files[0]):
            os.remove(cwd_files[0])

        test_batch_data = DPBGExplainer.prepare_batched_data(batched_data, test_data)
        test_scores_args = [test_batch_data, self.tot_item_num, self.test_batch_size, self.item_tensor]
        self.compute_model_predictions(test_scores_args, topk)
        test_model_topk = self.model_topk_idx.detach().cpu().numpy()

        rec_batch_data = DPBGExplainer.prepare_batched_data(batched_data, self.rec_data_loader)
        rec_scores_args = [rec_batch_data, self.tot_item_num, self.test_batch_size, self.item_tensor]
        self.compute_model_predictions(rec_scores_args, topk)
        rec_model_topk = self.model_topk_idx.detach().cpu().numpy()

        filtered_users = None
        if self.increase_disparity:
            batched_data, test_model_topk, test_scores_args, rec_model_topk, rec_scores_args = self.increase_dataset_unfairness(
                batched_data,
                test_data,
                rec_model_topk,
                topk=topk
            )

            filtered_users = batched_data

        self.determine_adv_group(batched_data, rec_model_topk)

        if self.group_deletion_constraint:
            if filtered_users is None:
                filtered_users = batched_data

            filtered_users = filtered_users[self.dataset.user_feat[self.sensitive_attribute][filtered_users] == self.adv_group]

        self.initialize_cf_model(filtered_users=filtered_users)

        orig_ndcg_loss = np.abs(self.results[self.adv_group] - self.results[self.disadv_group])
        self.logger.info(self.results)
        self.logger.info(f"M idx: {self.m_idx}")
        self.logger.info(f"Original Fair Loss: {orig_ndcg_loss}")

        iter_epochs = tqdm.tqdm(
            range(epochs),
            total=epochs,
            ncols=100,
            desc=set_color(f"Epochs   ", 'blue'),
        )

        new_example = None
        fair_losses = []
        orig_lr = [pg['lr'] for pg in self.cf_optimizer.param_groups]
        for epoch in iter_epochs:
            iter_data = self.prepare_iter_batched_data(batched_data)
            iter_data = [iter_data[i] for i in np.random.permutation(len(iter_data))]

            lr_scaling = np.linspace(0.1, 1, len(iter_data))

            if self.previous_ndcg:
                previous_ndcg = dict.fromkeys(self.dataset.user_feat[self.sensitive_attribute].unique().numpy())
                for gr in previous_ndcg:
                    previous_ndcg[gr] = np.array([])
            else:
                previous_ndcg = None

            epoch_fair_loss = []
            for batch_idx, batch_user in enumerate(iter_data):
                # Learning rate is scaled such that last epochs are more relevant, i.e. higher lr
                if self.previous_batch_LR_scaling:
                    for pg_idx, pg in enumerate(self.cf_optimizer.param_groups):
                        pg['lr'] = orig_lr[pg_idx] * lr_scaling[batch_idx]

                batched_data_epoch = DPBGExplainer.prepare_batched_data(batch_user, self.rec_data_loader)
                batch_scores_args = [batched_data_epoch, self.tot_item_num, self.test_batch_size, self.item_tensor]
                self.compute_model_predictions(batch_scores_args, topk)

                new_example, loss_total, fair_loss = self.train(epoch, batch_scores_args, batch_user, topk=topk, previous_ndcg=previous_ndcg)
                epoch_fair_loss.append(fair_loss)

            if self.previous_batch_LR_scaling:
                for pg_idx, pg in enumerate(self.cf_optimizer.param_groups):
                    pg['lr'] = orig_lr[pg_idx]

            epoch_fair_loss = np.mean(epoch_fair_loss)
            fair_losses.append(epoch_fair_loss)

            if self.verbose:
                DPBGExplainer._verbose_plot(fair_losses, epoch)

            if new_example is not None:
                new_example[6] = epoch_fair_loss

                wandb.log({'loss': epoch_fair_loss, '# Del Edges': new_example[-2].shape[1]})

                # Recommendations generated passing test set (items in train and validation are considered watched)
                self.cf_model.eval()
                with torch.no_grad():
                    test_cf_scores_pred = self.get_scores(self.cf_model, *test_scores_args, pred=True)
                    _, test_cf_topk_pred_idx = self.get_top_k(test_cf_scores_pred, **self.topk_args)
                test_cf_topk_pred_idx = test_cf_topk_pred_idx.detach().cpu().numpy()
                test_cf_dist = [
                    self.dist(_pred, _topk_idx) for _pred, _topk_idx in zip(test_cf_topk_pred_idx, test_model_topk)
                ]

                with torch.no_grad():
                    rec_cf_scores_pred = self.get_scores(self.cf_model, *rec_scores_args, pred=True)
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

                if self.earlys.check(epoch_fair_loss):
                    self.logger.info(self.earlys)
                    best_epoch = epoch + 1 - self.earlys.patience
                    self.logger.info(f"Early Stopping: best epoch {best_epoch}")

                    # stub exampled added to find again the best epoch when explanations are loaded
                    best_loss = self.update_best_cf_example(best_cf_example, new_example, loss_total, best_loss, orig_ndcg_loss, force_update=True)

                    break

                if self.earlys_pvalue is not None:
                    if self._earlys_pvalue_check(new_example):
                        break

                best_loss = self.update_best_cf_example(best_cf_example, new_example, loss_total, best_loss, orig_ndcg_loss)

            self.logger.info("{} CF examples".format(len(best_cf_example)))

        return best_cf_example

    # @profile
    def train(self, epoch, scores_args, users_ids, topk=10, previous_ndcg=None):
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
            cf_scores_pred = self.get_scores(self.cf_model, *scores_args, pred=True)
            cf_scores_pred_topk, cf_topk_pred_idx = self.get_top_k(cf_scores_pred, **self.topk_args)

        self.cf_optimizer.zero_grad()
        self.cf_model.train()

        # compute differentiable permutation of adj matrix
        # cf_scores uses differentiable P_hat ==> adjacency matrix not binary, but needed for training
        cf_scores = self.get_scores(self.cf_model, *scores_args, pred=False)

        # remove neginf from output
        cf_scores = torch.nan_to_num(cf_scores, neginf=(torch.min(cf_scores[~torch.isinf(cf_scores)]) - 1).item())
        cf_scores_topk, cf_topk_idx = self.get_top_k(cf_scores, **self.topk_args)

        user_feat = self.dataset.user_feat
        user_id_mask = users_ids.unsqueeze(-1) if users_ids.dim() == 0 else users_ids
        user_feat = {k: feat[user_id_mask] for k, feat in user_feat.interaction.items()}

        target = torch.zeros_like(cf_scores, dtype=torch.float, device=cf_scores.device)
        target[torch.arange(target.shape[0])[:, None], self.rec_data.history_item_matrix()[0][user_feat['user_id']]] = 1
        target[:, 0] = 0

        loss_total, orig_loss_graph_dist, loss_graph_dist, fair_loss, adj_sub_cf_adj = self.cf_model.loss(
            cf_scores,
            DPNDCGLoss(
                self.sensitive_attribute,
                user_feat,
                topk=topk,
                adv_group_data=(self.only_adv_group, self.disadv_group, self.results[self.disadv_group]),
                previous_ndcg=previous_ndcg
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
        self.logger.info(f"{self.cf_model.__class__.__name__} " +
                         f"Explain duration: {time.strftime('%H:%M:%S', time.gmtime(time.time() - t))}, " +
                         # 'User id: {}, '.format(str(users_ids)) +
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

            adj_del_edges = adj_sub_cf_adj.detach().cpu()
            del_edges = adj_del_edges.indices()[:, adj_del_edges.values().nonzero().squeeze()]

            del_edges = del_edges[:, (del_edges[0, :] < self.dataset.user_num) & (del_edges[0, :] > 0)].numpy()  # remove duplicated edges

            cf_stats = [users_ids.detach().numpy(),
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
        interaction, history_index, _, _ = batched_data
        inter_data = interaction.to(_model.device)
        try:
            scores_kws = {'pred': pred} if pred is not None else {}
            scores = _model.full_sort_predict(inter_data, **scores_kws)

        except NotImplementedError:
            inter_len = len(interaction)
            new_inter = interaction.to(_model.device, **scores_kws).repeat_interleave(tot_item_num)
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
                 sensitive_attribute: str,
                 user_feat,
                 topk=10,
                 adv_group_data: Tuple[str, int, float] = None,
                 previous_ndcg: Dict[str, np.ndarray] = None,
                 size_average=None, reduce=None, reduction: str = 'mean', temperature=0.1) -> None:
        super(DPNDCGLoss, self).__init__(size_average, reduce, reduction)

        self.ndcg_loss: Callable = utils.NDCGApproxLoss(
            size_average=size_average,
            topk=topk,
            reduce=reduce,
            reduction=reduction,
            temperature=temperature
        )
        self.sensitive_attribute = sensitive_attribute
        self.user_feat = user_feat
        self.adv_group_data = adv_group_data
        self.previous_ndcg = previous_ndcg

    def forward(self, _input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        groups = self.user_feat[self.sensitive_attribute].unique().numpy()
        masks = []
        for gr in groups:
            masks.append((self.user_feat[self.sensitive_attribute] == gr).numpy())
        masks = np.stack(masks)

        ndcg_values = self.ndcg_loss(_input, target)

        masked_ndcg = []
        for gr, mask in zip(groups, masks):
            if self.previous_ndcg is not None:
                gr_ndcg = (ndcg_values[mask].sum() + self.previous_ndcg[gr].sum())
                gr_ndcg = gr_ndcg / (ndcg_values[mask].shape[0] + self.previous_ndcg[gr].shape[0])
                masked_ndcg.append(gr_ndcg.unsqueeze(dim=-1))
                if self.previous_ndcg[gr].shape[0] > 0:
                    self.previous_ndcg[gr] = np.concatenate([
                        ndcg_values[mask].detach().cpu().numpy(),
                        self.previous_ndcg[gr]
                    ], axis=0)
                else:
                    self.previous_ndcg[gr] = ndcg_values[mask].detach().cpu().numpy()
            else:
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
