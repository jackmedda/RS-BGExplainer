# Based on https://github.com/RexYing/gnn-model-explainer/blob/master/explainer/explain.py

import os
import sys
import time
import math
from logging import getLogger

import tqdm
import wandb
import gmpy2
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.signal as sp_signal
import matplotlib.pyplot as plt
from recbole.evaluator import Evaluator
from recbole.utils import set_color
from recbole.data.interaction import Interaction

import gnnuers.utils as utils
import gnnuers.models as exp_models
import gnnuers.evaluation as eval_utils
from gnnuers.utils.early_stopping import EarlyStopping
from gnnuers.losses import get_ranking_loss, get_fair_loss

from . import utils as exp_utils


class Explainer:

    def __init__(self, config, dataset, rec_data, model, dist="damerau_levenshtein", **kwargs):
        super(Explainer, self).__init__()
        self.config = config

        # self.cf_model = None
        self.model = model
        self.model.eval()
        self._cf_model = None

        self.dataset = dataset
        self.rec_data = rec_data
        self._pred_as_rec = config['exp_rec_data'] == 'rec'
        self._test_history_matrix = None

        self.cf_optimizer = None
        self.mini_batch_descent = config['mini_batch_descent']

        self.beta = config['cf_beta']
        self.device = config['device']
        self.only_subgraph = config['only_subgraph']
        self.unique_graph_dist_loss = config['save_unique_graph_dist_loss']
        self.old_graph_dist = 0

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

        self._metric_loss = get_ranking_loss(config['metric_loss'] or 'ndcg')
        self._fair_loss = get_fair_loss('dp')

        self.eval_metric = config['eval_metric'] or 'ndcg'
        self.evaluator = Evaluator(config)

        attr_map = dataset.field2id_token[self.sensitive_attribute]
        self.f_idx, self.m_idx = (attr_map == 'F').nonzero()[0][0], (attr_map == 'M').nonzero()[0][0]

        self.results = None
        self.adv_group, self.disadv_group = None, None
        self.only_adv_group = config['only_adv_group']

        self.earlys = EarlyStopping(
            config['early_stopping']['patience'],
            config['early_stopping']['ignore'],
            method=config['early_stopping']['method'],
            fn=config['early_stopping']['mode'],
            delta=config['early_stopping']['delta']
        )

        self.previous_loss_value = config['previous_loss_value']
        self.previous_batch_LR_scaling = config['previous_batch_LR_scaling']
        self.lr_scaler = None

        self.increase_disparity = config['explainer_policies']['increase_disparity']
        self.group_deletion_constraint = config['explainer_policies']['group_deletion_constraint']
        self.random_perturbation = config['explainer_policies']['random_perturbation']
        self.neighborhood_perturbation = config['explainer_policies']['neighborhood_perturbation']

        wandb.config.update(config.final_config_dict)
        # wandb.watch(self.cf_model)

    @property
    def cf_model(self):
        if self._cf_model is None:
            print("Counterfactual Model Explainer is not initialized yet. Execute 'explain' to initialize it.")
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

    def _get_model_score_data(self, batched_data, dataset, topk):
        dset_batch_data = Explainer.prepare_batched_data(batched_data, dataset)
        dset_scores_args = [dset_batch_data, self.tot_item_num, self.test_batch_size, self.item_tensor]
        self.compute_model_predictions(dset_scores_args, topk)
        dset_model_topk = self.model_topk_idx.detach().cpu().numpy()

        return dset_scores_args, dset_model_topk

    def _get_no_grad_pred_model_score_data(self, scores_args, model_topk=None, compute_dist=False):
        self.cf_model.eval()
        # When recommendations are generated passing test set the items in train and validation are considered watched
        with torch.no_grad():
            cf_scores_pred = self.get_scores(self.cf_model, *scores_args, pred=True)
            _, cf_topk_pred_idx = self.get_top_k(cf_scores_pred, **self.topk_args)
        cf_topk_pred_idx = cf_topk_pred_idx.detach().cpu().numpy()

        cf_dist = None
        if compute_dist:
            cf_dist = [
                self.dist(_pred, _topk_idx) for _pred, _topk_idx in zip(cf_topk_pred_idx, model_topk)
            ]

        return cf_topk_pred_idx, cf_dist

    def logging_exp_per_group(self, new_example, model_topk):
        em_str = self.eval_metric.upper()

        pref_data = pd.DataFrame(zip(*new_example[:2], model_topk), columns=['user_id', 'cf_topk_pred', 'topk_pred'])
        orig_res = self.compute_eval_metric(self.rec_data.dataset, pref_data, 'topk_pred')
        cf_res = self.compute_eval_metric(self.rec_data.dataset, pref_data, 'cf_topk_pred')

        user_feat = Interaction({k: v[torch.tensor(new_example[0])] for k, v in self.dataset.user_feat.interaction.items()})

        m_users = user_feat[self.sensitive_attribute] == self.m_idx
        f_users = user_feat[self.sensitive_attribute] == self.f_idx

        orig_f, orig_m = np.mean(orig_res[f_users, -1]), np.mean(orig_res[m_users, -1])
        cf_f, cf_m = np.mean(cf_res[f_users, -1]), np.mean(cf_res[m_users, -1])
        self.logger.info(f"Original => {em_str} F: {orig_f}, {em_str} M: {orig_m}, Diff: {np.abs(orig_f - orig_m)} \n"
                         f"CF       => {em_str} F: {cf_f}, {em_str} M: {cf_m}, Diff: {np.abs(cf_f - cf_m)}")

    def log_epoch(self, initial_time, epoch, *losses, **verbose_kws):
        loss_total, fair_loss, loss_graph_dist, orig_loss_graph_dist = losses
        self.logger.info(f"{self.cf_model.__class__.__name__} " +
                         f"Explain duration: {time.strftime('%H:%M:%S', time.gmtime(time.time() - initial_time))}, " +
                         # 'User id: {}, '.format(str(users_ids)) +
                         'Epoch: {}, '.format(epoch + 1) +
                         'loss: {:.4f}, '.format(loss_total.item()) +
                         'fair loss: {:.4f}, '.format(fair_loss) +
                         'graph loss: {:.4f}, '.format(loss_graph_dist.item()) +
                         'del edges: {:.4f}, '.format(int(orig_loss_graph_dist.item())))
        if self.verbose:
            self.logger.info('Orig output: {}\n'.format(self.model_scores) +
                             # 'Output: {}\n'.format(verbose_kws.get('cf_scores', None)) +
                             # 'Output nondiff: {}\n'.format(verbose_kws.get('cf_scores_pred', None)) +
                             '{:20}: {},\n {:20}: {},\n {:20}: {}\n'.format(
                                 'orig pred', self.model_topk_idx,
                                 'new pred', verbose_kws.get('cf_topk_idx', None),
                                 'new pred nondiff', verbose_kws.get('cf_topk_pred_idx', None))
                             )
        self.logger.info(" ")

    def compute_eval_metric(self, dataset, pref_data, col):
        return eval_utils.compute_metric(self.evaluator, dataset, pref_data, col, self.eval_metric)

    def compute_fair_metric(self, user_id, topk_pred, dataset, iterations=100):
        sens_attr = self.sensitive_attribute
        dset_name = dataset.dataset_name

        # minus 1 encodes the demographic groups to {0, 1} instead of {1, 2}
        pref_data = pd.DataFrame(
            zip(user_id, topk_pred, dataset.user_feat[sens_attr][user_id].numpy() - 1),
            columns=['user_id', 'cf_topk_pred', 'Demo Group']
        )

        result = eval_utils.compute_metric(self.evaluator, dataset, pref_data, 'cf_topk_pred', self.eval_metric)
        pref_data[self.eval_metric] = result[:, -1]

        # it prevents from using memoization
        if hasattr(eval_utils.compute_DP_across_random_samples, "generated_groups"):
            if (dset_name, sens_attr) in eval_utils.compute_DP_across_random_samples.generated_groups:
                del eval_utils.compute_DP_across_random_samples.generated_groups[(dset_name, sens_attr)]

        fair_metric, _ = eval_utils.compute_DP_across_random_samples(
            pref_data, sens_attr, 'Demo Group', dset_name, self.eval_metric,
            iterations=iterations, batch_size=self.user_batch_exp
        )

        return fair_metric[:, -1].mean()

    def update_best_cf_example(self,
                               best_cf_example,
                               new_example,
                               loss_total,
                               best_loss,
                               first_fair_loss,
                               model_topk=None,
                               force_update=False):
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

            if self.verbose and model_topk is not None:
                self.logging_exp_per_group(new_example, model_topk)

            if not self.unique_graph_dist_loss:
                return abs(loss_total)
        return best_loss

    def update_new_example(self,
                           new_example,
                           detached_batched_data,
                           test_scores_args,
                           test_model_topk,
                           rec_scores_args,
                           rec_model_topk,
                           epoch_fair_loss):
        test_cf_topk_pred_idx, test_cf_dist = self._get_no_grad_pred_model_score_data(
            test_scores_args, model_topk=test_model_topk, compute_dist=True
        )
        rec_cf_topk_pred_idx, rec_cf_dist = self._get_no_grad_pred_model_score_data(
            rec_scores_args, model_topk=rec_model_topk, compute_dist=True
        )

        # TODO: detached_batched_data could be saved once and not repeated for each explanation
        new_example = [
            detached_batched_data,
            # rec_model_topk,
            # test_model_topk,
            rec_cf_topk_pred_idx,
            test_cf_topk_pred_idx,
            rec_cf_dist,
            test_cf_dist,
            *new_example[4:]
        ]

        epoch_fair_metric = self.compute_fair_metric(
            detached_batched_data,
            rec_cf_topk_pred_idx,
            self.rec_data.dataset
        )

        new_example[utils.exp_col_index('fair_loss')] = epoch_fair_loss
        new_example[utils.exp_col_index('fair_metric')] = epoch_fair_metric

        wandb.log({
            'loss': epoch_fair_loss,
            'fair_metric': epoch_fair_metric,
            '# Del Edges': new_example[-2].shape[1]
        })

        return new_example, epoch_fair_metric

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
            history_item = data.uid2history_item[data_df[data.dataset.uid_field]]
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
            batched_attr_data = self.dataset.user_feat[self.sensitive_attribute][batched_data]
            iter_data = batched_data[batched_attr_data == self.adv_group].split(self.user_batch_exp)

        return iter_data

    def _prepare_test_history_matrix(self, test_data, topk=10):
        uids = test_data.dataset.user_feat.interaction[test_data.dataset.uid_field][1:]

        dset_scores_args, dset_model_topk = self._get_model_score_data(uids, test_data, topk)

        # add -1 as item ids for the padding user
        dset_model_topk = np.vstack((np.array([-1] * topk, dtype=dset_model_topk.dtype), dset_model_topk))
        self._test_history_matrix = dset_model_topk

        return dset_scores_args, dset_model_topk

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

    def _check_loss_trend_epoch_images(self):
        cwd_files = [f for f in os.listdir() if f.startswith('loss_trend_epoch')]
        if len(cwd_files) > 0 and os.path.isfile(cwd_files[0]) and cwd_files[0][-3:] == 'png':
            os.remove(cwd_files[0])

    def _check_policies(self, batched_data, rec_model_topk):
        filtered_users = None
        test_model_topk, test_scores_args, rec_scores_args = [None] * 3
        if self.increase_disparity:
            batched_data, test_model_topk, test_scores_args,\
                rec_model_topk, rec_scores_args = self.increase_dataset_unfairness(
                    batched_data,
                    test_data,
                    rec_model_topk,
                    topk=topk
                )

            filtered_users = batched_data

        self.determine_adv_group(batched_data, rec_model_topk)

        if self.group_deletion_constraint and self.random_perturbation:
            raise NotImplementedError(
                'The policies `group_deletion_constraint` and `random_perturbation` cannot be both True'
            )
        elif self.group_deletion_constraint:
            if filtered_users is None:
                filtered_users = batched_data

            filtered_users = filtered_users[
                self.dataset.user_feat[self.sensitive_attribute][filtered_users] == self.adv_group
            ]
        elif self.random_perturbation:
            # overwrites `increase_disparity` policy
            filtered_users = exp_models.PerturbedModel.RANDOM_POLICY

        return batched_data, filtered_users, (test_model_topk, test_scores_args, rec_model_topk, rec_scores_args)

    def _check_previous_loss_value(self):
        previous_loss_value = None
        if self.previous_loss_value:
            previous_loss_value = dict.fromkeys(self.dataset.user_feat[self.sensitive_attribute].unique().numpy())
            for gr in previous_loss_value:
                previous_loss_value[gr] = np.array([])

        return previous_loss_value

    def _check_early_stopping(self, check_value, *update_best_example_args):
        if self.earlys.check(check_value):
            self.logger.info(self.earlys)
            best_epoch = epoch + 1 - self.earlys.patience
            self.logger.info(f"Early Stopping: best epoch {best_epoch}")

            # stub example added to find again the best epoch when explanations are loaded
            self.update_best_cf_example(*update_best_example_args, force_update=True)

            return True
        return False

    def determine_adv_group(self, batched_data, rec_model_topk):
        pref_users = batched_data.numpy()
        pref_data = pd.DataFrame(zip(pref_users, rec_model_topk.tolist()), columns=['user_id', 'topk_pred'])

        orig_res = self.compute_eval_metric(self.rec_data.dataset, pref_data, 'topk_pred')

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

        orig_res = self.compute_eval_metric(self.rec_data.dataset, pref_data, 'topk_pred')

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
        test_scores_args, test_model_topk = self._get_model_score_data(batched_data, test_data, topk)
        rec_scores_args, rec_model_topk = self._get_model_score_data(batched_data, self.rec_data, topk)

        return batched_data, test_model_topk, test_scores_args, rec_model_topk, rec_scores_args

    def run_epoch(self, epoch, batched_data, fair_losses, topk=10):
        iter_data = self.prepare_iter_batched_data(batched_data)
        iter_data = [iter_data[i] for i in np.random.permutation(len(iter_data))]

        previous_loss_value = self._check_previous_loss_value()

        if not self.mini_batch_descent:
            self.cf_optimizer.zero_grad()

        epoch_fair_loss = []
        new_example, loss_total = None, None
        for batch_idx, batch_user in enumerate(iter_data):
            if self.previous_batch_LR_scaling:
                self.lr_scaler.update(batch_idx)

            batch_scores_args, _ = self._get_model_score_data(batch_user, self.rec_data, topk)

            torch.cuda.empty_cache()
            new_example, loss_total, fair_loss = self.train(
                epoch, batch_scores_args, batch_user, topk=topk, previous_loss_value=previous_loss_value
            )
            epoch_fair_loss.append(fair_loss)

        if not self.mini_batch_descent:
            self.cf_optimizer.step()

        if self.previous_batch_LR_scaling:
            self.lr_scaler.restore()

        epoch_fair_loss = np.mean(epoch_fair_loss)
        fair_losses.append(epoch_fair_loss)

        return new_example, loss_total, epoch_fair_loss

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

        self._check_loss_trend_epoch_images()

        test_scores_args, test_model_topk = self._get_model_score_data(batched_data, test_data, topk)

        # recommendations generated by the model are considered the ground truth
        if self._pred_as_rec:
            rec_scores_args, rec_model_topk = self._prepare_test_history_matrix(test_data, topk=topk)
        else:
            rec_scores_args, rec_model_topk = self._get_model_score_data(batched_data, self.rec_data, topk)

        batched_data, filtered_users, inc_disp_model_data = self._check_policies(batched_data, rec_model_topk)
        if self.increase_disparity:
            test_model_topk, test_scores_args, rec_model_topk, rec_scores_args = inc_disp_model_data

        # logs of fairness consider validation as seen when model recommendations are used as ground truth
        if self._pred_as_rec:
            self.rec_data = test_data

        detached_batched_data = batched_data.detach().numpy()
        self.initialize_cf_model(filtered_users=filtered_users)

        orig_loss = np.abs(self.results[self.adv_group] - self.results[self.disadv_group])
        self.logger.info(self.results)
        self.logger.info(f"M idx: {self.m_idx}")
        self.logger.info(f"Original Fair Loss: {orig_loss}")

        iter_epochs = tqdm.tqdm(
            range(epochs),
            total=epochs,
            ncols=100,
            desc=set_color(f"Epochs   ", 'blue'),
        )

        if self.previous_batch_LR_scaling:
            iter_data = self.prepare_iter_batched_data(batched_data)
            self.lr_scaler = exp_utils.LRScaler(self.cf_optimizer, len(iter_data))

        fair_losses = []
        for epoch in iter_epochs:
            new_example, loss_total, epoch_fair_loss = self.run_epoch(epoch, batched_data, fair_losses, topk=topk)

            if self.verbose:
                Explainer._verbose_plot(fair_losses, epoch)

            if new_example is not None:
                new_example, epoch_fair_metric = self.update_new_example(
                    new_example,
                    detached_batched_data,
                    test_scores_args,
                    test_model_topk,
                    rec_scores_args,
                    rec_model_topk,
                    epoch_fair_loss
                )

                earlys_check_value = epoch_fair_loss
                if self._pred_as_rec and earlys_check_value == epoch_fair_metric:
                    raise ValueError(f"`exp_rec_data` = `rec` stores test data to log fairness metric. "
                                     f"Cannot be used as value for early stopping check")

                update_best_example_args = [best_cf_example, new_example, loss_total, best_loss, orig_loss]
                if self._check_early_stopping(earlys_check_value, *update_best_example_args):
                    break

                best_loss = self.update_best_cf_example(*update_best_example_args, model_topk=rec_model_topk)

            self.logger.info("{} CF examples".format(len(best_cf_example)))

        return best_cf_example, (rec_model_topk, test_model_topk)

    def train(self, epoch, scores_args, users_ids, topk=10, previous_loss_value=None):
        """
        Training procedure of explanation
        :param epoch:
        :param topk:
        :return:
        """
        t = time.time()

        # `_get_no_grad_pred_model_score_data` call torch `eval` inside
        cf_topk_pred_idx, _ = self._get_no_grad_pred_model_score_data(scores_args)

        if self.mini_batch_descent:
            self.cf_optimizer.zero_grad()
        self.cf_model.train()

        # compute differentiable permutation of adj matrix
        # cf_scores uses differentiable P_hat ---> adjacency matrix not binary, but needed for training
        cf_scores = self.get_scores(self.cf_model, *scores_args, pred=False)

        # remove neginf from output
        cf_scores = torch.nan_to_num(cf_scores, neginf=(torch.min(cf_scores[~torch.isinf(cf_scores)]) - 1).item())
        cf_scores_topk, cf_topk_idx = self.get_top_k(cf_scores, **self.topk_args)

        user_feat = self.get_batch_user_feat(users_ids)

        target = self.get_target(cf_scores, user_feat)

        loss_total, orig_loss_graph_dist, loss_graph_dist, fair_loss, adj_sub_cf_adj = self.cf_model.loss(
            cf_scores,
            self._fair_loss(
                self.sensitive_attribute,
                user_feat,
                topk=topk,
                loss=self._metric_loss,
                adv_group_data=(self.only_adv_group, self.disadv_group, self.results[self.disadv_group]),
                previous_loss_value=previous_loss_value
            ),
            target
        )

        torch.cuda.empty_cache()
        loss_total.backward()

        # import pdb; pdb.set_trace()
        # for name, param in self.cf_model.named_parameters():
        #     if name == "P_symm":
        #         print(param.grad[param.grad])

        torch.nn.utils.clip_grad_norm_(self.cf_model.parameters(), 2.0)
        if self.mini_batch_descent:
            self.cf_optimizer.step()

        fair_loss = fair_loss.mean().item() if fair_loss is not None else torch.nan
        self.log_epoch(
            t, epoch, loss_total, fair_loss, loss_graph_dist, orig_loss_graph_dist,
            **dict(cf_topk_idx=cf_topk_idx, cf_topk_pred_idx=cf_topk_pred_idx)
        )

        cf_stats = None
        if orig_loss_graph_dist.item() > 0:
            cf_stats = self.get_batch_cf_stats(
                users_ids, adj_sub_cf_adj, cf_topk_pred_idx, loss_total, loss_graph_dist, fair_loss, epoch
            )

        return cf_stats, loss_total.item(), fair_loss

    def get_batch_user_feat(self, users_ids):
        user_feat = self.dataset.user_feat
        user_id_mask = users_ids.unsqueeze(-1) if users_ids.dim() == 0 else users_ids
        return {k: feat[user_id_mask] for k, feat in user_feat.interaction.items()}

    def get_target(self, cf_scores, user_feat):
        target = torch.zeros_like(cf_scores, dtype=torch.float, device=cf_scores.device)
        if not self._pred_as_rec:
            hist_matrix, _, _ = self.rec_data.dataset.history_item_matrix()
            rec_data_interactions = hist_matrix[user_feat[self.dataset.uid_field]]
        else:
            rec_data_interactions = self._test_history_matrix[user_feat[self.dataset.uid_field]]
        target[torch.arange(target.shape[0])[:, None], rec_data_interactions] = 1
        target[:, 0] = 0

        return target

    def get_batch_cf_stats(self, u_ids, adj_sub_cf_adj, cf_topk, loss_total, loss_graph_dist, fair_loss, epoch):
        # Compute distance between original and perturbed list. Explanation maintained only if dist > 0
        # cf_dist = [self.dist(_pred, _topk_idx) for _pred, _topk_idx in zip(cf_topk_pred_idx, self.model_topk_idx)]
        cf_dist = None

        adj_del_edges = adj_sub_cf_adj.detach().cpu()
        del_edges = adj_del_edges.indices()[:, adj_del_edges.values().nonzero().squeeze()]

        # remove duplicated edges
        del_edges = del_edges[:, (del_edges[0, :] < self.dataset.user_num) & (del_edges[0, :] > 0)].numpy()

        cf_stats = [
            u_ids.detach().numpy(), self.model_topk_idx.detach().cpu().numpy(), cf_topk,
            cf_dist, loss_total.item(), loss_graph_dist.item(), fair_loss, None, del_edges, epoch + 1
        ]

        if self.neighborhood_perturbation:
            self.cf_model.update_neighborhood(torch.Tensor(del_edges))

        return cf_stats

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
                scores = Explainer._spilt_predict(new_inter, batch_size, test_batch_size, test_batch_size)

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
