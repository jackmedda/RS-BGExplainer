# Based on https://github.com/RexYing/gnn-model-explainer/blob/master/explainer/explain.py

import os
import sys
import time
import math
from logging import getLogger

import wandb
import gmpy2
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.signal as sp_signal
import matplotlib.pyplot as plt
from tqdm import tqdm
from recbole.evaluator import Evaluator
from recbole.utils import set_color
from recbole.data.interaction import Interaction

import biga.utils as utils
import biga.models as exp_models
import biga.evaluation as eval_utils
from biga.utils.early_stopping import EarlyStopping
from biga.losses import get_ranking_loss, get_fair_loss

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
        self.fair_metric = config['fair_metric'] or 'DP_across_random_samples'

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
        self.earlys_check_value = config['early_stopping']['check_value']

        self.previous_loss_value = config['previous_loss_value']
        self.previous_batch_LR_scaling = config['previous_batch_LR_scaling']
        self.lr_scaler = None

        # Init policies
        self.increase_disparity = config['explainer_policies']['increase_disparity']
        self.group_deletion_constraint = config['explainer_policies']['group_deletion_constraint']
        self.random_perturbation = config['explainer_policies']['random_perturbation']
        self.neighborhood_perturbation = config['explainer_policies']['neighborhood_perturbation']
        self.users_zero_constraint = config['explainer_policies']['users_zero_constraint']
        self.users_zero_constraint_value = config['users_zero_constraint_value'] or 0
        self.users_low_degree = config['explainer_policies']['users_low_degree']
        self.users_low_degree_value = config['users_low_degree_value'] or 0.35
        self.items_preference_constraint = config['explainer_policies']['items_preference_constraint']
        self.items_preference_constraint_ratio = config['items_preference_constraint_ratio'] or 0.2
        self.users_furthest_constraint = config['explainer_policies']['users_furthest_constraint']
        self.users_furthest_constraint_ratio = config['users_furthest_constraint_ratio'] or 0.35
        self.sparse_users_constraint = config['explainer_policies']['sparse_users_constraint']
        self.sparse_users_constraint_ratio = config['sparse_users_constraint_ratio'] or 0.35
        self.niche_items_constraint = config['explainer_policies']['niche_items_constraint']
        self.niche_items_constraint_ratio = config['niche_items_constraint_ratio'] or 0.2

        self.ckpt_loading_path = None

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

    def set_checkpoint_path(self, path):
        self.ckpt_loading_path = path

    def _resume_checkpoint(self):
        ckpt = torch.load(self.ckpt_loading_path)
        epoch = ckpt['starting_epoch']
        fair_losses = ckpt['fair_losses']
        self.earlys = ckpt['early_stopping']
        best_cf_example = ckpt['best_cf_example']
        self.cf_model.load_cf_state_dict(ckpt)

        return fair_losses, epoch, best_cf_example

    def _save_checkpoint(self, epoch, fair_losses, best_cf_example):
        cf_state_dict = self.cf_model.cf_state_dict()
        ckpt = {
            'starting_epoch': epoch,
            'fair_losses': fair_losses,
            'early_stopping': self.earlys,
            'best_cf_example': best_cf_example,
            **cf_state_dict
        }
        torch.save(ckpt, self.ckpt_loading_path)

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

    def _get_scores_args(self, batched_data, dataset):
        dset_batch_data = Explainer.prepare_batched_data(batched_data, dataset)
        return [dset_batch_data, self.tot_item_num, self.test_batch_size, self.item_tensor]

    def _get_model_score_data(self, batched_data, dataset, topk):
        dset_scores_args = self._get_scores_args(batched_data, dataset)
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
        if compute_dist and model_topk is not None:
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
                         'perturbed edges: {:.4f}, '.format(int(orig_loss_graph_dist.item())))
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

    def update_best_cf_example(self,
                               best_cf_example,
                               new_example,
                               loss_total,
                               best_loss,
                               model_topk=None,
                               force_update=False):
        """
        Updates the explanations with new explanation (if not None) depending on new loss value
        :param best_cf_example:
        :param new_example:
        :param loss_total:
        :param best_loss:
        :return:
        """
        if force_update or (new_example is not None and (abs(loss_total) < best_loss or self.unique_graph_dist_loss)):
            if self.unique_graph_dist_loss and len(best_cf_example) > 0:
                self.old_graph_dist = best_cf_example[-1][1]
                new_graph_dist = new_example[1]
                if not force_update and not (self.old_graph_dist != new_graph_dist):
                    return best_loss

            best_cf_example.append(new_example)
            self.old_graph_dist = new_example[1]

            if self.verbose and model_topk is not None:
                self.logging_exp_per_group(new_example, model_topk)

            if not self.unique_graph_dist_loss:
                return abs(loss_total)
        return best_loss

    def update_new_example(self,
                           new_example,
                           detached_batched_data,
                           full_dataset,
                           train_data,
                           valid_data,
                           test_data,
                           test_model_topk,
                           rec_model_topk,
                           epoch_fair_loss):
        perturbed_edges = new_example[-2]

        if self.config['exp_rec_data'] == 'rec':
            raise NotImplementedErorr(
                'fair metric evaluation with dataloaders with perturbed edges not implemented for `exp_rec_data` == `rec`'
            )

        pert_sets = utils.get_dataloader_with_perturbed_edges(
            perturbed_edges, self.config, full_dataset, train_data, valid_data, test_data
        )
        pert_sets_dict = dict(zip(['train', 'valid', 'test'], pert_sets))

        test_scores_args = self._get_scores_args(detached_batched_data, pert_sets_dict['test'])
        rec_scores_args = self._get_scores_args(detached_batched_data, pert_sets_dict[self.config['exp_rec_data']])

        test_cf_topk_pred_idx, test_cf_dist = self._get_no_grad_pred_model_score_data(
            test_scores_args, model_topk=test_model_topk, compute_dist=True
        )
        rec_cf_topk_pred_idx, rec_cf_dist = self._get_no_grad_pred_model_score_data(
            rec_scores_args, model_topk=rec_model_topk, compute_dist=True
        )

        # new_example = [
        #     detached_batched_data,
        #     # rec_model_topk,
        #     # test_model_topk,
        #     rec_cf_topk_pred_idx,
        #     test_cf_topk_pred_idx,
        #     rec_cf_dist,
        #     test_cf_dist,
        #     *new_example[4:]
        # ]

        epoch_rec_fair_metric = self.compute_fair_metric(
            detached_batched_data,
            rec_cf_topk_pred_idx,
            pert_sets_dict[self.config['exp_rec_data']].dataset
        )

        epoch_test_fair_metric = self.compute_fair_metric(
            detached_batched_data,
            test_cf_topk_pred_idx,
            pert_sets_dict['test'].dataset
        )

#         def pert_edges_mapper(pe, rec_dset):
#             return pe

#         test_pert_df, valid_pert_df = eval_utils.extract_metrics_from_perturbed_edges(
#             {(self.dataset.dataset_name, self.sensitive_attribute): perturbed_edges},
#             models=[self.model.__class__.__name__],
#             metrics=[self.eval_metric],
#             models_path=os.path.join(os.getcwd(), 'saved'),
#             remap=pert_edges_mapper
#         )

        new_example[utils.exp_col_index('fair_loss')] = epoch_fair_loss
        new_example[utils.exp_col_index('fair_metric')] = epoch_rec_fair_metric

        wandb.log({
            'loss': epoch_fair_loss,
            'rec_fair_metric': epoch_rec_fair_metric,
            'test_fair_metric': epoch_test_fair_metric,
            '# Del Edges': perturbed_edges,
            'epoch': new_example[-1]
        })

        return new_example, epoch_rec_fair_metric

    @staticmethod
    def prepare_batched_data(batched_data, data, item_data=None):
        return utils.prepare_batched_data(batched_data, data, item_data=item_data)

    def get_iter_data(self, user_data):
        user_data = user_data.split(self.user_batch_exp)

        return (
            tqdm(
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
        filtered_items = None
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

        self.determine_adv_group(batched_data.detach().numpy(), rec_model_topk)

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

            if self.users_furthest_constraint:
                disadv_users = batched_data[
                    self.dataset.user_feat[self.sensitive_attribute][batched_data] == self.disadv_group
                ].numpy()

                igg = eval_utils.get_bipartite_igraph(self.dataset, remove_first_row_col=True)
                mean_dist = np.array(igg.distances(source=filtered_users, target=disadv_users)).mean(axis=1)
                furthest_users = np.argsort(mean_dist)

                filtered_users = filtered_users[
                    furthest_users[-int(self.users_furthest_constraint_ratio * furthest_users.shape[0]):]
                ]
        elif self.random_perturbation:
            # overwrites `increase_disparity` policy
            filtered_users = exp_models.PerturbedModel.RANDOM_POLICY

        if self.users_zero_constraint and self.random_perturbation:
            raise NotImplementedError(
                'The policies `users_zero_constraint` and `random_perturbation` cannot be both True'
            )
        elif self.users_zero_constraint:
            if filtered_users is None:
                filtered_users = batched_data

            pref_data = self._pref_data_sens_and_metric(batched_data.detach().numpy(), rec_model_topk)
            zero_users = pref_data.loc[(pref_data[self.eval_metric] <= self.users_zero_constraint_value), 'user_id']
            filtered_users = torch.from_numpy(np.intersect1d(zero_users.to_numpy(), filtered_users.numpy()))

        if self.users_low_degree and self.random_perturbation:
            raise NotImplementedError(
                'The policies `users_low_degree` and `random_perturbation` cannot be both True'
            )
        elif self.users_low_degree:
            if filtered_users is None:
                filtered_users = batched_data

            _, _, hist_len = self.dataset.history_item_matrix()
            hist_len = hist_len[filtered_users]

            lowest_degree = torch.argsort(hist_len)[:int(self.users_low_degree_value * hist_len.shape[0])]
            filtered_users = filtered_users[lowest_degree]

        # sparse users are connected to low-degree items
        if self.sparse_users_constraint and self.random_perturbation:
            raise NotImplementedError(
                'The policies `sparse_users_constraint` and `random_perturbation` cannot be both True'
            )
        elif self.sparse_users_constraint:
            if filtered_users is None:
                filtered_users = batched_data

            sparsity_df = eval_utils.extract_graph_metrics_per_node(
                self.dataset, remove_first_row_col=True, metrics=["Sparsity"]
            )
            sparsity = torch.from_numpy(
                sparsity_df.set_index('Node').loc[filtered_users.numpy(), 'Sparsity'].to_numpy()
            )

            most_sparse = torch.argsort(sparsity)[-int(self.sparse_users_constraint_ratio * sparsity.shape[0]):]
            filtered_users = filtered_users[most_sparse]

        if self.items_preference_constraint:
            ihist_m, _, ihist_len = self.dataset.history_user_matrix()

            sens_map = self.dataset.user_feat[self.sensitive_attribute]
            n_adv = (sens_map == self.adv_group).sum() / (sens_map.shape[0] - 1)

            sens_ihist_m = sens_map[ihist_m]
            adv_ratio = torch.nan_to_num((sens_ihist_m == self.adv_group).sum(dim=1) / ihist_len, nan=0)
            # values higher than 1 means the advantaged group prefer those items w.r.t. to their representation
            adv_ratio = adv_ratio / n_adv

            filtered_items = torch.argsort(adv_ratio)[-int(self.items_preference_constraint_ratio * adv_ratio.shape[0]):]

        if self.niche_items_constraint:
            if filtered_items is None:
                filtered_items = torch.arange(1, self.dataset.item_num)
            _, _, ihist_len = self.dataset.history_user_matrix()
            ihist_len = ihist_len[filtered_items]

            filtered_items = filtered_items[
                torch.argsort(ihist_len)[:int(self.niche_items_constraint_ratio * ihist_len.shape[0])]
            ]

        return batched_data, filtered_users, filtered_items, (test_model_topk, test_scores_args, rec_model_topk, rec_scores_args)

    def _check_previous_loss_value(self):
        previous_loss_value = None
        if self.previous_loss_value:
            previous_loss_value = dict.fromkeys(self.dataset.user_feat[self.sensitive_attribute].unique().numpy())
            for gr in previous_loss_value:
                previous_loss_value[gr] = np.array([])

        return previous_loss_value

    def _check_early_stopping(self, check_value, epoch, *update_best_example_args):
        if self.earlys.check(check_value):
            self.logger.info(self.earlys)
            best_epoch = epoch + 1 - self.earlys.patience
            self.logger.info(f"Early Stopping: best epoch {best_epoch}")

            # stub example added to find again the best epoch when explanations are loaded
            self.update_best_cf_example(*update_best_example_args, force_update=True)

            return True
        return False

    def _pref_data_sens_and_metric(self, pref_users, model_topk, eval_data=None):
        pref_data = pd.DataFrame(
            zip(pref_users, model_topk, self.dataset.user_feat[self.sensitive_attribute][pref_users].numpy()),
            columns=['user_id', 'topk_pred', 'Demo Group']
        )
        pref_data[self.eval_metric] = self.compute_eval_metric(eval_data or self.rec_data.dataset, pref_data, 'topk_pred')[:, -1]

        return pref_data

    def compute_eval_metric(self, dataset, pref_data, col):
        return eval_utils.compute_metric(self.evaluator, dataset, pref_data, col, self.eval_metric)

    def compute_eval_result(self, pref_users: np.ndarray, model_topk: np.ndarray, eval_data=None):
        return self._pref_data_sens_and_metric(pref_users, model_topk, eval_data=eval_data)[self.eval_metric].to_numpy()

    def compute_f_m_result(self, batched_data: np.ndarray, model_topk, eval_data=None):
        pref_data = self._pref_data_sens_and_metric(batched_data, model_topk, eval_data=eval_data)

        f_result = pref_data.loc[pref_data['Demo Group'] == self.f_idx, self.eval_metric].mean()
        m_result = pref_data.loc[pref_data['Demo Group'] == self.m_idx, self.eval_metric].mean()

        return f_result, m_result

    def compute_fair_metric(self, pref_users, model_topk, dataset, iterations=100):
        pref_data = self._pref_data_sens_and_metric(pref_users, model_topk, eval_data=dataset)

        return exp_utils.get_fair_metric_value(
            self.fair_metric,
            pref_data,
            self.eval_metric,
            dataset.dataset_name,
            self.sensitive_attribute,
            self.user_batch_exp,
            iterations=iterations
        )

    def determine_adv_group(self, batched_data: np.ndarray, rec_model_topk):
        f_result, m_result = self.compute_f_m_result(batched_data, rec_model_topk)

        check_func = "__ge__" if self.config['perturb_adv_group'] else "__lt__"

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
        self._fair_loss.update_previous_loss_value(previous_loss_value)

        if not self.mini_batch_descent:
            self.cf_optimizer.zero_grad()

        epoch_fair_loss = []
        new_example, loss_total = None, None
        for batch_idx, batch_user in enumerate(iter_data):
            if self.previous_batch_LR_scaling:
                self.lr_scaler.update(batch_idx)

            batch_scores_args = self._get_scores_args(batch_user, self.rec_data)

            torch.cuda.empty_cache()
            new_example, loss_total, fair_loss = self.train(
                epoch, batch_scores_args, batch_user, topk=topk, previous_loss_value=previous_loss_value
            )
            epoch_fair_loss.append(fair_loss)

            if batch_idx != len(iter_data) - 1:
                torch.nn.utils.clip_grad_norm_(self.cf_model.parameters(), 2.0)
                if self.mini_batch_descent:
                    self.cf_optimizer.step()

        if self.previous_batch_LR_scaling:
            self.lr_scaler.restore()

        epoch_fair_loss = np.mean(epoch_fair_loss)
        fair_losses.append(epoch_fair_loss)

        return new_example, loss_total, epoch_fair_loss

    def explain(self, batched_data, full_dataset, train_data, valid_data, test_data, epochs, topk=10):
        """
        The method from which starts the perturbation of the graph by optimization of `pred_loss` or `fair_loss`
        :param batched_data:
        :param test_data:
        :param epochs:
        :param topk:
        :return:
        """
        best_loss = np.inf

        self._check_loss_trend_epoch_images()

        test_scores_args, test_model_topk = self._get_model_score_data(batched_data, test_data, topk)

        # recommendations generated by the model are considered the ground truth
        if self._pred_as_rec:
            rec_scores_args, rec_model_topk = self._prepare_test_history_matrix(test_data, topk=topk)
        else:
            rec_scores_args, rec_model_topk = self._get_model_score_data(batched_data, self.rec_data, topk)

        batched_data, filtered_users, filtered_items, inc_disp_model_data = self._check_policies(batched_data, rec_model_topk)
        if self.increase_disparity:
            test_model_topk, test_scores_args, rec_model_topk, rec_scores_args = inc_disp_model_data

        self._fair_loss = self._fair_loss(
            self.sensitive_attribute,
            topk=topk,
            loss=self._metric_loss,
            adv_group_data=(self.only_adv_group, self.disadv_group, self.results[self.disadv_group])
        )

        # logs of fairness consider validation as seen when model recommendations are used as ground truth
        if self._pred_as_rec:
            self.rec_data = test_data

        detached_batched_data = batched_data.detach().numpy()
        self.initialize_cf_model(filtered_users=filtered_users, filtered_items=filtered_items)

        if self.ckpt_loading_path is not None and os.path.exists(self.ckpt_loading_path):
            fair_losses, starting_epoch, best_cf_example = self._resume_checkpoint()
            last_earlys_check_value = self.earlys.history.pop()
            if self.earlys.check(last_earlys_check_value):
                raise AttributeError("A checkpoint of a completed run cannot be resumed")
        else:
            starting_epoch = 0
            fair_losses = []
            best_cf_example = []

        orig_rec_dp = eval_utils.compute_DP(
            self.results[self.adv_group], self.results[self.disadv_group]
        )
        orig_test_dp = eval_utils.compute_DP(
            *self.compute_f_m_result(detached_batched_data, test_model_topk, eval_data=test_data.dataset)
        )
        self.logger.info("*********** Rec Data ***********")
        self.logger.info(self.results)
        self.logger.info(f"M idx: {self.m_idx}")
        self.logger.info(f"Original Rec Fairness: {orig_rec_dp}")
        self.logger.info("*********** Test Data ***********")
        self.logger.info(f"Original Test Fairness: {orig_test_dp}")

        iter_epochs = tqdm(
            range(starting_epoch, epochs),
            total=epochs,
            ncols=100,
            initial=starting_epoch,
            desc=set_color(f"Epochs   ", 'blue'),
        )

        if self.previous_batch_LR_scaling:
            iter_data = self.prepare_iter_batched_data(batched_data)
            self.lr_scaler = exp_utils.LRScaler(self.cf_optimizer, len(iter_data))

        for epoch in iter_epochs:
            new_example, loss_total, epoch_fair_loss = self.run_epoch(epoch, batched_data, fair_losses, topk=topk)

            if self.verbose:
                Explainer._verbose_plot(fair_losses, epoch)

            if new_example is not None:
                new_example, epoch_fair_metric = self.update_new_example(
                    new_example,
                    detached_batched_data,
                    full_dataset,
                    train_data,
                    valid_data,
                    test_data,
                    test_model_topk,
                    rec_model_topk,
                    epoch_fair_loss
                )

                earlys_check_value = {
                    'fair_loss': epoch_fair_loss,
                    'fair_metric': epoch_fair_metric
                }[self.earlys_check_value]
                if self._pred_as_rec and earlys_check_value == epoch_fair_metric:
                    raise ValueError(f"`exp_rec_data` = `rec` stores test data to log fairness metric. "
                                     f"Cannot be used as value for early stopping check")

                update_best_example_args = [best_cf_example, new_example, loss_total, best_loss]
                earlys_check = self._check_early_stopping(earlys_check_value, epoch, *update_best_example_args)
                print("*" * 7 + " Early Stopping History " + "*" * 7)
                print(self.earlys.history)
                print("*" * 7 + "************************" + "*" * 7)
                if self.ckpt_loading_path is not None:
                    if not earlys_check:
                        # epoch + 1 because the current one is already finished
                        self._save_checkpoint(epoch + 1, fair_losses, best_cf_example)

                if earlys_check:
                    break

                best_loss = self.update_best_cf_example(*update_best_example_args, model_topk=rec_model_topk)

            # the optimizer step of the last epoch is done here to prevent computations
            # done to update new example to be related to the new state of the model
            torch.nn.utils.clip_grad_norm_(self.cf_model.parameters(), 2.0)
            if self.mini_batch_descent:
                self.cf_optimizer.step()

            self.logger.info("{} CF examples".format(len(best_cf_example)))

        return best_cf_example, detached_batched_data, (rec_model_topk, test_model_topk)

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
        self._fair_loss.update_user_feat(user_feat)

        target = self.get_target(cf_scores, user_feat)

        loss_total, orig_loss_graph_dist, loss_graph_dist, fair_loss, adj_sub_cf_adj = self.cf_model.loss(
            cf_scores,
            self._fair_loss,
            target
        )

        torch.cuda.empty_cache()
        loss_total.backward()

        # for name, param in self.cf_model.named_parameters():
        #     if name == "P_symm":
        #         print(param.grad)
        # import pdb; pdb.set_trace()

        fair_loss = fair_loss.mean().item() if fair_loss is not None else torch.nan
        self.log_epoch(
            t, epoch, loss_total, fair_loss, loss_graph_dist, orig_loss_graph_dist,
            **dict(cf_topk_idx=cf_topk_idx, cf_topk_pred_idx=cf_topk_pred_idx)
        )

        cf_stats = None
        if orig_loss_graph_dist.item() > 0:
            cf_stats = self.get_batch_cf_stats(
                adj_sub_cf_adj, loss_total, loss_graph_dist, fair_loss, epoch
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
        target[:, 0] = 0  # item 0 is a padding item

        return target

    def get_batch_cf_stats(self, adj_sub_cf_adj, loss_total, loss_graph_dist, fair_loss, epoch):
        # Compute distance between original and perturbed list. Explanation maintained only if dist > 0
        # cf_dist = [self.dist(_pred, _topk_idx) for _pred, _topk_idx in zip(cf_topk_pred_idx, self.model_topk_idx)]
        cf_dist = None

        adj_pert_edges = adj_sub_cf_adj.detach().cpu()
        pert_edges = adj_pert_edges.indices()[:, adj_pert_edges.values().nonzero().squeeze()]

        # remove duplicated edges
        pert_edges = pert_edges[:, (pert_edges[0, :] < self.dataset.user_num) & (pert_edges[0, :] > 0)].numpy()

        cf_stats = [loss_total.item(), loss_graph_dist.item(), fair_loss, None, pert_edges, epoch + 1]

        if self.neighborhood_perturbation:
            self.cf_model.update_neighborhood(torch.Tensor(pert_edges))

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
        return exp_utils.get_scores(_model, batched_data, tot_item_num, test_batch_size, item_tensor, pred=pred)

    @staticmethod
    def get_top_k(scores_tensor, topk=10):
        return utils.get_top_k(scores_tensor, topk=topk)
