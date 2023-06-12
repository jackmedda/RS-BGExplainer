import time

import tqdm
import torch
import numpy as np
import pandas as pd
from recbole.utils import set_color

import biga.evaluation as eval_utils

from .base import Explainer


class BaB(Explainer):
    """
    BaB: Batch after Batch
    """
    def __init__(self, config, dataset, rec_data, model, dist="damerau_levenshtein", **kwargs):
        super(BaB, self).__init__(config, dataset, rec_data, model, dist=dist, **kwargs)
        self.mini_batch_descent = None
        self.min_del_edges_batch = config['bab_min_del_edges'] or 1
        self.max_tries_batch = config['bab_max_tries'] or 20

    def compute_rec_test_eval_metric_model_and_cf(self, new_example, test_data, test_model_topk, cf_test_topk):
        test_pref = pd.DataFrame(
            zip(new_example[0], test_model_topk, cf_test_topk),
            columns=['user_id', 'topk_pred', 'cf_topk_pred']
        )
        orig_test_res = eval_utils.compute_metric(
            self.evaluator, test_data, test_pref, 'topk_pred', self.eval_metric
        )
        cf_test_res = eval_utils.compute_metric(
            self.evaluator, test_data, test_pref, 'cf_topk_pred', self.eval_metric
        )

        rec_pref = pd.DataFrame(
            zip(*new_example[:3]), columns=['user_id', 'topk_pred', 'cf_topk_pred']
        )
        orig_rec_res = self.compute_eval_metric(self.rec_data.dataset, rec_pref, 'topk_pred')
        cf_rec_res = self.compute_eval_metric(self.rec_data.dataset, rec_pref, 'cf_topk_pred')

        return orig_test_res, cf_test_res, orig_rec_res, cf_rec_res

    def run_epoch(self, epoch, batched_data, test_data, fair_losses, topk=10):
        iter_data = self.prepare_iter_batched_data(batched_data)
        iter_data = [iter_data[i] for i in np.random.permutation(len(iter_data))]

        bab_epoch_data = []
        epoch_fair_loss = []
        for batch_idx, batch_user in enumerate(iter_data):
            new_example, loss_total = None, None
            batch_scores_args, _ = self._get_model_score_data(batch_user, self.rec_data, topk)

            def batch_check(ne, tc):
                return ne is None or (ne[-2].shape[1] < self.min_del_edges_batch and tc < self.max_tries_batch)

            tries_counter = 0
            del_edges_batch_first_iter = False

            while batch_check(new_example, tries_counter):
                torch.cuda.empty_cache()
                new_example, loss_total, fair_loss = self.train(
                    epoch, batch_scores_args, batch_user, topk=topk, previous_loss_value=None
                )

                if batch_check(new_example, tries_counter):
                    loss_total.backward()
                    torch.nn.utils.clip_grad_norm_(self.cf_model.parameters(), 2.0)
                    self.cf_optimizer.step()

                    del_edges_batch_first_iter = True
                tries_counter += 1

            if del_edges_batch_first_iter:
                self.cf_model.reset_param()

            epoch_fair_loss.append(fair_loss)

            if new_example is not None:
                uids = new_example[0]
                items_involved = np.asarray(np.unique(new_example[-2][1], return_counts=True))

                batch_test_scores_args, test_model_topk_idx = self._get_model_score_data(uids, test_data, topk)
                cf_test_topk_pred_idx, _ = self._get_no_grad_pred_model_score_data(batch_test_scores_args)

                orig_test_res, cf_test_res, orig_rec_res, cf_rec_res = self.compute_rec_test_eval_metric_model_and_cf(
                    new_example, test_data.dataset, test_model_topk_idx, cf_test_topk_pred_idx
                )

                fair_metric_data = []
                sens_attr = self.sensitive_attribute
                for res, dset in zip(
                    [orig_rec_res, orig_test_res, cf_rec_res, cf_test_res],
                    [self.rec_data.dataset, test_data.dataset, self.rec_data.dataset, test_data.dataset]
                ):
                    gr1_mask = dset.user_feat[sens_attr][uids].numpy() == self.adv_group
                    gr2_mask = dset.user_feat[sens_attr][uids].numpy() == self.disadv_group
                    fair_metric_data.append(eval_utils.compute_dp_with_masks(res[:, -1], gr1_mask, gr2_mask))

                bab_epoch_data.append([
                    new_example[0],
                    new_example[-2],
                    items_involved,
                    np.stack((orig_rec_res[:, -1], orig_test_res[:, -1])),
                    np.stack((cf_rec_res[:, -1], cf_test_res[:, -1])),
                    fair_metric_data[:2],
                    fair_metric_data[2:]
                ])
            else:
                bab_epoch_data.append(None)

        epoch_fair_loss = np.mean(epoch_fair_loss)
        fair_losses.append(epoch_fair_loss)

        return bab_epoch_data, epoch_fair_loss

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

        bab_data = []
        fair_losses = []
        for epoch in iter_epochs:
            bab_epoch_data, epoch_fair_loss = self.run_epoch(
                epoch, batched_data, test_data, fair_losses, topk=topk
            )

            if self.verbose:
                self._verbose_plot(fair_losses, epoch)

            bab_data.append(bab_epoch_data)

            # if new_example is not None:
            #     self.update_new_example(
            #         new_example,
            #         detached_batched_data,
            #         test_scores_args,
            #         test_model_topk,
            #         rec_scores_args,
            #         rec_model_topk,
            #         epoch_fair_loss
            #     )
            #
            #     update_best_example_args = [best_cf_example, new_example, loss_total, best_loss, orig_loss]
            #     # if self._check_early_stopping(epoch_fair_metric, *update_best_example_args)
            #     if self._check_early_stopping(epoch_fair_loss, *update_best_example_args):
            #         break
            #
            #     best_loss = self.update_best_cf_example(*update_best_example_args, model_topk=rec_model_topk)

            self.logger.info("{} CF examples".format(len(best_cf_example)))

        return bab_data, (rec_model_topk, test_model_topk)

    def train(self, epoch, scores_args, users_ids, topk=10, previous_loss_value=None):
        t = time.time()

        # `_get_no_grad_pred_model_score_data` call torch `eval` inside
        cf_topk_pred_idx, _ = self._get_no_grad_pred_model_score_data(scores_args)

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
        self.cf_optimizer.zero_grad()

        # import pdb; pdb.set_trace()
        # for name, param in self.cf_model.named_parameters():
        #     if name == "P_symm":
        #         print(param.grad[param.grad])

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

        return cf_stats, loss_total, fair_loss
