# Based on https://github.com/RexYing/gnn-model-explainer/blob/master/explainer/explain.py

import sys
import time
import copy
from typing import Iterable

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
        self.cat_intersharing = kwargs.get("cat_intersharing_distrib", None)
        self.attr_cat_distrib = kwargs.get("attr_cat_distrib", None)
        self.sensitive_attributes = config['sensitive_attributes']

        self.scores_args, self.topk_args = None, None
        self.model_scores, self.model_scores_topk, self.model_topk_idx = None, None, None

        self.verbose = kwargs.get("verbose", False)

        self.target_scope = config['target_scope']
        self.group_explain = config['group_explain']
        self.user_batch_exp = config['user_batch_exp']

        self.fair_target_lambda = config['fair_target_lambda']
        self.intershare_eta = config['intershare_eta']
        self.knapsack = config['knapsack']
        self.knap_theta = config['knap_theta']

        if config['filter_categories'] is not None or config['cats_vs_all'] is not None:
            self.item_categories_map = kwargs.get('item_cats', None)
        else:
            self.item_categories_map = self.dataset.item_feat['class']

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

        if self.knapsack and self.group_explain:
            knap_counts_data = {}
            for attr in self.train_pref_ratio:
                knap_counts_data[attr] = {}
                user_count = torch.bincount(self.dataset.user_feat[attr][batched_data])
                for gr, gr_data in self.train_pref_ratio[attr].items():
                    import pdb; pdb.set_trace()
                    if gr_data is not None:
                        knap_counts_data[attr][gr] = (user_count[gr] * topk * gr_data).round().int()
                    else:
                        knap_counts_data[attr][gr] = None
        else:
            knap_counts_data = None

        for epoch in iter_epochs:
            if self.group_explain:
                knap_counts = copy.deepcopy(knap_counts_data) if knap_counts_data is not None else None
                iter_data = batched_data[torch.randperm(batched_data.shape[0])].split(self.user_batch_exp)
                for batch_idx, batch_user in enumerate(iter_data):
                    self.user_id = batch_user
                    batched_data_epoch = BGExplainer.prepare_batched_data(batch_user, data)
                    self.compute_model_predictions(batched_data_epoch, topk)
                    new_example, loss_total, fair_loss = self.train(epoch, topk=topk, knap_counts=knap_counts)
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
                best_loss = self.update_best_cf_example(best_cf_example, new_example, loss_total, best_loss, first_fair_loss)

            print("{} CF examples for user = {}".format(len(best_cf_example), self.user_id))

        return best_cf_example, self.model_scores.detach().cpu().numpy()

    # @profile
    def train(self, epoch, topk=10, knap_counts=None):
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
        if self.train_pref_ratio is not None:  # prepare the loss function for fairness purposes
            user_feat = self.dataset.user_feat
            user_id_mask = self.user_id.unsqueeze(-1) if self.user_id.dim() == 0 else self.user_id
            user_feat = {k: feat[user_id_mask] for k, feat in user_feat.interaction.items()}

            if self.explain_fairness_NDCGApprox:
                kwargs = {
                    "fair_loss_f": utils.NDCGApproxLoss(),
                    "target": get_bias_disparity_target_NDCGApprox(
                        cf_scores,
                        self.train_pref_ratio,
                        self.item_categories_map,
                        self.sensitive_attributes,
                        user_feat,
                        target_scope=self.target_scope,
                        topk=topk,
                        lmb=self.fair_target_lambda,
                        cat_sharing_prob=self.cat_sharing_prob,
                        cat_intersharing=self.cat_intersharing,
                        attr_cat_distrib=self.attr_cat_distrib,
                        eta=self.intershare_eta,
                        knap_counts=knap_counts,
                        knap_theta=self.knap_theta
                    )
                }
            else:
                kwargs = {
                    "fair_loss_f": BiasDisparityLoss(
                        self.train_pref_ratio,
                        self.item_categories_map,
                        self.sensitive_attributes,
                        target_scope=self.target_scope,
                        topk=topk,
                        lmb=self.fair_target_lambda,
                        cat_sharing_prob=self.cat_sharing_prob,
                        cat_intersharing=self.cat_intersharing,
                        attr_cat_distrib=self.attr_cat_distrib,
                        eta=self.intershare_eta,
                        knap_counts=knap_counts,
                        knap_theta=self.knap_theta
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

        ###############################################
        debug = True
        if debug and cf_stats is not None and loss_graph_dist.item() > self.old_graph_dist:
            # target = []
            # cf_topk_pred_idx_np = cf_topk_pred_idx.cpu().numpy()
            # for i, l_t in enumerate(kwargs['target'].cpu().numpy()):
            #     recs = l_t.nonzero()[0]
            #     if len(recs) < topk:
            #         l_ctp = cf_topk_pred_idx_np[i]
            #         l_ctp_mask = (l_ctp[:, None] == np.intersect1d(l_ctp, recs)).any(axis=1)
            #         target.append(np.concatenate([recs, l_ctp[~l_ctp_mask]])[:topk].tolist())
            #     else:
            #         target.append(recs.tolist())
            # # target = pd.DataFrame(kwargs['target'].nonzero().cpu().numpy())
            # # target[0] = target[0].map(dict(zip(range(self.user_id.shape[0]), self.user_id.numpy())))
            # # target = target.groupby(0).apply(lambda _df: pd.Series({1: _df[1].tolist()})).reset_index()
            # # target.rename(columns={0: 'user_id', 1: 'cf_topk_pred'}, inplace=True)
            # target_df = pd.DataFrame(zip(self.user_id.numpy(), target), columns=['user_id', 'cf_topk_pred'])

            with torch.no_grad():
                cf_scores_pred_after = self.get_scores(self.cf_model, *self.scores_args, pred=True)
                _, cf_topk_pred_idx_after = self.get_top_k(cf_scores_pred, **self.topk_args)

            self.cf_model.eval()
            target = cf_topk_pred_idx_after.cpu().numpy()

            target_df = pd.DataFrame(zip(self.user_id.numpy(), target), columns=['user_id', 'cf_topk_pred'])

            class a:
                pass

            a.dataset = self.dataset
            config = {'USER_ID_FIELD': 'user_id', 'ITEM_ID_FIELD': 'item_id'}
            old_pred = pd.DataFrame(zip(self.user_id.numpy(), cf_topk_pred_idx.cpu().numpy())).rename(
                columns={0: 'user_id', 1: 'cf_topk_pred'})
            train_bias_user, _ = utils.generate_bias_ratio(a,
                                                           config,
                                                           sensitive_attrs=self.sensitive_attributes,
                                                           user_subset=self.user_id.numpy(),
                                                           item_cats=self.item_categories_map)

            old_rec_bias, _ = utils.generate_bias_ratio(a,
                                                        config,
                                                        sensitive_attrs=self.sensitive_attributes,
                                                        history_matrix=old_pred,
                                                        item_cats=self.item_categories_map)

            rec_bias, _ = utils.generate_bias_ratio(a,
                                                    config,
                                                    sensitive_attrs=self.sensitive_attributes,
                                                    history_matrix=target_df,
                                                    item_cats=self.item_categories_map)

            bd_old = utils.compute_bias_disparity(train_bias_user, old_rec_bias, a)
            bd_new = utils.compute_bias_disparity(train_bias_user, rec_bias, a)

            n_cat = self.item_categories_map.unique().shape[0]

            fake_data = np.zeros_like(bd_old['gender']['M'].numpy())
            bd_df = {
                (self.dataset.field2id_token['gender'] == 'M').nonzero()[0].item(): pd.DataFrame(zip(
                        np.concatenate([
                            fake_data,
                            fake_data,
                            np.abs(bd_old['gender']['M'].numpy()) - np.abs(bd_new['gender']['M'].numpy()),
                        ]),
                        list(range(n_cat)) + list(range(n_cat)) + list(range(n_cat)),
                        [''] * n_cat + [''] * n_cat + ['bd'] * n_cat
                     )),
                (self.dataset.field2id_token['gender'] == 'F').nonzero()[0].item(): pd.DataFrame(zip(
                        np.concatenate([
                            fake_data,
                            fake_data,
                            np.abs(bd_old['gender']['F'].numpy()) - np.abs(bd_new['gender']['F'].numpy()),
                        ]),
                        list(range(n_cat)) + list(range(n_cat)) + list(range(n_cat)),
                        [''] * n_cat + [''] * n_cat + ['bd'] * n_cat
                     )),
            }

            import seaborn as sns
            import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(1, 2, sharey=True, figsize=(12, 5))
            # sns.barplot(y=0, x=1, data=m_df, ax=ax[0])
            # sns.barplot(y=0, x=1, data=f_df, ax=ax[1])
            # plt.show()
            # plt.close()

            fig, ax = plt.subplots(1, 2, sharey=True, figsize=(12, 5))
            i = 0
            check = False
            for gr, gr_s in enumerate(self.dataset.field2id_token['gender']):
                gr_mask = user_feat['gender'] == gr
                if not gr_mask.any():
                    continue

                new_items = torch.tensor(target)[gr_mask].flatten()
                n_users = (self.user_id[gr_mask].shape[0] * topk)
                new_pref = torch.bincount(self.item_categories_map[new_items].flatten(), minlength=n_cat) / n_users
                new_pref[0] = 0

                old_pred_items = cf_topk_pred_idx.cpu().numpy()[gr_mask].flatten()
                old_pref = torch.bincount(self.item_categories_map[old_pred_items].flatten(), minlength=n_cat) / n_users
                old_pref[0] = 0

                orig_pref = self.train_pref_ratio['gender']

                pref_df = pd.DataFrame(zip(
                    np.concatenate([
                        orig_pref[gr].numpy() - new_pref.numpy(),
                        orig_pref[gr].numpy() - old_pref.numpy(),
                        np.zeros(orig_pref[gr].shape[0])
                    ]),
                    list(range(n_cat)) + list(range(n_cat)) + list(range(n_cat)),
                    ['new'] * n_cat + ['old'] * n_cat + [''] * n_cat
                ))

                if (np.stack(target_df['cf_topk_pred']) != np.stack(old_pred['cf_topk_pred'])).any():
                    check = True

                sns.barplot(x=1, y=0, hue=2, data=pref_df, ax=ax[i])

                ax_twinx = ax[i].twinx()
                sns.barplot(x=1, y=0, data=bd_df[gr], ax=ax_twinx)

                i += 1

            # kkk = (self.item_categories_map[new_items.T] == 5).nonzero()[:,:2]
            # self.item_categories_map[new_items.T][kkk[:, 0], kkk[:, 1]]

            if check:
                plt.show(block=False)
                plt.pause(1)
                import pdb; pdb.set_trace()

            # import pdb; pdb.set_trace()
            plt.close()
        ###############################################

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
                 train_pref_ratio: dict,
                 item_categories_map: torch.Tensor,
                 sensitive_attributes: Iterable[str],
                 target_scope="group",
                 topk=10,
                 lmb=0.5,
                 eta=0.1,
                 cat_sharing_prob=None,
                 cat_intersharing=None,
                 attr_cat_distrib=None,
                 knap_counts=None,
                 knap_theta=8,
                 size_average=None, reduce=None, reduction: str = 'mean', margin=0.1) -> None:
        super(BiasDisparityLoss, self).__init__(size_average, reduce, reduction)

        self.train_pref_ratio = train_pref_ratio
        self.item_categories_map = item_categories_map
        self.sensitive_attributes = sensitive_attributes
        self.target_scope = target_scope
        self.topk = topk
        self.lmb = lmb
        self.eta = eta
        self.cat_sharing_prob = cat_sharing_prob
        self.cat_intersharing = cat_intersharing
        self.attr_cat_distrib = attr_cat_distrib
        self.knap_counts = knap_counts
        self.knap_theta = knap_theta

        self.margin = margin

    def forward(self, _input: torch.Tensor, demo_groups: torch.Tensor) -> torch.Tensor:
        sorted_target, sorted_input, _ = get_bias_disparity_sorted_target(_input,
                                                                          self.train_pref_ratio,
                                                                          self.item_categories_map,
                                                                          self.sensitive_attributes,
                                                                          demo_groups,
                                                                          target_scope=self.target_scope,
                                                                          topk=self.topk,
                                                                          lmb=self.lmb,
                                                                          eta=self.eta,
                                                                          cat_sharing_prob=self.cat_sharing_prob,
                                                                          cat_intersharing=self.cat_intersharing,
                                                                          attr_cat_distrib=self.attr_cat_distrib,
                                                                          knap_counts=self.knap_counts,
                                                                          knap_theta=self.knap_theta)

        return (-sorted_target * sorted_input).mean(dim=1)


def get_bias_disparity_target_NDCGApprox(scores,
                                         train_pref_ratio,
                                         item_categories_map,
                                         sensitive_attributes,
                                         demo_groups,
                                         target_scope="group",
                                         topk=10,
                                         lmb=0.5,
                                         eta=0.1,
                                         cat_sharing_prob=None,
                                         cat_intersharing=None,
                                         attr_cat_distrib=None,
                                         knap_counts=None,
                                         knap_theta=0.01):
    sorted_target, _, sorted_idxs = get_bias_disparity_sorted_target(scores,
                                                                     train_pref_ratio,
                                                                     item_categories_map,
                                                                     sensitive_attributes,
                                                                     demo_groups,
                                                                     target_scope=target_scope,
                                                                     topk=topk,
                                                                     lmb=lmb,
                                                                     eta=eta,
                                                                     cat_sharing_prob=cat_sharing_prob,
                                                                     cat_intersharing=cat_intersharing,
                                                                     attr_cat_distrib=attr_cat_distrib,
                                                                     knap_counts=knap_counts,
                                                                     knap_theta=knap_theta)

    return torch.gather(sorted_target, 1, torch.argsort(sorted_idxs))


def get_bias_disparity_sorted_target(scores,
                                     train_pref_ratio,
                                     item_categories_map,
                                     sensitive_attributes,
                                     demo_groups,
                                     target_scope="group",
                                     topk=10,
                                     lmb=1.2,
                                     eta=0.1,
                                     offset=0.05,
                                     cat_sharing_prob=None,
                                     cat_intersharing=None,
                                     attr_cat_distrib=None,
                                     knap_counts=None,
                                     knap_theta=8):
    sorted_scores, sorted_idxs = torch.topk(scores, scores.shape[1])

    target = torch.zeros_like(sorted_idxs, dtype=torch.float)
    if target_scope == "group":
        for attr in sensitive_attributes:
            attr_pref_ratio = train_pref_ratio[attr]  # dict that maps demographic groups to pref ratio for each category
            assert len(sensitive_attributes) == 1, "Not supported with multiple sensitive attributes at once"

            cat_order = torch.randperm(item_categories_map.max() + 1)  # random order of cat to avoid attention on first
            for gr, gr_pref_ratio in attr_pref_ratio.items():
                if gr in demo_groups[attr]:
                    gr_idxs = sorted_idxs[demo_groups[attr] == gr, :]
                    gr_target = torch.zeros_like(gr_idxs, dtype=torch.float)

                    if knap_counts is not None:
                        # cats = item_categories_map[gr_idxs]
                        # cats_flat = cats.view((cats.shape[0] * cats.shape[1], cats.shape[-1])).numpy()
                        # cats_data = np.stack([
                        #     np.repeat(np.arange(cats.shape[0]), cats.shape[1]),
                        #     np.tile(np.arange(cats.shape[1]), cats.shape[0])
                        # ], axis=0)
                        #
                        # rank_df = pd.DataFrame(zip(*cats_data, cats_flat), columns=['user', 'pos', 'cats'])
                        # rank_df['cats'] = rank_df['cats'].map(lambda x: np.array([]) if all(cat == 0 for cat in x) else np.array(x))
                        # rank_df['safe_cat'] = rank_df['cats'].map(lambda x: knap_counts[attr][gr][x[x.nonzero()]].all().item())
                        # rank_df = rank_df[rank_df['safe_cat']]
                        # cat_distrib = rank_df['cats'].map(lambda x: np.mean([attr_cat_distrib[attr][gr][c] for c in x] if len(x) > 0 else np.nan))
                        # rank_df['rank'] = (2 ** ((rank_df['pos'] + 1) / knap_theta)).replace(0, np.nan) * cat_distrib
                        #
                        # rank_df.sort_values('rank', ascending=True, inplace=True)
                        #
                        # target_items = rank_df.groupby("user").head(topk)
                        #
                        # for cat in target_items['cats']:
                        #     knap_counts[attr][gr][cat] -= 1
                        #
                        # target_items = torch.tensor(target_items[["user", "pos"]].values.tolist())
                        #
                        # gr_target[target_items[:, 0], target_items[:, 1]] = 1

                        for i in range(gr_idxs.shape[0]):
                            i_cats = item_categories_map[gr_idxs[i]]
                            one_counts = 0

                            if not (knap_counts[attr][gr] > 0).any():
                                continue

                            distrib = attr_cat_distrib[attr][gr][i_cats.numpy()].mean(axis=1)

                            pos_rec = np.arange(i_cats.shape[0])

                            # take low percentage items if they are close to the topk and insert them into the final
                            # spots of the topk
                            candidate_items = (distrib < np.quantile(distrib[~np.isnan(distrib)], 1 / cat_order.shape[0]))
                            candidate_items, = candidate_items[:(topk + round(topk * knap_theta))].nonzero()
                            candidate_items = candidate_items[candidate_items >= topk]
                            if candidate_items.shape[0] > 0:
                                n_items = candidate_items.shape[0]
                                temp = i_cats[(topk - n_items):topk].clone()
                                i_cats[(topk - n_items):topk] = i_cats[candidate_items]
                                i_cats[candidate_items] = temp
                                temp_pos = pos_rec[(topk - n_items):topk].copy()
                                pos_rec[(topk - n_items):topk] = pos_rec[candidate_items]
                                pos_rec[candidate_items] = temp_pos

                            # rank_score = 2 ** ((np.arange(gr_idxs[i].shape[0]) + 1) / knap_theta)
                            # # rank_score2 = np.log10(np.arange(gr_idxs[i].shape[0]) + 1) / np.log10(np.full(gr_idxs[i].shape[0], topk))
                            # rank_score *= distrib
                            # rank = np.argsort(rank_score)

                            # if (np.sort(rank[:topk]) == np.arange(topk)).all():
                            #     rank = np.concatenate([np.arange(topk), rank[topk:]])

                            for pos, j_cat, dstb in zip(pos_rec, i_cats, distrib):
                                safe_cat = j_cat[j_cat.nonzero()]
                                if (knap_counts[attr][gr][safe_cat] > 0).all():
                                    knap_counts[attr][gr][safe_cat] -= 1
                                    gr_target[i, pos] = 1
                                    one_counts += 1

                                if one_counts == topk:
                                    break
                    else:
                        mean_intersharing = None
                        if attr_cat_distrib is not None and cat_intersharing is not None:
                            mean_intersharing = np.nanmean(attr_cat_distrib[attr][gr] * cat_intersharing, axis=1)
                            mean_intersharing = mean_intersharing.sum() / (cat_intersharing.shape[0] - 1)

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

                                    # array(['[PAD]',
                                    #        'Animation', "Children's", 'Comedy', 'Action',
                                    #        'Adventure', 'Thriller', 'Drama', 'Crime',
                                    #        'Sci-Fi', 'War', 'Romance', 'Horror',
                                    #        'Musical', 'Documentary', 'Western', 'Fantasy',
                                    #        'Film-Noir', 'Mystery', 'unknown'], dtype='<U11')

                                    # take the percentage of distribution of this category for each user
                                    # if gr_pref_ratio[cat] = 0.3, n_target would be 3 in a top-10 setting
                                    # this means that for each user we can choose maximum 3 items of this cat
                                    prob = gr_pref_ratio[cat].item()
                                    prob_lmb = prob * lmb

                                    p = np.array([1 - prob_lmb if prob_lmb < 1 else 0., prob_lmb if prob_lmb < 1 else 1.])
                                    p /= p.sum()
                                    # if the probability is too low for n_target to be 1, then n_target becomes 1 depending
                                    # on a random choice based on the probability itself
                                    n_target = max(round(prob * topk * one_counts.shape[0]), np.random.choice([0, 1], p=p))
                                    # n_target is reduced for a category `cat` if many items share `cat` with other categories
                                    if cat_sharing_prob is not None:
                                        if mean_intersharing is not None:
                                            share_prob = 1 - cat_sharing_prob[cat]
                                            curr_cat_intershare = np.nanmean(attr_cat_distrib[attr][gr] * cat_intersharing[cat])
                                            share_prob *= 1 - eta * (curr_cat_intershare - mean_intersharing)
                                        else:
                                            share_prob = 1 - cat_sharing_prob[cat]
                                    else:
                                        share_prob = 1

                                    n_target = int(round((n_target * share_prob)))

                                    target_items = df_filt.groupby("user").apply(
                                        lambda _df: _df.head(min(_df.shape[0], topk - _df["count"].head(1).item()))
                                    ).reset_index(drop=True).iloc[:n_target]
                                    target_items = torch.tensor(target_items[["user", "rank"]].values)

                                    gr_target[target_items[:, 0], target_items[:, 1]] = 1
                                else:
                                    break
                    target[demo_groups[attr] == gr, :] = gr_target
    elif target_scope == "individual":
        cat_order = torch.randperm(item_categories_map.max() + 1)  # random order of cat to avoid attention on first

        # mean_intersharing = None
        # if attr_cat_distrib is not None and cat_intersharing is not None:
        #     mean_intersharing = np.nanmean(attr_cat_distrib[attr][gr] * cat_intersharing, axis=1)
        #     mean_intersharing = mean_intersharing.sum() / (cat_intersharing.shape[0] - 1)

        for cat in cat_order:
            # counts how many ones are already in the target of each user
            one_counts = torch.count_nonzero((target > 0), dim=1).cpu().numpy()
            count_df = pd.DataFrame(
                zip(np.arange(one_counts.shape[0]), one_counts),
                columns=["user", "count"]
            )

            # generate dataframe with all items with category cat in the list of scores of the current users
            rank_df = pd.DataFrame(
                (item_categories_map[sorted_idxs] == cat).nonzero().detach().cpu().numpy(),
                columns=['user', 'rank', 'cat']
            )

            df = rank_df.join(count_df.set_index("user"), on="user")
            # if some users have the topk full => don't add
            df_filt = df[df["count"] < topk]

            if not df_filt.empty:
                df_filt = df_filt.sort_values("rank")

                # array(['[PAD]',
                #        'Animation', "Children's", 'Comedy', 'Action',
                #        'Adventure', 'Thriller', 'Drama', 'Crime',
                #        'Sci-Fi', 'War', 'Romance', 'Horror',
                #        'Musical', 'Documentary', 'Western', 'Fantasy',
                #        'Film-Noir', 'Mystery', 'unknown'], dtype='<U11')

                # take the percentage of distribution of this category for each user
                # if gr_pref_ratio[cat] = 0.3, n_target would be 3 in a top-10 setting
                # this means that for each user we can choose maximum 3 items of this cat
                prob = train_pref_ratio[demo_groups['user_id'], cat].numpy().round(3)
                prob_lmb = prob * lmb
                p = np.array([[1 - p_l, p_l] if p_l < 1 else [0., 1.] for p_l in prob_lmb])
                p /= p.sum(axis=1)[:, None]

                # if the probability is too low for n_target to be 1, then n_target becomes 1 depending
                # on a random choice based on the probability itself
                n_target = []
                for user_prob, user_p in zip(prob_lmb, p):
                    n_target.append(max(int(user_prob * topk), np.random.choice([0, 1], p=user_p)))
                n_target = np.array(n_target)

                # n_target is reduced for a category `cat` if many items share `cat` with other categories
                share_prob = 1 - cat_sharing_prob[cat] if cat_sharing_prob is not None else 1

                n_target = (n_target * share_prob).round().astype(int)

                target_items = df_filt.groupby("user").apply(
                    lambda _df: _df.take(np.arange(min(n_target[_df["user"].head(1).values], topk - _df["count"].head(1).values)))
                )
                target_items = torch.tensor(target_items[["user", "rank"]].values)

                target[target_items[:, 0], target_items[:, 1]] = 1
            else:
                break
    else:
        raise NotImplementedError(f"target_scope = `{target_scope}` is not supported")

    return target, sorted_scores, sorted_idxs
