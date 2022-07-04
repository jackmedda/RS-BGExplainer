import os
import inspect
from typing import Iterable

import tqdm
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Describer(object):
    """
    Plot ideas:
    1) Distribution of demographic groups representation in removed edges (more females, more males, etc.)
    2) Distribution of number of removed edges for demographic group of explained users
    3) Distribution of item categories representation in removed edges (more Action, more Death Metal, etc.)
    4) Are there items that are more removed from the top-k than others? Which is their category? (Subplots)
    5) Which are the category of items that are removed the most from the top-k list across demographic groups?
    6) Which are the users and items that lose interactions the most? (Subplot, one for user, one for item)
    """
    
    _plot_methods_prefix = "plot_"

    def __init__(self, exps, train_data, config, sensitive_attributes=None, out_base_path=''):
        self.desc_data = []
        self.exps = exps
        self.train_data = train_data
        self.sens_attrs = ['gender', 'age'] if sensitive_attributes is None else sensitive_attributes

        user_feat = self.train_data.dataset.user_feat
        self.user_df = pd.DataFrame({
            'user_id': user_feat['user_id'].numpy(),
            **{sens_attr: user_feat[sens_attr].numpy() for sens_attr in self.sens_attrs}
        })

        item_feat = self.train_data.dataset.item_feat
        self.item_df = pd.DataFrame({
            'item_id': item_feat['item_id'].numpy(),
            'class': map(lambda x: [el for el in x if el != 0], item_feat['class'].numpy().tolist())
        })

        self.sens_maps = [self.train_data.dataset.field2id_token[sens_attr] for sens_attr in self.sens_attrs]
        self.item_class_map = self.train_data.dataset.field2id_token['class']

        user_hist_matrix, _, user_hist_len = self.train_data.dataset.history_item_matrix()
        self.user_hist_matrix, self.user_hist_len = user_hist_matrix.numpy(), user_hist_len.numpy()

        item_hist_matrix, _, item_hist_len = self.train_data.dataset.history_user_matrix()
        self.item_hist_matrix, self.item_hist_len = item_hist_matrix.numpy(), item_hist_len.numpy()

        script_path = os.path.abspath(os.path.dirname(inspect.getsourcefile(lambda: 0)))
        _fair = config["explain_fairness"]
        self.out_base_path = out_base_path or os.path.join(script_path,
                                                           os.pardir,
                                                           os.pardir,
                                                           f'{"fair" if _fair else ""}_plots_analysis',
                                                           config['dataset'])

        self.extract_desc_data()
        self.user_index = dict.fromkeys(self.user_df.user_id.tolist(), -1)
        for i, data in enumerate(self.desc_data):
            if self.user_index[data[0]] != -1:
                self.user_index[data[0]].append(i)
            else:
                self.user_index[data[0]] = [i]

    def extract_desc_data(self):
        if not self.desc_data:
            for user in self.exps:
                for exp in exps[user]:
                    desc_user = [user] + exp[1:4] + exp[6:11]
                    cf_adj, orig_adj = exp[-3:-1]
                    del_edges = (orig_adj != cf_adj).nonzero()
                    del_edges = torch.stack((del_edges[:, 0], del_edges[:, 1]))
                    desc_user.extend([del_edges])

                    self.desc_data.append(desc_user)

    def plot(self, how="all"):
        plots = []
        if isinstance(how, Iterable):
            plots = self.filter_supported_plots(how)
        elif how == "all":
            plots = self.supported_plots()

        for plot in plots:
            plot_f = getattr(Describer, plot)
            plot_f()

    @staticmethod
    def plot_handle(func):
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            plt.close()

        return wrapper

    @plot_handle
    def plot_demo_groups_over_del_edges(self):
        for attr, sens_map in zip(self.sens_attrs, self.sens_maps):
            user_sens = self._get_user_sens_attr_from_id([data[0] for data in self.desc_data], attr)
            del_edges = [data[-1] for data in self.desc_data]

            sens_edges_counts = []
            exp_edges_counts = []
            repeats = []
            for edges in del_edges:
                sens_edges = self._get_user_sens_attr_from_id(edges[0], attr)
                uni, counts = np.unique(sens_edges, return_counts=True)
                exp_edges_counts.append(counts)
                sens_edges_counts.append(uni)
                repeats.append(len(counts))
            exp_edges_counts = np.concatenate(exp_edges_counts)
            sens_edges_counts = np.concatenate(sens_edges_counts)

            user_sens = np.repeat(user_sens, repeats)

            plot_df = pd.DataFrame(
                zip(user_sens, exp_edges_counts.flatten(), sens_edges_counts.flatten()),
                columns=['sens_exp', 'edges_counts', 'sens_edges']
            )
            rows = 1 + int(np.ceil((len(sens_map) - 1) / 3))
            cols = min(3, len(sens_map) - 1)

            fig = plt.figure(figsize=(12, 12))
            fig.suptitle("Distribution Demo Groups over Deleted Edges")

            ax1 = plt.subplot(rows, 1, 1)
            ax1.set_title("All Explained Users")
            sns.boxplot(y="edges_counts", hue="sens_edges", data=plot_df, ax=ax1)
            
            plot_idx = 2
            for r in range(2, rows + 1):
                for _ in range(1, cols + 1):
                    _ax = plt.subplot(rows, cols, plot_idx)
                    _ax.set_title(f"Only users with {attr} = {sens_map[plot_idx - 1]}")
                    sns.boxplot(
                        x="sens_edges", hue="edges_counts",
                        data=plot_df[plot_df['sens_exp'] == sens_map[plot_idx - 1]], ax=ax1)
                    _ax.set_xticklabels([sens_map[x] for x in _ax.get_xticklabels()])
                    plot_idx += 1

    @plot_handle
    def plot_del_edges_dist_over_demo_groups(self):
        for attr, sens_map in zip(self.sens_attrs, self.sens_maps):
            user_sens = self._get_user_sens_attr_from_id([data[0] for data in self.desc_data], attr)
            n_del_edges = [len(data[-1]) for data in self.desc_data]

            plot_df = pd.DataFrame(zip(user_sens, n_del_edges), columns=['sens_exp', 'n_del_edges'])

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            fig.suptitle("Distribution of Number of Deleted Edges over Explained Demo Groups")

            sns.boxplot(y="n_del_edges", hue="sens_exp", data=plot_df, ax=ax1)
            ax.set_xticklabels([sens_map[x] for x in ax.get_xticklabels()])

    @plot_handle
    def plot_item_categories_over_del_edges(self):
        for attr, sens_map in zip(self.sens_attrs, self.sens_maps):
            user_sens = self._get_user_sens_attr_from_id([data[0] for data in self.desc_data], attr)
            del_edges = [data[-1] for data in self.desc_data]

            cat_edges_counts = []
            item_cats_edges_counts = []
            repeats = []
            for edges in del_edges:
                item_cats_edges = self._get_item_categories_from_id(edges[1])
                uni, counts = np.unique(item_cats_edges, return_counts=True)
                item_cats_edges_counts.append(counts)
                cat_edges_counts.append(uni)
                repeats.append(len(counts))
            item_cats_edges_counts = np.concatenate(item_cats_edges_counts)
            cat_edges_counts = np.concatenate(cat_edges_counts)

            user_sens = np.repeat(user_sens, repeats)

            plot_df = pd.DataFrame(
                zip(user_sens, item_cats_edges_counts.flatten(), cat_edges_counts.flatten()),
                columns=['sens_exp', 'edges_counts', 'cat_edges']
            )
            rows = 1 + int(np.ceil((len(sens_map) - 1) / 3))
            cols = min(2, len(sens_map) - 1)

            fig = plt.figure(figsize=(12, 12))
            fig.suptitle("Distribution Item Categories over Deleted Edges")

            ax1 = plt.subplot(rows, 1, 1)
            ax1.set_title("All Explained Users")
            sns.boxplot(y="edges_counts", hue="cat_edges", data=plot_df, ax=ax1)

            plot_idx = 2
            for r in range(2, rows + 1):
                for _ in range(1, cols + 1):
                    _ax = plt.subplot(rows, cols, plot_idx)
                    _ax.set_title(f"Only users with {attr} = {sens_map[plot_idx - 1]}")
                    sns.boxplot(
                        x="cat_edges", y="edges_counts",
                        data=plot_df[plot_df['sens_exp'] == sens_map[plot_idx - 1]], ax=ax1)
                    _ax.set_xticklabels([sens_map[x] for x in _ax.get_xticklabels()])
                    plot_idx += 1

    @plot_handle
    def plot_dist_removed_items_topk_k_and_categories(self, topk_removed=10):
        for attr, sens_map in zip(self.sens_attrs, self.sens_maps):
            user_sens = self._get_user_sens_attr_from_id([data[0] for data in self.desc_data], attr)
            topks = [(data[1], data[2]) for data in self.desc_data]

            del_items_topk = []
            item_cats_topk_counts = []
            repeats = []
            for topk, cf_topk in topks:
                del_items = np.setdiff1d(topk, cf_topk)
                item_cats_topk = self._get_item_categories_from_id(del_items)
                del_items_topk.append(del_items)
                item_cats_topk_counts.append(item_cats_topk)
                repeats.append(len(del_items))
            del_items_topk = np.concatenate(del_items_topk)
            item_cats_topk_counts = np.concatenate(item_cats_topk_counts)

            user_sens = np.repeat(user_sens, repeats)

            plot_df = pd.DataFrame(
                zip(user_sens, del_items_topk.flatten(), del_items_topk.flatten(), item_cats_topk_counts.flatten()),
                columns=['sens_exp', 'del_topk_items', 'item_count' 'del_topk_cats']
            )
            plot_df['item_count'] = plot_df['item_count'].map(plot_df['item_count'].value_counts())
            plot_df.sort_values('item_count', ascending=False, inplace=True)
            top_items_removed = plot_df['del_topk_items'].drop_duplicates()[:topk_removed].values
            topk_plot_df = plot_df[plot_df['del_topk_items'].isin(top_items_removed)]
            
            rows = 1 + int(np.ceil((len(sens_map) - 1) / 3))
            cols = min(2, len(sens_map) - 1)

            fig = plt.figure(figsize=(12, 12))
            fig.suptitle("Top Removed Items and Categories")

            ax1 = plt.subplot(rows, 2, 1)
            ax1.set_title("All Explained Users")
            sns.barplot(x="del_topk_items", y="item_count", ci=None, data=topk_plot_df, ax=ax1)
            
            cats_plot_df = topk_plot_df.explode("del_topk_cats")
            cats_plot_df = cats_plot_df.groupby("del_topk_cats").sum().reset_index()
            cats_plot_df = cats_plot_df.sort_values("item_count", ascending=False)
            ax2 = plt.subplot(rows, 2, 2)
            ax2.set_title("All Explained Users")
            sns.barplot(x="del_topk_cats", y="item_count", ci=None, data=cats_plot_df, ax=ax2)

            plot_idx = 2
            for r in range(2, rows + 1):
                for _ in range(1, cols + 1):
                    _ax = plt.subplot(rows, cols, plot_idx)
                    _ax.set_title(f"Only users with {attr} = {sens_map[plot_idx - 1]}")
                    sns.boxplot(
                        x="cat_edges", y="edges_counts",
                        data=plot_df[plot_df['sens_exp'] == sens_map[plot_idx - 1]], ax=ax1)
                    _ax.set_xticklabels([sens_map[x] for x in _ax.get_xticklabels()])
                    plot_idx += 1

    def _get_user_sens_attr_from_id(self, user_ids, sens_attr):
        user_ids = user_ids if isinstance(user_ids, Iterable) else [user_ids]
        return self.user_df.set_index('user_id').loc[user_ids, sens_attr].numpy()

    def _get_item_categories_from_id(self, item_ids):
        item_ids = item_ids if isinstance(item_ids, Iterable) else [item_ids]
        return np.array([cl for cl_list in self.item_df.set_index('item_id').loc[item_ids, "class"] for cl in cl_list])

    @staticmethod
    def supported_plots():
        plot_methods = inspect.getmembers(Describer, predicate=inspect.isroutine)
        plot_methods = sorted([p[0] for p in plot_methods if p[0].startswith(Describer._plot_methods_prefix)])
        
        return plot_methods
    
    @staticmethod
    def filter_supported_plots(plot_list):
        supp_plots = Describer.supported_plots()
        filtered_plots = []
        for plot_f in plot_list:
            full_plot_f_name = Describer._plot_methods_prefix + plot_f
            if full_plot_f_name in supp_plots:
                filtered_plots.append(full_plot_f_name)
            else:
                print(f"{Describer.__name__}: `{plot_f}` is not a supported plot type")

        return filtered_plots
