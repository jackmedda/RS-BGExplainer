import os
import pickle
import inspect
import argparse
from typing import Iterable

import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import utils


class Describer(object):
    """
    Plot ideas:
    1) Distribution of demographic groups representation in removed edges (more females, more males, etc.)
    2) Distribution of number of removed edges for demographic group of explained users
    3) Distribution of item categories representation in removed edges (more Action, more Death Metal, etc.)
    4) Are there items that are more removed from the top-k than others? Which is their category? (Subplots)
    5) Which are the category of items that are removed the most from the top-k list across demographic groups?
    6) Which are the features and ids of the users and items present in most of removed edges?
    """

    _plot_methods_prefix = "plot_"

    def __init__(self, base_exps_file, train_data, sensitive_attributes=None, out_base_path='', best_exp=None):
        exps = utils.load_exps_file(base_exps_file)

        self.desc_data = []
        self.exps = exps
        self.train_data = train_data
        self.sens_attrs = ['gender', 'age'] if sensitive_attributes is None else sensitive_attributes
        self.best_exp = best_exp if best_exp is not None else ["loss_total"]

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
        paths_metadata = base_exps_file.split(os.sep)[(-4 if 'Fair' in base_exps_file else -3):]
        self.out_base_path = out_base_path or os.path.join(script_path,
                                                           os.pardir,
                                                           os.pardir,
                                                           'plots',
                                                           *paths_metadata)
        if not os.path.exists(self.out_base_path):
            os.makedirs(self.out_base_path)

        self.extract_desc_data()
        self.user_index = dict.fromkeys(self.user_df.user_id.tolist(), -1)
        for i, data in enumerate(self.desc_data):
            if self.user_index[data[0]] != -1:
                self.user_index[data[0]].append(i)
            else:
                self.user_index[data[0]] = [i]

    def extract_desc_data(self):
        if not self.desc_data:
            top_exp_col = [utils.EXPS_COLUMNS.index(be) for be in self.best_exp] if self.best_exp is not None else None
            user_num = max(self.exps.keys())
            for user in self.exps:
                user_exps = self.exps[user]
                if top_exp_col is not None and user_exps:
                    for tec in top_exp_col:
                        user_exps = sorted(user_exps, key=lambda x: x[tec])
                    user_exps = [user_exps[0]]
                for exp in user_exps:
                    desc_user = [user] + exp[1:3] + [exp[-3]]

                    # check if del_edges is not in right order, all users ids in edge[0] (row)
                    if desc_user[-1].shape == (1, 2) or \
                            desc_user[-1].shape[0] > 2 or \
                            (desc_user[-1].shape == (2, 2) and (desc_user[-1] > user_num).sum(axis=1)[0] > 0):
                        desc_user[-1] = desc_user[-1].T

                    # remap item ids
                    desc_user[-1][1] -= user_num

                    self.desc_data.append(desc_user)

    def plot(self, how="all"):
        plots = []
        if isinstance(how, Iterable) and not isinstance(how, str):
            plots = self.filter_supported_plots(how)
        elif how == "all":
            plots = self.supported_plots()
        else:
            raise ValueError(f"how = {how} is not supported")

        for plot in plots:
            plot_f = getattr(Describer, plot)
            plot_f(self)

    def plot_handle(func):
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            plt.close("all")

        return wrapper

    @staticmethod
    def _plot_for_each_sens(rows, cols, attr, sens_map, plot_df,
                            topk_cats=None, plot_type="barplot", ascending=None, x_map=None,
                            rotation=0, **sns_kwargs):
        plot_idx = cols + 1
        for r in range(2, rows + 1):
            row_axs = []
            for col in range(1, cols + 1):
                _ax = plt.subplot(rows, cols, plot_idx, sharey=row_axs[-1] if row_axs else None)
                _ax.set_title(f"Only users with {attr} = {sens_map[(r - 1) + (col - 1)]}")

                sens_plot_df = plot_df[plot_df['sens_exp'] == ((r - 1) + (col - 1))]

                if topk_cats is not None:
                    sens_plot_df = sens_plot_df.groupby(sns_kwargs['x']).sum().reset_index()

                order = None
                if ascending is not None:
                    order = sens_plot_df.sort_values(sns_kwargs['y'], ascending=ascending)[sns_kwargs['x']].unique()
                getattr(sns, plot_type)(**sns_kwargs, data=sens_plot_df, ax=_ax, order=order)
                if x_map is not None:
                    _ax.set_xticklabels([x_map[int(x.get_text())] for x in _ax.get_xticklabels()])
                _ax.tick_params(axis='x', rotation=rotation)
                plot_idx += 1
                row_axs.append(_ax)

    @plot_handle
    def plot_demo_groups_over_del_edges(self):
        """
        1) Distribution of demographic groups representation in removed edges (more females, more males, etc.)
        :return:
        """
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
                zip(user_sens, exp_edges_counts, sens_edges_counts),
                columns=['sens_exp', 'edges_counts', 'sens_edges']
            )
            plot_df = plot_df.groupby(['sens_exp', 'sens_edges']).sum().reset_index()

            rows = 1 + int(np.ceil((len(sens_map) - 1) / 3))
            cols = min(3, len(sens_map) - 1)

            fig = plt.figure(figsize=(12, 12))
            fig.suptitle("Distribution Demo Groups over Deleted Edges")

            ax1 = plt.subplot(rows, cols, (1, cols))
            ax1.set_title("All Explained Users")
            sns.barplot(y="edges_counts", hue="sens_edges", data=plot_df, ax=ax1)

            self._plot_for_each_sens(rows, cols, attr, sens_map, plot_df,
                                     y="edges_counts", hue="sens_edges", plot_type="barplot", x_map=None)

            plt.tight_layout()
            fig.savefig(os.path.join(self.out_base_path, f'({attr})_groups_over_del_edges.png'))

    @plot_handle
    def plot_del_edges_dist_over_demo_groups(self):
        """
        2) Distribution of number of removed edges for demographic group of explained users
        :return:
        """
        for attr, sens_map in zip(self.sens_attrs, self.sens_maps):
            user_sens = self._get_user_sens_attr_from_id([data[0] for data in self.desc_data], attr)
            n_del_edges = [data[-1].shape[1] for data in self.desc_data]

            plot_df = pd.DataFrame(zip(user_sens, n_del_edges), columns=['sens_exp', 'n_del_edges'])

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            fig.suptitle("Distribution of Number of Deleted Edges over Explained Demo Groups")

            sns.boxplot(y="n_del_edges", x="sens_exp", data=plot_df, ax=ax)
            ax.set_xticklabels([sens_map[int(x.get_text())] for x in ax.get_xticklabels()])

            plt.tight_layout()
            fig.savefig(os.path.join(self.out_base_path, f'({attr})_del_edges_dist_over_demo_groups.png'))

    @plot_handle
    def plot_item_categories_over_del_edges(self):
        """
        3) Distribution of item categories representation in removed edges (more Action, more Death Metal, etc.)
        :return:
        """
        for attr, sens_map in zip(self.sens_attrs, self.sens_maps):
            user_sens = self._get_user_sens_attr_from_id([data[0] for data in self.desc_data], attr)
            del_edges = [data[-1] for data in self.desc_data]

            cat_edges_counts = []
            item_cats_edges_counts = []
            repeats = []
            for edges in del_edges:
                item_cats_edges = self._get_item_categories_from_id(edges[1], flat=True)
                uni, counts = np.unique(item_cats_edges, return_counts=True)
                item_cats_edges_counts.append(counts)
                cat_edges_counts.append(uni)
                repeats.append(len(counts))
            item_cats_edges_counts = np.concatenate(item_cats_edges_counts)
            cat_edges_counts = np.concatenate(cat_edges_counts)

            user_sens = np.repeat(user_sens, repeats)

            plot_df = pd.DataFrame(
                zip(user_sens, item_cats_edges_counts, cat_edges_counts),
                columns=['sens_exp', 'edges_counts', 'cat_edges']
            )
            rows = 1 + int(np.ceil((len(sens_map) - 1) / 3))
            cols = min(2, len(sens_map) - 1)

            fig = plt.figure(figsize=(12, 12))
            fig.suptitle("Distribution Item Categories over Deleted Edges")

            all_users_plot_df = plot_df.groupby('cat_edges').sum().reset_index()
            ax1 = plt.subplot(rows, cols, (1, cols))
            ax1.set_title("All Explained Users")
            sns.barplot(y="edges_counts", x="cat_edges", data=all_users_plot_df, ax=ax1,
                        order=all_users_plot_df.sort_values("edges_counts", ascending=False)['cat_edges'].unique())
            ax1.set_xticklabels([self.item_class_map[int(x.get_text())] for x in ax1.get_xticklabels()])
            ax1.tick_params(axis='x', rotation=45)

            self._plot_for_each_sens(rows, cols, attr, sens_map, plot_df.groupby(['sens_exp', 'cat_edges']).sum().reset_index(),
                                     x="cat_edges", y="edges_counts", plot_type="barplot", x_map=self.item_class_map, ascending=False,
                                     rotation=45)

            plt.tight_layout()
            fig.savefig(os.path.join(self.out_base_path, f'({attr})_item_categories_over_del_edges.png'))

    @plot_handle
    def plot_dist_removed_items_topk_and_categories(self, topk_removed=10):
        """
        4) Are there items that are more removed from the top-k than others? Which is their category? (Subplots)
        5) Which are the category of items that are removed the most from the top-k list across demographic groups?
        :param topk_removed:
        :return:
        """
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
                if len(item_cats_topk) > 0:
                    for cat_topk in item_cats_topk:
                        item_cats_topk_counts.append(cat_topk)
                repeats.append(len(del_items))
            del_items_topk = np.concatenate(del_items_topk)

            user_sens = np.repeat(user_sens, repeats)

            plot_df = pd.DataFrame(
                zip(user_sens, del_items_topk, del_items_topk, item_cats_topk_counts),
                columns=['sens_exp', 'del_topk_items', 'item_count', 'del_topk_cats']
            )
            plot_df['item_count'] = plot_df['item_count'].map(plot_df['item_count'].value_counts())
            plot_df.sort_values('item_count', ascending=False, inplace=True)
            top_items_removed = plot_df['del_topk_items'].drop_duplicates()[:topk_removed].values
            topk_plot_df = plot_df[plot_df['del_topk_items'].isin(top_items_removed)]

            rows = 1 + int(np.ceil((len(sens_map) - 1) / 3))
            cols = min(2, len(sens_map) - 1)

            fig = plt.figure(figsize=(12, 12))
            fig.suptitle("Top Removed Items and Categories from Top-k List")

            ax1 = plt.subplot(rows, 2, 1)
            ax1.set_title("All Explained Users")
            sns.barplot(x="del_topk_items", y="item_count", ci=None, data=topk_plot_df, ax=ax1,
                        order=topk_plot_df.sort_values("item_count", ascending=False)['del_topk_items'].unique())

            cats_plot_df_expl = topk_plot_df.explode("del_topk_cats")
            cats_plot_df_expl["item_count"] = 1
            cats_plot_df = cats_plot_df_expl.groupby("del_topk_cats").sum().reset_index()
            ax2 = plt.subplot(rows, 2, 2)
            ax2.set_title("All Explained Users")
            sns.barplot(x="del_topk_cats", y="item_count", ci=None, data=cats_plot_df, ax=ax2,
                        order=cats_plot_df.sort_values("item_count", ascending=False)['del_topk_cats'].unique())
            ax2.set_xticklabels([self.item_class_map[int(x.get_text())] for x in ax2.get_xticklabels()])
            ax2.tick_params(axis='x', rotation=45)

            self._plot_for_each_sens(rows, cols, attr, sens_map, cats_plot_df_expl, topk_cats=topk_removed, plot_type="barplot",
                                     x="del_topk_cats", y="item_count", ci=None, ascending=False, x_map=self.item_class_map, rotation=45)

            plt.tight_layout()
            fig.savefig(os.path.join(self.out_base_path, f'({attr})_dist_removed_items_topk_and_categories.png'))

    @plot_handle
    def plot_dist_removed_users_items(self, top_removed=10):
        """
        6) Which are the features and ids of the users and items present in most of removed edges?
        :param top_removed:
        :return:
        """
        del_edges = [data[-1] for data in self.desc_data]

        user_edges, item_edges = np.concatenate(del_edges, axis=1)

        fig = plt.figure(figsize=(12, 12))
        fig.suptitle("Users and Items Most Present in Removed Edges")

        zip_data = [
            [user_edges, item_edges],
            ['user_edges', 'item_edges'],
            ['user_edges_counts', 'item_edges_counts'],
            ["Users of Removed Edges", "Items of Removed Edges"],
            [self.user_hist_len, self.item_hist_len]
        ]

        for i, (edge, col_edge, col_edge_counts, ax_title, hist_len) in enumerate(zip(*zip_data)):
            plot_df = pd.DataFrame(
                zip(edge, edge),
                columns=[col_edge, col_edge_counts]
            )
            plot_df[col_edge_counts] = plot_df[col_edge_counts].map(plot_df[col_edge_counts].value_counts())
            plot_df.sort_values(col_edge_counts, ascending=False, inplace=True)
            top_edges = plot_df[col_edge].drop_duplicates()[:top_removed].values
            top_plot_df = plot_df[plot_df[col_edge].isin(top_edges)]

            order_plots = top_plot_df.sort_values(col_edge_counts, ascending=False)[col_edge].unique()
            ax = plt.subplot(2, 2, i + 1)
            ax.set_title(ax_title)
            sns.barplot(x=col_edge, y=col_edge_counts, ci=None, data=top_plot_df, ax=ax, order=order_plots)

            plot_df = pd.DataFrame(
                zip(edge, edge),
                columns=[col_edge, col_edge_counts]
            )

            plot_df[col_edge_counts] = plot_df[col_edge].map(lambda x: hist_len[x])
            top_plot_df = plot_df[plot_df[col_edge].isin(top_edges)]

            ax = plt.subplot(2, 2, i + 1 + 2)
            ax.set_title("History Len of " + ax_title)
            sns.barplot(x=col_edge, y=col_edge_counts, ci=None, data=top_plot_df, ax=ax, order=order_plots)

        plt.tight_layout()
        fig.savefig(os.path.join(self.out_base_path, f'dist_removed_users_items.png'))

    def _get_user_sens_attr_from_id(self, user_ids, sens_attr):
        user_ids = user_ids if isinstance(user_ids, Iterable) else [user_ids]
        return self.user_df.set_index('user_id').loc[user_ids, sens_attr].to_numpy()

    def _get_item_categories_from_id(self, item_ids, flat=False):
        item_ids = item_ids if isinstance(item_ids, Iterable) else [item_ids]
        if flat:
            return np.array([cl for cl_list in self.item_df.set_index('item_id').loc[item_ids, "class"] for cl in cl_list])
        else:
            return [cl_list for cl_list in self.item_df.set_index('item_id').loc[item_ids, "class"].tolist()]

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', required=True)
    parser.add_argument('--base_exps_file', required=True)
    parser.add_argument('--explainer_config_file', default='../../config/gcmc_explainer.yaml')
    parser.add_argument('--plot_types', default="all")
    parser.add_argument('--out_base_path', default="")
    parser.add_argument('--best_exp', nargs='+', default=["loss_total"])

    args = parser.parse_args()

    script_path = os.path.abspath(os.path.dirname(inspect.getsourcefile(lambda: 0)))

    print(args)

    config, model, dataset, train_data, valid_data, test_data = utils.load_data_and_model(args.model_file,
                                                                                          args.explainer_config_file)
    describer = Describer(
        args.base_exps_file,
        train_data,
        sensitive_attributes=config['sensitive_attributes'],
        out_base_path=args.out_base_path,
        best_exp=args.best_exp
    )
    describer.plot(args.plot_types)
