# %%
import os
import pickle
import argparse
import inspect
from collections import defaultdict

import wandb
import numpy as np
import pandas as pd
import igraph as ig
import networkx as nx
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_tick
from adjustText import adjust_text
from matplotlib.lines import Line2D
from recbole.evaluator import Evaluator

import biga.utils as utils
import biga.utils.plot_utils as plot_utils


# %%
def get_plots_path():
    plots_path = os.path.join(
        script_path,
        os.pardir,
        f'dp_plots',
        dataset.dataset_name,
        model_name,
        '_'.join(map(str, args.best_exp_col)) if isinstance(args.best_exp_col, list) else \
            ('_'.join(map(str, list(args.best_exp_col.items()))) if isinstance(args.best_exp_col, dict) else args.best_exp_col),
    )

    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    return plots_path


# %%
def plot_lineplot_per_epoch_per_group(res_epoch_group,
                                      del_edges_epoch,
                                      fair_loss_per_epoch,
                                      orig_ndcg,
                                      test_orig_ndcg=None,
                                      data_info="test"):
    columns = ["Epoch", "Group", "metric", "value"]

    df_test_result = None
    orig_m_ndcg, orig_f_ndcg = orig_ndcg
    for e_type in res_epoch_group:
        plots_path = os.path.join(get_plots_path(), 'comparison', f"epochs_{epochs}", load_config_id, sens_attr)
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)

        group_map = train_data.dataset.field2id_token[sens_attr]
        df_data = [
            [0, m_idx, 'ndcg', orig_m_ndcg],
            [0, f_idx, 'ndcg', orig_f_ndcg]
        ]
        del_edge_data = [
            [0, m_idx, np.array([[]])],
            [0, f_idx, np.array([[]])]
        ]
        for epoch in res_epoch_group[e_type]:
            for gr, gr_res in res_epoch_group[e_type][epoch].items():
                for metric in gr_res:
                    df_data.append([epoch, gr, metric, gr_res[metric]])
                del_edge_data.append([epoch, gr, del_edges_epoch[e_type][epoch][gr]])

        df = pd.DataFrame(df_data, columns=columns)
        df["Group"] = df["Group"].map(group_map.__getitem__)
        if real_group_map is not None:
            df["Group"] = df["Group"].map(real_group_map.__getitem__)

        if data_info != "test":
            _test_orig_m_ndcg, _test_orig_f_ndcg = test_orig_ndcg
            df_test_result_data = [
                [0, m_idx, 'ndcg', _test_orig_m_ndcg],
                [0, f_idx, 'ndcg', _test_orig_f_ndcg]
            ]
            for epoch in test_result_per_epoch_per_group[e_type]:
                for gr, gr_res in test_result_per_epoch_per_group[e_type][epoch].items():
                    for metric in gr_res:
                        df_test_result_data.append([epoch, gr, metric, gr_res[metric]])

            df_test_result = pd.DataFrame(df_test_result_data, columns=columns)
            df_test_result["Group"] = df_test_result["Group"].map(group_map.__getitem__)
            if real_group_map is not None:
                df_test_result["Group"] = df_test_result["Group"].map(real_group_map.__getitem__)

        df_del_data = pd.DataFrame(del_edge_data, columns=["Epoch", "Group", "del_edges"])
        df_del_data["Group"] = df_del_data["Group"].map(group_map.__getitem__)
        if real_group_map is not None:
            df_del_data["Group"] = df_del_data["Group"].map(real_group_map.__getitem__)
        df_del_data[edges_ylabel] = df_del_data["del_edges"].map(lambda x: x.shape[1])
        df_del_data[edges_ylabel + "Lab"] = (df_del_data[edges_ylabel] / train_data.dataset.inter_num * 100).map("{:.2f}%".format)
        df_del_data.sort_values(edges_ylabel, inplace=True)

        rec_str = exp_rec_data.title() if data_info != 'test' else "Test"
        for metric in evaluator.metrics:
            metr_str = metric.upper()

            fig = plt.figure(figsize=(10, 10))
            plot_df = df[df["metric"] == metric].copy()
            plot_df.rename(columns={"value": metr_str}, inplace=True)

            rec_m_val = plot_df.loc[plot_df['Group'] == m_label].sort_values("Epoch")[metr_str].to_numpy()
            rec_f_val = plot_df.loc[plot_df['Group'] == f_label].sort_values("Epoch")[metr_str].to_numpy()
            rec_intersects = np.argwhere(np.diff(np.sign(rec_m_val - rec_f_val))).flatten()
            rec_x_intersects = plot_df.loc[plot_df['Group'] == m_label].sort_values("Epoch")["Epoch"].iloc[rec_intersects].to_numpy()

            if data_info != "test":
                plot_test_df = df_test_result[df_test_result["metric"] == metric].copy()
                plot_test_df.rename(columns={"value": metr_str}, inplace=True)

                test_m_val = plot_test_df.loc[plot_test_df['Group'] == m_label].sort_values("Epoch")[metr_str].to_numpy()
                test_f_val = plot_test_df.loc[plot_test_df['Group'] == f_label].sort_values("Epoch")[metr_str].to_numpy()
                test_intersects = np.argwhere(np.diff(np.sign(test_m_val - test_f_val))).flatten()
                test_x_intersects = plot_test_df.loc[plot_test_df['Group'] == m_label].sort_values("Epoch")["Epoch"].iloc[test_intersects].to_numpy()
            else:
                plot_test_df, test_m_val, test_f_val, test_intersects, test_x_intersects = [None] * 5

            colors = sns.color_palette("colorblind", n_colors=2)

            if data_info != "test":
                gs = plt.GridSpec(8, 6)

                ax_rec = fig.add_subplot(gs[2:5, :])
                ax_test = fig.add_subplot(gs[5:, :], sharex=ax_rec)
                ax_metric_diff = fig.add_subplot(gs[1, :], sharex=ax_rec)
                ax_del_edges = fig.add_subplot(gs[0, :], sharex=ax_rec)

                plot_utils.off_margin_ticks(ax_rec, ax_metric_diff, ax_del_edges)
            else:
                gs = plt.GridSpec(6, 6)

                ax_rec = fig.add_subplot(gs[2:, :])
                ax_test = None
                ax_metric_diff = fig.add_subplot(gs[1, :], sharex=ax_rec)
                ax_del_edges = fig.add_subplot(gs[0, :], sharex=ax_rec)

                plot_utils.off_margin_ticks(ax_metric_diff, ax_del_edges)

            df_diff = plot_df.loc[plot_df["Group"] == m_label, ["Epoch"]].copy()
            df_diff[metr_str] = np.abs(
                plot_df.loc[plot_df["Group"] == m_label, metr_str].values -
                plot_df.loc[plot_df["Group"] == f_label, metr_str].values
            )
            df_diff.rename(columns={metr_str: f"{metr_str} Diff"}, inplace=True)
            df_diff["Source Eval Data"] = f"{metr_str} {rec_str}"

            if metric == "ndcg":
                df_fair_loss_data = []
                for epoch in fair_loss_per_epoch[e_type]:
                    df_fair_loss_data.append([epoch, fair_loss_per_epoch[e_type][epoch]])

                df_fair_loss_df = pd.DataFrame(df_fair_loss_data, columns=["Epoch", f"{metr_str} Diff"])

                df_fair_loss_df["Source Eval Data"] = "ApproxNDCG Test"

                df_diff = pd.concat([df_diff, df_fair_loss_df]).reset_index()

            if data_info != "test":
                df_test_diff = plot_test_df.loc[plot_test_df["Group"] == m_label, ["Epoch", "Group"]].copy()
                df_test_diff[metr_str] = np.abs(
                    plot_test_df.loc[plot_test_df["Group"] == m_label, metr_str].values -
                    plot_test_df.loc[plot_test_df["Group"] == f_label, metr_str].values
                )
                df_test_diff.rename(columns={metr_str: f"{metr_str} Diff"}, inplace=True)

                df_test_diff["Source Eval Data"] = "Test"
                df_diff = pd.concat([df_diff, df_test_diff]).reset_index()

            lines = []
            for ax, data_df, data_type in zip([ax_rec, ax_test], [plot_df, plot_test_df], [rec_str, "Test"]):
                if ax is not None:
                    ax_lines = sns.lineplot(x="Epoch", y=metr_str, data=data_df, hue="Group", palette=colors, hue_order=[m_label, f_label], ax=ax)
                    lines.append(ax_lines)

            diff_groups = df_diff.groupby("Source Eval Data")
            # diff_colors = sns.color_palette("colorblind")[::-1][:diff_groups.ngroups]
            diff_colors = sns.color_palette("colorblind6", n_colors=diff_groups.ngroups)

            sns.lineplot(x="Epoch", y=f"{metr_str} Diff", data=df_diff, palette=diff_colors, hue="Source Eval Data", ax=ax_metric_diff)
            # for i, (eval_type, eval_diff_df) in enumerate(diff_groups):
            #     zorder = 1 if eval_type == "ApproxNDCG Loss" else 2
            #     ax_metric_diff.fill_between(eval_diff_df["Epoch"], eval_diff_df[f"{metr_str} Diff"], color=diff_colors[i], alpha=0.3, zorder=zorder)
            ax_metric_diff.grid(axis='y', ls=':')
            ax_metric_diff.legend(*ax_metric_diff.get_legend_handles_labels(), ncol=3)

            sns.lineplot(x="Epoch", y=edges_ylabel, hue="Group", data=df_del_data, palette=colors, hue_order=[m_label, f_label], ax=ax_del_edges)

            df_del_data_epoch_group = df_del_data.set_index(["Epoch", "Group"])
            max_epoch = df_del_data["Epoch"].max()
            if df_del_data_epoch_group.loc[(max_epoch, f_label), edges_ylabel] <= df_del_data_epoch_group.loc[(max_epoch, m_label), edges_ylabel]:
                lower_del_edges_group = f_label
            else:
                lower_del_edges_group = m_label
            ax_del_edges.annotate(
                df_del_data_epoch_group.loc[(max_epoch, lower_del_edges_group), edges_ylabel + "Lab"],
                (max_epoch, df_del_data_epoch_group.loc[(max_epoch, lower_del_edges_group), edges_ylabel])
            )

            ax_del_edges.yaxis.set_major_formatter(mpl_tick.FuncFormatter(lambda x, pos: f"{x / train_data.dataset.inter_num * 100:.2f}%"))

            texts = []
            for ax, m_val, f_val, intersects, x_intersects in zip([ax_rec, ax_test],
                                                                          [rec_m_val, test_m_val],
                                                                          [rec_f_val, test_f_val],
                                                                          [rec_intersects, test_intersects],
                                                                          [rec_x_intersects, test_x_intersects]):
                if ax is not None:
                    y_scatter = []
                    texts.append([])
                    for i, (x_cross, cross) in enumerate(zip(x_intersects, intersects)):
                        mean_metric = (m_val[cross] + f_val[cross]) / 2
                        diff_metric = abs(m_val[cross] - f_val[cross])
                        texts[-1].append(ax.text(x_cross, mean_metric, f"{mean_metric:.4f} ({diff_metric:.4f})"))
                        y_scatter.append(mean_metric)

                    if len(intersects) > 0:
                        ax.scatter(x_intersects, y_scatter, marker="X", c="k")

            if metric == "ndcg":
                if test_orig_ndcg is not None:
                    _test_orig_m_ndcg, _test_orig_f_ndcg = test_orig_ndcg
                    x_lim = [plot_test_df["Epoch"].min(), plot_test_df["Epoch"].max()]
                    ax_test.plot(x_lim, [_test_orig_m_ndcg, _test_orig_m_ndcg], c=colors[0], ls='--')
                    ax_test.plot(x_lim, [_test_orig_f_ndcg, _test_orig_f_ndcg], c=colors[1], ls='--')

            for ax, ax_texts, ax_lines in zip([ax_rec, ax_test], texts, lines):
                ax.minorticks_on()
                ax.grid(True, which="both", ls=':')

                adjust_text(
                    ax_texts,
                    only_move={'points': 'xy', 'text': 'xy'},
                    arrowprops=dict(arrowstyle="->", color='k', lw=0.5),
                    ax=ax,
                    expand_text=(1.1, 1.2),
                    expand_points=(1.1, 2.5),
                    expand_objects=(0.3, 0.2),
                    add_objects=ax_lines.lines[:2]
                )

            fig.suptitle(title)

            # utils.legend_without_duplicate_labels(ax)

            plt.tight_layout()
            final_plot_path = os.path.join(plots_path, f'{data_info}_lineplot_per_epoch_per_group_{e_type}_{metric}.png')
            plt.savefig(final_plot_path)
            wandb.log({f'Lineplot Per Epoch Per Group ({e_type} {data_info} {metric})': wandb.Image(final_plot_path)})
            plt.close()


# %%
def create_lineplot_metrics_over_del_edges(_result_all_data, _pref_dfs, orig_result, config_id, n_bins=10, hist_type="test", test_f="f_oneway"):
    order = [model_name, model_dp_s]

    P_005 = '*'
    P_001 = '^'

    for metric in metrics_names:
        plots_path = os.path.join(get_plots_path(), 'comparison', f"epochs_{epochs}", config_id, sens_attr)
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)

        # e_type with highest number of explanations
        max_del_edges_e_type = max([(k, len(x)) for k, x in _result_all_data.items()], key=lambda v: v[1])[0]
        del_edges = sorted(list(_result_all_data[max_del_edges_e_type]))

        bin_size = max(del_edges) // n_bins
        bin_map = {i: f"{i * bin_size + 1 if i != 0 else 1}-{(i + 1) * bin_size}" for i in
                   range(max(del_edges) // bin_size + 1)}

        del_edges_map = {x: bin_map[x // bin_size] for x in del_edges}

        d_grs = np.arange(1, len(attr_map))

        exp_data, stats_data, final_bins = plot_utils.compute_exp_stats_data(
            _result_all_data, _pref_dfs, orig_result, order, sens_attr, user_df, d_grs, del_edges_map, metric, test_f=test_f
        )

        exp_data = np.asarray(exp_data)

        final_bins = ['-'.join(map(lambda x: f"{int(x) / train_data.dataset.inter_num * 100:.2f}%", _bin.split('-')))
                      for _bin in final_bins]

        df_attr = pd.DataFrame(zip(
            exp_data.flatten(),
            np.repeat(order, len(final_bins)),
            np.tile(final_bins, len(order))
        ), columns=['DP', 'model', '% Del Edges'])

        sns.lineplot(x='% Del Edges', y='DP', hue='model', data=df_attr)

        plt.tight_layout()
        final_plot_path = os.path.join(plots_path, f"{hist_type}_lineplot_over_edges_{sens_attr}_{metric}.png")
        plt.savefig(final_plot_path)
        wandb.log({f'Lineplot Over Edges ({sens_attr} {hist_type} {metric})': wandb.Image(final_plot_path)})
        plt.close()


# %%
def create_metric_access_over_del_edges_per_group(
    _result_all_data,
    _pref_dfs,
    orig_result,
    config_id,
    hist_type="test",
    zerometric=False,
    n_bins=6
):
    pref_df = _pref_dfs[model_dp_s][['user_id', 'n_del_edges']]

    uid_list = next(pref_df.groupby('n_del_edges').__iter__())[1].user_id.to_numpy()
    joint_df = user_df.join(pref_df.set_index("user_id"), on="user_id", how='right').reset_index(drop=True)
    joint_df[sens_attr] = joint_df[sens_attr].map(attr_map.__getitem__)

    plots_path = os.path.join(get_plots_path(), 'comparison', f"epochs_{epochs}", config_id, sens_attr)
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    for metric in metrics_names:
        metric_df_index = ['user_id', 'n_del_edges']
        metric_df_cols = ['user_id', metric.upper(), 'n_del_edges', 'Graph Type']

        if zerometric:
            zerometric_s = "_zero_metric"
            zero_users = uid_list[orig_result[metric][:, -1] == 0]
        else:
            zerometric_s = ""
            zero_users = None

        fig, axs = plt.subplots(1 + zerometric, joint_df[sens_attr].unique().shape[0], figsize=(20, 12), sharey=True)
        for i, (group, gr_df) in enumerate(joint_df.groupby(sens_attr)):
            del_edges_perc = np.sort(np.fromiter(_result_all_data[model_dp_s].keys(), int))
            del_edges_perc = dict(zip(
                del_edges_perc,
                [f"{x / train_data.dataset.inter_num * 100:.2f}%" for x in del_edges_perc]
            ))

            gr_df['n_del_edges'] = gr_df['n_del_edges'].map(del_edges_perc)

            plot_metric_df = [[], []]
            for n_del in _result_all_data[model_dp_s]:
                for gt_i, (gt, m_data) in enumerate(zip(["Perturbed", "Original"], [_result_all_data[model_dp_s][n_del], orig_result])):
                    plot_metric_df[gt_i].extend(list(zip(
                        uid_list,
                        m_data[metric][:, -1].tolist(),
                        [del_edges_perc[n_del]] * len(uid_list),
                        [gt] * len(uid_list)
                    )))

            for gt_i, gt in enumerate(["Perturbed", "Original"]):
                plot_metric_df[gt_i] = pd.DataFrame(plot_metric_df[gt_i], columns=metric_df_cols)
                # plot_metric_df[gt_i][sens_attr] = gr_df.set_index(metric_df_index),
                #     on=metric_df_index
                # ).reset_index(drop=True)

            plot_df = pd.concat(plot_metric_df, ignore_index=True)
            plot_df.rename(columns={'n_del_edges': edges_ylabel}, inplace=True)

            del_edges_bins = np.array(list(del_edges_perc.values()))
            bins = np.linspace(0, del_edges_bins.shape[0] - 1, n_bins, dtype=int)
            del_edges_bins = del_edges_bins[bins]
            plot_df_gr = plot_df.groupby(edges_ylabel)
            plot_df = pd.concat([plot_df_gr.get_group(del_bin) for del_bin in del_edges_bins], ignore_index=True)

            if zerometric:
                mask = plot_df.user_id.isin(zero_users)
                plot_zero_df, plot_nonzero_df = plot_df[mask], plot_df[~mask]

                if not plot_zero_df.empty:
                    sns.boxplot(x=edges_ylabel, y=metric.upper(), hue="Graph Type", data=plot_zero_df, fliersize=0, ax=axs[0, i])
                    sns.lineplot(x=edges_ylabel, y=metric.upper(), hue="Graph Type",
                                 data=plot_zero_df[plot_zero_df["Graph Type"] == "Perturbed"], ax=axs[0, i])

                if not plot_nonzero_df.empty:
                    sns.boxplot(x=edges_ylabel, y=metric.upper(), hue="Graph Type", data=plot_nonzero_df, fliersize=0, ax=axs[1, i])
                    sns.lineplot(x=edges_ylabel, y=metric.upper(), hue="Graph Type",
                                 data=plot_nonzero_df[plot_nonzero_df["Graph Type"] == "Perturbed"], ax=axs[1, i])

                axs[0, i].set_title(f"{sens_attr.title()}: {group_name_map[real_group_map[group]]} with {metric.upper()} = 0")
                axs[1, i].set_title(f"{sens_attr.title()}: {group_name_map[real_group_map[group]]} with {metric.upper()} ≠ 0")
            else:
                sns.boxplot(x=edges_ylabel, y=metric.upper(), hue="Graph Type", data=plot_df, fliersize=0, ax=axs[i])
                sns.lineplot(x=edges_ylabel, y=metric.upper(), hue="Graph Type", data=plot_df[plot_df["Graph Type"] == "Perturbed"], ax=axs[i])

                axs[i].set_title(f"{sens_attr.title()}: {group_name_map[real_group_map[group]]}")

        fig.suptitle(title)

        plt.tight_layout()
        final_plot_path = os.path.join(plots_path, f"{hist_type}_{metric}_access_plot_over_edges_{sens_attr}{zerometric_s}.png")
        plt.savefig(final_plot_path)
        wandb.log({f'Access Plot Over Edges ({sens_attr} {hist_type} {metric})': wandb.Image(final_plot_path)})
        plt.close()


# %%
def create_user_user_homophily_plot_over_del_edges_per_group(
    _pref_dfs,
    config_id,
    hist_type="test"
):
    del_edges_perc = np.sort(_pref_dfs[model_dp_s]['n_del_edges'].unique())

    pref_df = _pref_dfs[model_dp_s][['user_id', 'n_del_edges', 'del_edges']]
    pref_df.rename(columns={'n_del_edges': edges_ylabel}, inplace=True)

    joint_df = user_df.join(pref_df.set_index("user_id"), on="user_id", how='right').reset_index(drop=True)
    joint_df[sens_attr] = joint_df[sens_attr].map(attr_map.__getitem__)

    sens_groups = joint_df[sens_attr].unique()

    orig_user_graph_df = plot_utils.get_user_user_data_sens_df(train_data.dataset, user_df, sens_attr, attr_map=attr_map.__getitem__)
    gr_sizes = (user_df[sens_attr].map(attr_map.__getitem__).value_counts() / (user_df.shape[0] - 1)).to_dict()
    orig_user_homophily = plot_utils.compute_homophily(orig_user_graph_df, gr_sizes, sens_attr)

    plots_path = os.path.join(get_plots_path(), 'comparison', f"epochs_{epochs}", config_id, sens_attr)
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    plot_df_cols = ['Graph Type', 'User-User Homophily', sens_attr, edges_ylabel]

    plot_data_df = [[], []]
    joint_df_gr = joint_df.groupby(edges_ylabel)
    for del_i, n_del in enumerate(del_edges_perc):
        del_df = joint_df_gr.get_group(n_del)
        pert_train_dataset = utils.get_dataset_with_perturbed_edges(del_df['del_edges'].iloc[0], train_data.dataset)

        pert_user_graph_df = plot_utils.get_user_user_data_sens_df(pert_train_dataset, user_df, sens_attr, attr_map=attr_map.__getitem__)
        del_user_df = user_df[user_df['user_id'].isin(pert_train_dataset.inter_feat[uid_field].unique().numpy())]
        pert_gr_sizes = (del_user_df[sens_attr].map(attr_map.__getitem__).value_counts() / (del_user_df.shape[0])).to_dict()

        pert_user_homophily = plot_utils.compute_homophily(pert_user_graph_df, pert_gr_sizes, sens_attr)

        for gr in sens_groups:
            for gt_i, (gt, hom_data) in enumerate(zip(["Perturbed", "Original"], [pert_user_homophily[gr], orig_user_homophily[gr]])):
                plot_data_df[gt_i].append((gt, hom_data, gr, n_del))

    for gt_i, gt in enumerate(["Perturbed", "Original"]):
        plot_data_df[gt_i] = pd.DataFrame(plot_data_df[gt_i], columns=plot_df_cols)

    plot_df = pd.concat(plot_data_df, ignore_index=True)
    plot_df[sens_attr] = plot_df[sens_attr].map(real_group_map)

    fig, axs = plt.subplots(1, sens_groups.shape[0], figsize=(20, 12))
    for i, gr in enumerate(sens_groups):
        sns.lineplot(x=edges_ylabel, y="User-User Homophily", hue="Graph Type",
                     data=plot_df[plot_df[sens_attr] == real_group_map[gr]], ax=axs[i], ci=None)
        axs[i].set_title(f"{sens_attr.title()}: {group_name_map[real_group_map[gr]]}")
        axs[i].xaxis.set_major_formatter(mpl_tick.FuncFormatter(lambda x, pos: f"{x / train_data.dataset.inter_num * 100:.2f}%"))

    fig.suptitle(title)

    plt.tight_layout()
    final_plot_path = os.path.join(plots_path, f"{hist_type}_user_user_homophily_plot_over_edges_per_{sens_attr}_group.png")
    plt.savefig(final_plot_path)
    wandb.log({f'User-User Homophily Over Edges ({sens_attr} {hist_type})': wandb.Image(final_plot_path)})
    plt.close()


# %%
def create_item_item_homophily_plot_over_del_edges_per_popularity(
    _pref_dfs,
    config_id,
    hist_type="test",
    short_head=0.05
):
    pop_label = 'Popularity'

    del_edges_perc = np.sort(_pref_dfs[model_dp_s]['n_del_edges'].unique())

    pref_df = _pref_dfs[model_dp_s][['user_id', 'n_del_edges', 'del_edges']]
    pref_df.rename(columns={'n_del_edges': edges_ylabel}, inplace=True)

    item_df = pd.DataFrame({
        'item_id': train_data.dataset.item_feat['item_id'].numpy()
    })
    orig_sh_pop = plot_utils.get_data_sh_lt(train_data, short_head=short_head)
    orig_sh_pop = {k + 1: v for k, v in orig_sh_pop.items()}
    item_df[pop_label] = item_df['item_id'].map(orig_sh_pop)

    pop_groups = list(set(orig_sh_pop.values()))

    orig_item_graph_df = plot_utils.get_item_item_data_pop_df(train_data.dataset, item_df, pop_label)
    gr_sizes = (item_df[pop_label].value_counts() / (item_df.shape[0] - 1)).to_dict()
    orig_item_homophily = plot_utils.compute_homophily(orig_item_graph_df, gr_sizes, pop_label)

    plots_path = os.path.join(get_plots_path(), 'comparison', f"epochs_{epochs}", config_id, sens_attr)
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    plot_df_cols = ['Graph Type', 'Item-Item Homophily', pop_label, edges_ylabel]

    plot_data_df = [[], []]
    pref_df_gr = pref_df.groupby(edges_ylabel)
    for del_i, n_del in enumerate(del_edges_perc):
        del_df = pref_df_gr.get_group(n_del)
        pert_train_dataset = utils.get_dataset_with_perturbed_edges(del_df['del_edges'].iloc[0], train_data.dataset)

        pert_item_graph_df = plot_utils.get_item_item_data_pop_df(pert_train_dataset, item_df, pop_label)
        del_item_df = item_df[item_df['item_id'].isin(pert_train_dataset.inter_feat[iid_field].unique().numpy())]
        pert_gr_sizes = (del_item_df[pop_label].value_counts() / (del_item_df.shape[0])).to_dict()

        pert_item_homophily = plot_utils.compute_homophily(pert_item_graph_df, pert_gr_sizes, pop_label)

        for gr in pop_groups:
            for gt_i, (gt, hom_data) in enumerate(zip(["Perturbed", "Original"], [pert_item_homophily[gr], orig_item_homophily[gr]])):
                plot_data_df[gt_i].append((gt, hom_data, gr, n_del))

    for gt_i, gt in enumerate(["Perturbed", "Original"]):
        plot_data_df[gt_i] = pd.DataFrame(plot_data_df[gt_i], columns=plot_df_cols)

    plot_df = pd.concat(plot_data_df, ignore_index=True)

    fig, axs = plt.subplots(1, len(pop_groups), figsize=(20, 12))
    for i, gr in enumerate(pop_groups):
        sns.lineplot(x=edges_ylabel, y="Item-Item Homophily", hue="Graph Type",
                     data=plot_df[plot_df[pop_label] == gr], ax=axs[i], ci=None)
        axs[i].set_title(f"{pop_label}: {gr}")
        axs[i].xaxis.set_major_formatter(mpl_tick.FuncFormatter(lambda x, pos: f"{x / train_data.dataset.inter_num * 100:.2f}%"))

    fig.suptitle(title)

    plt.tight_layout()
    final_plot_path = os.path.join(plots_path, f"{hist_type}_item_item_homophily_plot_over_edges_per_{sens_attr}_popularity.png")
    plt.savefig(final_plot_path)
    wandb.log({f'Item-Item Homophily Over Edges ({sens_attr} {hist_type})': wandb.Image(final_plot_path)})
    plt.close()


def create_bias_ratio_categories_over_groups_plot_per_del_edges(
    _pref_dfs,
    config_id,
    hist_type="test",
    n_bins=6
):
    del_edges_perc = np.sort(_pref_dfs[model_dp_s]['n_del_edges'].unique())
    bins = np.linspace(0, del_edges_perc.shape[0] - 1, n_bins, dtype=int)
    del_edges_perc = del_edges_perc[bins]

    item_categories_map = train_data.dataset.field2id_token['class'][1:]
    n_cats = len(item_categories_map)

    norm = mpl.colors.Normalize(vmin=0, vmax=del_edges_perc.max())
    cmap = sns.color_palette("Blues_d", as_cmap=True)

    pref_df = _pref_dfs[model_dp_s][['user_id', 'n_del_edges', 'del_edges', 'cf_topk_pred']]
    pref_df.rename(columns={'n_del_edges': edges_ylabel}, inplace=True)

    user_joint_df = user_df.join(pref_df.set_index("user_id"), on="user_id", how='right').reset_index(drop=True)
    user_joint_df[sens_attr] = user_joint_df[sens_attr].map(attr_map.__getitem__)

    orig_bias_ratio, orig_pref_ratio = utils.generate_bias_ratio(
        train_data,
        config,
        sensitive_attr=sens_attr,
        history_matrix=best_test_pref_data[model_dp_s],
        pred_col='topk_pred',
        mapped_keys=True
    )

    sens_groups = [gr for gr in orig_bias_ratio[sens_attr] if orig_bias_ratio[sens_attr][gr] is not None]

    item_df = pd.DataFrame({
        'item_id': train_data.dataset.item_feat['item_id'].numpy(),
        'class': item_class
    })

    train_df_joint = train_df.join(item_df.set_index('item_id'), on='item_id').join(user_df.set_index('user_id'), on='user_id')

    class_counts = train_df_joint.groupby(sens_attr).apply(lambda x: x.explode('class').value_counts('class'))
    class_counts.index = class_counts.index.map(lambda x: (x[0], item_categories_map[x[1] - 1]))
    cats_inter_counts = class_counts.reset_index().groupby(sens_attr).apply(lambda x: x[['class', 0]].set_index('class').to_dict()[0])
    cats_inter_counts.index = cats_inter_counts.index.map(attr_map.__getitem__)
    cats_inter_counts = cats_inter_counts.to_dict()

    order_cic, vals_cic = map(list, zip(*(sorted(cats_inter_counts[sens_groups[0]].items(), key=lambda x: x[1])[::-1])))
    inter_bar_data_df = {
        sens_groups[0]: pd.DataFrame(zip(order_cic, vals_cic), columns=['Category', '# Interactions']),
        sens_groups[1]: pd.DataFrame(
            zip(order_cic, [cats_inter_counts[sens_groups[1]][cat] if cat in cats_inter_counts[sens_groups[1]] else 0 for cat in order_cic]),
            columns=['Category', '# Interactions']
        )
    }

    order_cats = (item_categories_map[:, None] == np.array(order_cic)).nonzero()[1]

    plots_path = os.path.join(get_plots_path(), 'comparison', f"epochs_{epochs}", config_id, sens_attr)
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    plot_df_cols = ['Bias Ratio', 'Category', edges_ylabel, 'Group']

    orig_plot_data_df = []
    for gr in sens_groups:
        if orig_bias_ratio[sens_attr][gr] is not None:
            orig_plot_data_df += [*zip(
                orig_bias_ratio[sens_attr][gr][1:][order_cats], range(n_cats), [0.] * n_cats, [gr] * n_cats
            )]
    orig_plot_df = pd.DataFrame(orig_plot_data_df, columns=plot_df_cols)

    plot_data_df = []
    user_joint_df_gr = user_joint_df.groupby(edges_ylabel)
    for del_i, n_del in enumerate(del_edges_perc):
        del_df = user_joint_df_gr.get_group(n_del)
        pert_bias_ratio, pert_pref_ratio = utils.generate_bias_ratio(
            train_data,
            config,
            sensitive_attr=sens_attr,
            history_matrix=del_df,
            pred_col='cf_topk_pred',
            mapped_keys=True
        )

        for gr in sens_groups:
            if pert_bias_ratio[sens_attr][gr] is not None:
                plot_data_df += [*zip(
                    pert_bias_ratio[sens_attr][gr][1:][order_cats], range(n_cats), [n_del] * n_cats, [gr] * n_cats
                )]

    pert_plot_df = pd.DataFrame(plot_data_df, columns=plot_df_cols)

    fig = plt.figure(figsize=(20, 12))
    gs = plt.GridSpec(1, 10)
    ax_gr1 = fig.add_subplot(gs[:, :4])
    ax_gr2 = fig.add_subplot(gs[:, 5:9], sharey=ax_gr1)
    ax_bar_gr1 = fig.add_subplot(gs[:, 4], sharey=ax_gr1)
    ax_bar_gr2 = fig.add_subplot(gs[:, 9], sharey=ax_gr1)

    axs = [ax_gr1, ax_gr2]
    axs_bar = [ax_bar_gr1, ax_bar_gr2]
    orig_plot_df_gr = orig_plot_df.groupby("Group")
    for i, (gr, plot_gr) in enumerate(pert_plot_df.groupby("Group")):
        orig_gr = orig_plot_df_gr.get_group(gr)
        sns.scatterplot(y="Category", x="Bias Ratio", hue=edges_ylabel, size=edges_ylabel, palette=cmap,
                        sizes=(80, 250), hue_norm=norm, data=plot_gr, ax=axs[i])
        sns.scatterplot(y="Category", x="Bias Ratio", marker='X', color='#780808', s=150, data=orig_gr, ax=axs[i])
        axs[i].set_title(f"{sens_attr.title()}: {group_name_map[real_group_map[gr]]}")
        axs[i].yaxis.set_major_locator(mpl_tick.FixedLocator(list(range(n_cats))))
        axs[i].yaxis.set_major_formatter(mpl_tick.FixedFormatter([item_categories_map[x] for x in range(n_cats)]))
        handles, labels = axs[i].get_legend_handles_labels()
        axs[i].legend(
            [Line2D([0], [0], marker='X', color='w', markerfacecolor='#780808', markersize=15)] + handles,
            ["0.0%"] + [f"{int(x) / train_data.dataset.inter_num * 100:.2f}%" if x.isdigit() else x for x in labels]
        )
        axs[i].grid(axis='y', ls=':')
        axs[i].plot([1, 1], axs[i].get_ylim(), 'k--')

        sns.barplot(y="Category", x="# Interactions", data=inter_bar_data_df[gr], ax=axs_bar[i], color='black', order=order_cic)

    plot_utils.off_margin_ticks(ax_gr2, *axs_bar, axis='y')
    for _ax in [ax_gr2, *axs_bar]:
        _ax.set_ylabel('', labelpad=None)

    fig.suptitle(title)

    plt.tight_layout()
    final_plot_path = os.path.join(plots_path, f"{hist_type}_bias_ratio_over_edges_per_{sens_attr}.png")
    plt.savefig(final_plot_path)
    wandb.log({f'Bias Ratio Over Edges ({sens_attr} {hist_type})': wandb.Image(final_plot_path)})
    plt.close()


# %%
def create_distribution_diff_metric_random_groups(
    _result_all_data,
    _pref_dfs,
    orig_result,
    config_id,
    hist_type="test",
    iterations=100,
    n_bins=6
):
    pref_df = _pref_dfs[model_dp_s][['user_id', 'n_del_edges', 'epoch']]

    uid_list = next(pref_df.groupby('n_del_edges').__iter__())[1].user_id.to_numpy()
    u_sens_df = user_df.copy()
    u_sens_df[sens_attr] = u_sens_df[sens_attr].map(attr_map.__getitem__)

    plots_path = os.path.join(get_plots_path(), 'comparison', f"epochs_{epochs}", config_id, sens_attr)
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    for metric in metrics_names:
        metric_df_cols = ['user_id', metric.upper(), 'n_del_edges', 'Graph Type']

        plot_metric_df = [[], []]
        for n_del in _result_all_data[model_dp_s]:
            plot_metric_df[0].extend(list(zip(
                uid_list,
                _result_all_data[model_dp_s][n_del][metric][:, -1].tolist(),
                [n_del] * len(uid_list),
                ["Perturbed"] * len(uid_list)
            )))
        plot_metric_df[1].extend(list(zip(
            uid_list,
            orig_result[metric][:, -1].tolist(),
            [0] * len(uid_list),
            ["Original"] * len(uid_list)
        )))

        for gt_i, gt in enumerate(["Perturbed", "Original"]):
            plot_metric_df[gt_i] = pd.DataFrame(plot_metric_df[gt_i], columns=metric_df_cols)
            plot_metric_df[gt_i] = plot_metric_df[gt_i].join(
                u_sens_df.set_index('user_id'),
                on='user_id'
            ).reset_index(drop=True)

        plot_df = pd.concat(plot_metric_df, ignore_index=True)
        plot_df.rename(columns={'n_del_edges': edges_ylabel}, inplace=True)

        del_edges_bins = np.sort(plot_df[edges_ylabel].unique())
        bins = np.linspace(0, del_edges_bins.shape[0] - 1, n_bins, dtype=int)
        del_edges_bins = del_edges_bins[bins]

        dp_samples = []
        plot_df_gr = plot_df.groupby(edges_ylabel)

        for n_del in del_edges_bins:
            n_del_df = plot_df_gr.get_group(n_del)
            n_del_epoch = pref_df.loc[pref_df['n_del_edges'] == n_del, 'epoch'].iloc[0] if n_del > 0 else 0
            dp_samples.extend(list(zip(
                [n_del] * iterations,
                [n_del_epoch] * iterations,
                utils.compute_DP_across_random_samples(n_del_df, sens_attr, sens_attr, dataset.dataset_name, metric.upper(), batch_size=batch_exp, iterations=iterations)
            )))

        dp_samples_df = pd.DataFrame(dp_samples, columns=[edges_ylabel, "Epoch", f"{metric.upper()} DP"])
        dp_samples_df[edges_ylabel] = dp_samples_df[edges_ylabel].map(
            lambda x: f"{x / train_data.dataset.inter_num * 100:.2f}%"
        )

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        sns.lineplot(x=edges_ylabel, y=f"{metric.upper()} DP", data=dp_samples_df, ax=ax)

        twiny = ax.twiny()
        sns.lineplot(x="Epoch", y=f"{metric.upper()} DP", data=dp_samples_df, ax=twiny, alpha=0)
        twiny.spines["bottom"].set_position(("outward", 40))
        plot_utils.make_patch_spines_invisible(twiny)

        twiny.spines["bottom"].set_visible(True)
        twiny.xaxis.set_label_position('bottom')
        twiny.xaxis.set_ticks_position('bottom')
        twiny.set_xlabel('Epochs')

        ax.set_title(f"{sens_attr.title()} {metric.upper()} DP Across {iterations} Random Samples")

        fig.suptitle(title)

        plt.tight_layout()
        final_plot_path = os.path.join(plots_path, f"{hist_type}_{sens_attr}_{metric}_DP_across_{iterations}_random_samples.png")
        plt.savefig(final_plot_path)
        wandb.log({f'DP Across {iterations} Random Samples ({sens_attr} {hist_type} {metric})': wandb.Image(final_plot_path)})
        plt.close()


# %%
def plot_decomposition_perturbed(pref_df, tr_data, th=1e-5):
    plots_path = os.path.join(get_plots_path(), 'comparison', f"epochs_{epochs}", load_config_id, sens_attr)
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    train_pca, pert_train_pca = utils.get_decomposed_adj_matrix(pref_df, tr_data)
    sens_data = train_data.dataset.user_feat[sens_attr].numpy()[1:]
    sens_data = [group_name_map[real_group_map['F' if idx == f_idx else 'M']] for idx in sens_data]

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    for ax, data, ax_title in zip(axs, [train_pca, pert_train_pca], ['NoPolicy', policy]):
        if ax_title != 'NoPolicy':
            changes = np.abs(train_pca - pert_train_pca)
            mask = changes > th
            rel_chs, = np.bitwise_or.reduce(mask, axis=1).nonzero()
            irrel_chs, = np.bitwise_or.reduce(~mask, axis=1).nonzero()
            sns.scatterplot(x=data[irrel_chs, 0], y=data[irrel_chs, 1], hue=sens_data[irrel_chs], marker='o', ax=ax, alpha=0.2)
            sns.scatterplot(x=data[rel_chs, 0], y=data[rel_chs, 1], hue=sens_data[rel_chs], marker='o', ax=ax)
        else:
            sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=sens_data, marker='o', ax=ax)
        ax.set_title(ax_title)

    plt.tight_layout()
    final_plot_path = os.path.join(plots_path, f"{sens_attr}_adjacency_matrix_decomposition.png")
    plt.savefig(final_plot_path)
    wandb.log({f'PCA Decomposition of Adjacency Matrix ({sens_attr})': wandb.Image(final_plot_path)})


# %%
def create_table_metrics_over_del_edges(_result_all_data, _pref_dfs, orig_result, config_id, n_bins=10, hist_type="test", test_f="f_oneway"):
    order = [model_name, model_dp_s]

    P_005 = '*'
    P_001 = '^'

    for metric in metrics_names:
        tables_path = os.path.join(get_plots_path(), 'comparison', f"epochs_{epochs}", config_id, sens_attr)
        if not os.path.exists(tables_path):
            os.makedirs(tables_path)

        # e_type with highest number of explanations
        max_del_edges_e_type = max([(k, len(x)) for k, x in _result_all_data.items()], key=lambda v: v[1])[0]
        del_edges = sorted(list(_result_all_data[max_del_edges_e_type]))

        bin_size = max(del_edges) // n_bins
        bin_map = {i: f"{i * bin_size + 1 if i != 0 else 1}-{(i + 1) * bin_size}" for i in
                   range(max(del_edges) // bin_size + 1)}

        del_edges_map = {x: bin_map[x // bin_size] for x in del_edges}

        d_grs = np.arange(1, len(attr_map))

        exp_data, stats_data, final_bins = plot_utils.compute_exp_stats_data(
            _result_all_data, _pref_dfs, orig_result, order, sens_attr, user_df, d_grs, del_edges_map, metric, test_f=test_f
        )

        plot_vals = []
        for row, stat in zip(exp_data, stats_data):
            plot_vals.append([])
            for j in range(len(final_bins)):
                if row:
                    plot_vals[-1].append(f"{row[j]:.4f}"
                                         f"{P_001 if stat[j] < 0.01 else (P_005 if stat[j] < 0.05 else '')}")
                else:
                    plot_vals[-1].append("-")

        final_bins = ['-'.join(map(lambda x: f"{int(x) / train_data.dataset.inter_num * 100:.2f}%", bin.split('-')))
                      for bin in final_bins]

        df_attr = pd.DataFrame(plot_vals, columns=final_bins, index=order).T

        wandb.log({f"Table Over Edges ({hist_type} {sens_attr} {metric} {test_f})": df_attr})

        df_attr.to_markdown(os.path.join(tables_path, f"{hist_type}_table_over_edges_{sens_attr}_{metric}_{test_f}.md"), tablefmt="github")
        df_attr.to_latex(os.path.join(tables_path, f"{hist_type}_table_over_edges_{sens_attr}_{metric}_{test_f}.tex"), multirow=True)


# %%
def plot_dist_over_del_edges(_topk_dist_all, bd_all, config_id, max_del_edges=80):
    for e_type, _topk_dist in _topk_dist_all.items():
        bd = bd_all[e_type]
        bd_attrs = defaultdict(list)
        for n_del in bd:
            for d_gr, bd_d_gr in bd[n_del][sens_attr].items():
                if bd_d_gr is not None:
                    bd_attrs[sens_attr].extend(list(zip([n_del] * len(bd_d_gr), bd_d_gr.numpy(), [d_gr] * len(bd_d_gr))))

        bd_attrs = dict(bd_attrs)

        topk_dist_df = pd.DataFrame(_topk_dist, columns=['# Del Edges', 'Edit Dist', 'Set Dist'])

        for attr, bd_attr_data in bd_attrs.items():
            plots_path = os.path.join(get_plots_path(), 'comparison', f"epochs_{epochs}", config_id, attr)
            if not os.path.exists(plots_path):
                os.makedirs(plots_path)

            fig, axs = plt.subplots(2, 1, figsize=(8, 12), sharex=True)

            topk_dist_df_plot = topk_dist_df[topk_dist_df['# Del Edges'] <= max_del_edges]

            sns.lineplot(x='# Del Edges', y='Edit Dist', data=topk_dist_df_plot, label='Edit Dist', ax=axs[0], ci="sd")
            sns.lineplot(x='# Del Edges', y='Set Dist', data=topk_dist_df_plot, label='Set Dist', ax=axs[0], ci="sd")

            bd_attr_df = pd.DataFrame(bd_attr_data, columns=['# Del Edges', 'Bias Disparity', 'Demo Group']).dropna()

            bd_attr_df_plot = bd_attr_df[bd_attr_df['# Del Edges'] <= max_del_edges]

            sns.lineplot(x='# Del Edges', y='Bias Disparity', hue='Demo Group', data=bd_attr_df_plot, ax=axs[1], palette="colorblind", ci="sd")
            axs[1].xaxis.set_minor_locator(mpl_tick.AutoMinorLocator())

            axs[1].grid(which='both', axis='x')
            axs[1].plot(axs[1].get_xlim(), [0., 0.], 'k--')

            plt.tight_layout()
            final_plot_path = os.path.join(plots_path, f'edit_set_dist_over_del_edges_{e_type}.png')
            plt.savefig(final_plot_path)
            wandb.log({f'Edit Dist Over Edges ({e_type} {sens_attr})': wandb.Image(final_plot_path)})
            plt.close()


# %%
def plot_del_edges_hops(dfs):
    from scipy.sparse import coo_matrix

    color_0 = "red"
    color_1 = "blue"

    train_nx = utils.get_nx_biadj_matrix(train_data.dataset, remove_first_row_col=True)

    for e_type, e_df in dfs.items():
        _df = e_df[['user_id', 'del_edges']]

        del_edges_nx = nx.bipartite.biadjacency_matrix(coo_matrix(
            (np.ones((len(_df['del_edges']))), (*_df['del_edges'])),
            shape=(train_data.dataset.user_num(), train_data.dataset.item_num())
        ).tocsr()[1:, 1:])


# %%
parser = argparse.ArgumentParser()
parser.add_argument('--model_file', required=True)
parser.add_argument('--explainer_config_file', default=os.path.join("config", "explainer.yaml"))
parser.add_argument('--best_exp_col', nargs='+', default=["auto"])

args = parser.parse_args()  # r"--model_file saved\GCMC-ML-100K-Oct-02-2022_19-24-04.pth --explainer_config_file src\dp_ndcg_explanations\ml-100k\GCMC\FairDP\gender\epochs_1000\3\config.yaml".split())

script_path = os.path.abspath(os.path.dirname(inspect.getsourcefile(lambda: 0)))
script_path = os.path.join(script_path, 'src') if 'src' not in script_path else script_path

if args.model_file is None:
    raise FileNotFoundError("need to specify a saved file with `--model_file`")

print(args)

config, model, dataset, train_data, valid_data, test_data = utils.load_data_and_model(args.model_file,
                                                                                      args.explainer_config_file)

load_config_id = os.path.basename(os.path.dirname(args.explainer_config_file))

policy_map = {
    'force_removed_edges': 'MonDel',   # Monotonic Deletions
    'group_deletion_constraint': 'DelCons'  # Deletion Constraint
}

model_name = model.__class__.__name__
sens_attr, epochs, batch_exp = config['sensitive_attribute'], config['cf_epochs'], config['user_batch_exp']

attr_map = dataset.field2id_token[sens_attr]
f_idx, m_idx = (attr_map == 'F').nonzero()[0][0], (attr_map == 'M').nonzero()[0][0]
user_num, item_num = dataset.user_num, dataset.item_num
uid_field, iid_field = dataset.uid_field, dataset.iid_field
item_class = list(map(lambda x: [el for el in x if el != 0], train_data.dataset.item_feat['class'].numpy().tolist()))
evaluator = Evaluator(config)

utils.wandb_init(
    config,
    job_type="eval",
    name="Plots",
    group=f"{model_name}_{config['dataset']}_{sens_attr.title()}_epochs{config['cf_epochs']}_exp={load_config_id}",
    mode="disabled"
)
wandb.config.update({"exp": load_config_id, **config.final_config_dict})

metrics_names = evaluator.metrics
model_dp_s = f'{model_name}+FairDP'

exp_paths = {
    model_dp_s: os.path.join(script_path, 'dp_ndcg_explanations', dataset.dataset_name, model_name, 'FairDP',
                             sens_attr, f"epochs_{epochs}", load_config_id)
}

with open(os.path.join(exp_paths[model_dp_s], 'config.pkl'), 'rb') as f:
    exp_config = pickle.load(f)

edge_additions = exp_config['edge_additions']
exp_rec_data = exp_config['exp_rec_data']
delete_adv_group = exp_config['delete_adv_group']
rec_data = locals()[f"{exp_rec_data}_data"]
policy = '+'.join([pm for p, pm in policy_map.items() if exp_config['explainer_policies'][p]])

user_df = pd.DataFrame({
    'user_id': train_data.dataset.user_feat['user_id'].numpy(),
    sens_attr: train_data.dataset.user_feat[sens_attr].numpy()
})

train_df = pd.DataFrame(train_data.dataset.inter_feat.numpy())[["user_id", "item_id"]]

user_hist_matrix, _, user_hist_len = train_data.dataset.history_item_matrix()
item_hist_matrix, _, item_hist_len = train_data.dataset.history_user_matrix()

# %%
args.best_exp_col = args.best_exp_col[0] if len(args.best_exp_col) == 1 else args.best_exp_col
args.best_exp_col = {model_dp_s: 1}

# %%
best_test_pref_data, best_test_result = plot_utils.extract_best_metrics(
    exp_paths,
    args.best_exp_col,
    evaluator,
    test_data.dataset,
    config=config
)

if exp_rec_data != "test":
    best_rec_pref_data, best_rec_result = plot_utils.extract_best_metrics(
        exp_paths,
        args.best_exp_col,
        evaluator,
        rec_data.dataset,
        config=config
    )
else:
    best_rec_pref_data, best_rec_result = None, None

# %%
all_exp_test_dfs, test_result_all_data, test_n_users_data_all, test_topk_dist_all = plot_utils.extract_all_exp_metrics_data(
    exp_paths,
    train_data,
    test_data.dataset,
    evaluator,
    sens_attr,
    rec=False
)

if exp_rec_data != "test":
    all_exp_rec_dfs, rec_result_all_data, rec_n_users_data_all, rec_topk_dist_all = plot_utils.extract_all_exp_metrics_data(
        exp_paths,
        train_data,
        rec_data.dataset,
        evaluator,
        sens_attr,
        rec=True
    )
else:
    all_exp_rec_dfs, rec_result_all_data, rec_n_users_data_all, rec_topk_dist_all = [None] * 4

# Does not matter which explanation we take if we evaluate just the recommendations of the original model
exp_test_df = all_exp_test_dfs[model_dp_s][all_exp_test_dfs[model_dp_s]["epoch"] == all_exp_test_dfs[model_dp_s]["epoch"].unique()[0]]
if exp_rec_data != "test":
    exp_rec_df = all_exp_rec_dfs[model_dp_s][all_exp_rec_dfs[model_dp_s]["epoch"] == all_exp_rec_dfs[model_dp_s]["epoch"].unique()[0]]
else:
    exp_rec_df = None

test_orig_total_ndcg = utils.compute_metric(evaluator, test_data.dataset, exp_test_df, 'topk_pred', 'ndcg')
if exp_rec_data != "test":
    rec_orig_total_ndcg = utils.compute_metric(evaluator, rec_data.dataset, exp_rec_df, 'topk_pred', 'ndcg')
else:
    rec_orig_total_ndcg = None

test_orig_m_ndcg, test_orig_f_ndcg = utils.compute_metric_per_group(
    evaluator,
    test_data,
    user_df,
    all_exp_test_dfs[model_dp_s],
    sens_attr,
    (m_idx, f_idx),
    metric="ndcg"
)
if exp_rec_data != "test":
    rec_orig_m_ndcg, rec_orig_f_ndcg = utils.compute_metric_per_group(
        evaluator,
        rec_data,
        user_df,
        all_exp_rec_dfs[model_dp_s],
        sens_attr,
        (m_idx, f_idx),
        metric="ndcg"
    )

    value_orig_m_ndcg, value_orig_f_ndcg = rec_orig_m_ndcg, rec_orig_f_ndcg
else:
    rec_orig_m_ndcg, rec_orig_f_ndcg = None, None
    value_orig_m_ndcg, value_orig_f_ndcg = test_orig_m_ndcg, test_orig_f_ndcg

if value_orig_m_ndcg >= value_orig_f_ndcg:
    if delete_adv_group is not None:
        group_edge_del = m_idx if delete_adv_group else f_idx
    else:
        group_edge_del = m_idx
else:
    if delete_adv_group is not None:
        group_edge_del = f_idx if delete_adv_group else m_idx
    else:
        group_edge_del = f_idx

edges_ylabel = "% Del Edges" if not edge_additions else "% Added Edges"
title = "Edge Additions " if edge_additions else "Edge Deletions "
if sens_attr == "gender":
    title += "of Males " if group_edge_del == m_idx else "of Females "
    real_group_map = {'M': 'M', 'F': 'F'}
    m_label, f_label = "M", "F"
else:
    title += "of Younger " if group_edge_del == m_idx else "of Older "
    real_group_map = {'M': 'Y', 'F': 'O'}
    m_label, f_label = "Y", "O"
title += "Optimized on " + f"{exp_rec_data.title()} Data"

group_name_map = {
    "M": "Males",
    "F": "Females",
    "Y": "Younger",
    "O": "Older"
}

test_result_per_epoch_per_group, test_del_edges_per_epoch, test_fair_loss_per_epoch = plot_utils.result_data_per_epoch_per_group(
    all_exp_test_dfs,
    evaluator,
    (m_idx, f_idx),
    user_df,
    test_data.dataset,
    sens_attr
)
if exp_rec_data != "test":
    rec_result_per_epoch_per_group, rec_del_edges_per_epoch, rec_fair_loss_per_epoch = plot_utils.result_data_per_epoch_per_group(
        all_exp_rec_dfs,
        evaluator,
        (m_idx, f_idx),
        user_df,
        rec_data.dataset,
        sens_attr
    )
else:
    rec_result_per_epoch_per_group, rec_del_edges_per_epoch, rec_fair_loss_per_epoch = [None] * 3
#
# # %%
# plot_lineplot_per_epoch_per_group(
#     test_result_per_epoch_per_group,
#     test_del_edges_per_epoch,
#     test_fair_loss_per_epoch,
#     (test_orig_m_ndcg, test_orig_f_ndcg),
#     data_info="test"
# )
#
# if exp_rec_data != "test":
#     plot_lineplot_per_epoch_per_group(
#         rec_result_per_epoch_per_group,
#         rec_del_edges_per_epoch,
#         rec_fair_loss_per_epoch,
#         (rec_orig_m_ndcg, rec_orig_f_ndcg),
#         test_orig_ndcg=(test_orig_m_ndcg, test_orig_f_ndcg),
#         data_info=exp_rec_data
#     )
#
# # %%
# create_table_metrics_over_del_edges(
#     test_result_all_data,
#     all_exp_test_dfs,
#     best_test_result[model_name],
#     load_config_id,
#     n_bins=100,
#     hist_type="test",
#     test_f="f_oneway"
# )
#
# if exp_rec_data != "test":
#     create_table_metrics_over_del_edges(
#         rec_result_all_data,
#         all_exp_rec_dfs,
#         best_rec_result[model_name],
#         load_config_id,
#         n_bins=10,
#         hist_type=exp_rec_data,
#         test_f="f_oneway"
#     )
#
# if exp_rec_data != "test":
#     create_lineplot_metrics_over_del_edges(
#         rec_result_all_data,
#         all_exp_rec_dfs,
#         best_rec_result[model_name],
#         load_config_id,
#         n_bins=10,
#         hist_type=exp_rec_data,
#         test_f="f_oneway"
#     )
#
# create_lineplot_metrics_over_del_edges(
#     test_result_all_data,
#     all_exp_test_dfs,
#     best_test_result[model_name],
#     load_config_id,
#     n_bins=10,
#     hist_type="test",
#     test_f="f_oneway"
# )
#
#
# # %%
# if exp_rec_data != "test":
#     create_metric_access_over_del_edges_per_group(
#         rec_result_all_data,
#         all_exp_rec_dfs,
#         best_rec_result[model_name],
#         load_config_id,
#         hist_type=exp_rec_data,
#         zerometric=True
#     )
#
# create_metric_access_over_del_edges_per_group(
#     test_result_all_data,
#     all_exp_test_dfs,
#     best_test_result[model_name],
#     load_config_id,
#     hist_type="test",
#     zerometric=True
# )
#
# # %%
# if exp_rec_data != "test":
#     create_user_user_homophily_plot_over_del_edges_per_group(
#         all_exp_rec_dfs,
#         load_config_id,
#         hist_type=exp_rec_data
#     )
#
# create_user_user_homophily_plot_over_del_edges_per_group(
#     all_exp_test_dfs,
#     load_config_id,
#     hist_type="test"
# )
#
# # %%
# if exp_rec_data != "test":
#     create_item_item_homophily_plot_over_del_edges_per_popularity(
#         all_exp_rec_dfs,
#         load_config_id,
#         hist_type=exp_rec_data
#     )
#
# create_item_item_homophily_plot_over_del_edges_per_popularity(
#     all_exp_test_dfs,
#     load_config_id,
#     hist_type="test"
# )
#
# # %%
# if exp_rec_data != "test":
#     create_bias_ratio_categories_over_groups_plot_per_del_edges(
#         all_exp_rec_dfs,
#         load_config_id,
#         hist_type=exp_rec_data,
#         n_bins=6
#     )
#
# create_bias_ratio_categories_over_groups_plot_per_del_edges(
#     all_exp_test_dfs,
#     load_config_id,
#     hist_type="test",
#     n_bins=6
# )
#
# # %%
# plot_decomposition_perturbed(best_test_pref_data[model_dp_s], train_data)
#
# # %%
# if exp_rec_data != "test":
#     create_distribution_diff_metric_random_groups(
#         rec_result_all_data,
#         all_exp_rec_dfs,
#         best_rec_result[model_name],
#         load_config_id,
#         hist_type=exp_rec_data,
#         n_bins=12,
#         iterations=100
#     )
#
# create_distribution_diff_metric_random_groups(
#     test_result_all_data,
#     all_exp_test_dfs,
#     best_test_result[model_name],
#     load_config_id,
#     hist_type="test",
#     n_bins=12,
#     iterations=100
# )
