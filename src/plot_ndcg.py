# %%
import os
import pickle
import argparse
import inspect
import itertools
from collections import defaultdict

import tqdm
import pyvis
import torch
import scipy.stats
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mpl_tick
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import networkx as nx
from recbole.evaluator import Evaluator
from adjustText import adjust_text
from networkx.algorithms import bipartite

import src.utils as utils


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
def extract_best_metrics(_exp_paths, best_exp_col, data=None):
    data = test_data.dataset if data is None else data

    result_all = {}
    pref_data_all = {}

    for e_type, e_path in _exp_paths.items():
        if e_path is None:
            continue
        exps_data = utils.load_dp_exps_file(e_path)

        if isinstance(best_exp_col, dict):
            bec = best_exp_col[e_type]
        else:
            bec = best_exp_col

        if not isinstance(bec, list):
            bec = bec.lower() if isinstance(bec, str) else bec
        else:
            bec[0] = bec[0].lower()
        top_exp_func = None
        if isinstance(bec, int):
            def top_exp_func(exp): return exp[bec]
        elif bec not in ["first", "last", "mid"] and not isinstance(bec, list):
            top_exp_col = utils.EXPS_COLUMNS.index(bec) if bec is not None else None
            if top_exp_col is not None:
                def top_exp_func(exp): return sorted(exp, key=lambda x: x[top_exp_col])[0]
        elif bec == "first":
            def top_exp_func(exp): return exp[0]
        elif bec == "last":
            def top_exp_func(exp): return exp[-1]
        elif bec == "mid":
            def top_exp_func(exp): return exp[len(exp) // 2]
        elif isinstance(bec, list):
            top_exp_col = utils.EXPS_COLUMNS.index(bec[0])
            def top_exp_func(exp): return sorted(exp, key=lambda x: abs(x[top_exp_col] - bec[1]))[0]

        pref_data = []
        for exp_entry in exps_data:
            if top_exp_func is not None:
                _exp = top_exp_func(exp_entry)
            else:
                _exp = exp_entry[0]

            pref_data.extend(list(zip(_exp[0], _exp[1], _exp[3])))

        pref_data = pd.DataFrame(pref_data, columns=['user_id', 'topk_pred', 'cf_topk_pred'])
        pref_data_all[e_type] = pref_data

        if not pref_data.empty:
            result_all[e_type] = {}
            for metric in evaluator.metrics:
                result_all[e_type][metric] = utils.compute_metric(evaluator, data, pref_data, 'cf_topk_pred', metric)

                if model_name not in result_all:
                    result_all[model_name] = {}

                if metric not in result_all[model_name]:
                    result_all[model_name][metric] = utils.compute_metric(evaluator, data, pref_data, 'topk_pred', metric)
        else:
            print("Pref Data is empty!")

    return pref_data_all, result_all


def extract_all_exp_metrics_data(_exp_paths, data=None, rec=False):
    sensitive_map = train_data.dataset.field2id_token[sens_attr]

    data = test_data.dataset if data is None else data

    if not rec:
        cols = [2, 4, 6, 8, 9, 10, 11]
    else:
        cols = [1, 3, 5, 8, 9, 10, 11]

    col_names = [
        'user_id',
        'topk_pred',
        'cf_topk_pred',
        'topk_dist',
        'dist_loss',
        'fair_loss',
        'del_edges',
        'epoch'
    ]

    exp_dfs = {}
    result_data = {}
    n_users_data = {}
    topk_dist = {}
    for e_type, e_path in _exp_paths.items():
        if e_path is None:
            continue
        exps_data = utils.load_dp_exps_file(e_path)

        exp_data = []
        for exp_entry in exps_data:
            for _exp in exp_entry:
                exp_row_data = [_exp[0]]
                for col in cols:
                    if col in [1, 2, 3, 4, 5, 6]:
                        exp_row_data.append(_exp[col])
                    elif col == "set":
                        comm_items = np.array([len(set(orig) & set(pred)) for orig, pred in zip(_exp[1], _exp[2])])
                        exp_row_data.append(len(_exp[1][0]) - comm_items)
                    else:
                        exp_row_data.append([_exp[col]] * len(exp_row_data[0]))

                exp_data.extend(list(zip(*exp_row_data)))

        data_df = pd.DataFrame(exp_data, columns=col_names)
        data_df['n_del_edges'] = data_df['del_edges'].map(lambda x: x.shape[1])
        exp_dfs[e_type] = data_df

        if data_df.empty:
            print(f"User explanations are empty for {e_type}")
            continue

        result_data[e_type] = {}
        n_users_data[e_type] = {}
        topk_dist[e_type] = []
        for n_del, gr_df in tqdm.tqdm(data_df.groupby('n_del_edges'), desc="Extracting metrics from each explanation"):
            result_data[e_type][n_del] = {}
            for metric in evaluator.metrics:
                result_data[e_type][n_del][metric] = utils.compute_metric(evaluator, data, gr_df, 'cf_topk_pred', metric)

            t_dist = gr_df['topk_dist'].to_numpy()
            topk_dist[e_type].extend(list(
                zip([n_del] * len(t_dist), t_dist / len(t_dist), gr_df['topk_dist'].to_numpy() / len(t_dist))
            ))

            gr_df_attr = gr_df['user_id'].drop_duplicates().to_frame().join(user_df.set_index('user_id'), on='user_id')
            n_users_data[e_type][n_del] = {sens_attr: gr_df_attr[sens_attr].value_counts().to_dict()}
            n_users_del = n_users_data[e_type][n_del][sens_attr]
            n_users_data[e_type][n_del][sens_attr] = {sensitive_map[dg]: n_users_del[dg] for dg in n_users_del}

    return exp_dfs, result_data, n_users_data, topk_dist


def result_data_per_epoch_per_group(exp_dfs, data=None):
    data = test_data.dataset if data is None else data

    u_df = user_df.set_index('user_id')

    result_per_epoch = {}
    del_edges_per_epoch = {}
    fair_loss_per_epoch = {}
    for e_type, e_df in exp_dfs.items():
        result_per_epoch[e_type] = {}
        del_edges_per_epoch[e_type] = {}
        fair_loss_per_epoch[e_type] = {}
        for epoch, epoch_df in e_df.groupby("epoch"):
            result_per_epoch[e_type][epoch] = {}
            del_edges_per_epoch[e_type][epoch] = {}
            uid = epoch_df['user_id']

            m_mask = (u_df.loc[uid, sens_attr] == m_idx).values
            f_mask = ~m_mask
            m_df = epoch_df[m_mask]
            f_df = epoch_df[f_mask]

            result_per_epoch[e_type][epoch][m_idx], result_per_epoch[e_type][epoch][f_idx] = {}, {}
            for metric in evaluator.metrics:
                result_per_epoch[e_type][epoch][m_idx][metric] = utils.compute_metric(evaluator, data, m_df, 'cf_topk_pred', metric)[:, -1].mean()
                result_per_epoch[e_type][epoch][f_idx][metric] = utils.compute_metric(evaluator, data, f_df, 'cf_topk_pred', metric)[:, -1].mean()

            del_edges = epoch_df.iloc[0]['del_edges']
            del_edges_per_epoch[e_type][epoch][m_idx] = del_edges[:, (epoch_df.loc[m_mask].user_id.values[:, None] == del_edges[0]).nonzero()[1]]
            del_edges_per_epoch[e_type][epoch][f_idx] = del_edges[:, (epoch_df.loc[f_mask].user_id.values[:, None] == del_edges[0]).nonzero()[1]]

            fair_loss_per_epoch[e_type][epoch] = epoch_df.iloc[0]['fair_loss']

    return result_per_epoch, del_edges_per_epoch, fair_loss_per_epoch


# %%
def off_margin_ticks(*axs):
    # Turn off tick visibility for the measure axis on the marginal plots
    for ax in axs:
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_xticklabels(minor=True), visible=False)


# %%
def plot_lineplot_per_epoch_per_group(res_epoch_group,
                                      del_edges_epoch,
                                      fair_loss_per_epoch,
                                      orig_ndcg,
                                      test_orig_ndcg=None,
                                      data_info="test"):
    columns = ["Epoch", "Group", "metric", "value"]
    edges_ylabel = "# Del Edges" if not edge_additions else "# Added Edges"
    title = "Edge Additions " if edge_additions else "Edge Deletions "
    if sens_attr == "gender":
        title += "of Males " if group_edge_del == m_idx else "of Females "
        real_group_map = None
        m_label, f_label = "M", "F"
    else:
        title += "of Younger " if group_edge_del == m_idx else "of Older "
        real_group_map = {'M': 'Y', 'F': 'O'}
        m_label, f_label = "Y", "O"
    title += "Optimized on " + f"{exp_rec_data.title()} Data"

    df_test_result = None
    orig_m_ndcg, orig_f_ndcg = orig_ndcg
    for e_type in res_epoch_group:
        plots_path = os.path.join(get_plots_path(), 'comparison', f"epochs_{epochs}", '_'.join(cleaned_config_ids), sens_attr)
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)

        group_map = train_data.dataset.field2id_token[sens_attr]
        df_data = []
        del_edge_data = []
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
            df_test_result_data = []
            for epoch in test_result_per_epoch_per_group[e_type]:
                for gr, gr_res in test_result_per_epoch_per_group[e_type][epoch].items():
                    for metric in gr_res:
                        df_test_result_data.append([epoch, gr, metric, gr_res[metric]])

            df_test_result = pd.DataFrame(df_test_result_data, columns=columns)
            df_test_result["Group"] = df_test_result["Group"].map(group_map.__getitem__)

        df_del_data = pd.DataFrame(del_edge_data, columns=["Epoch", "Group", "del_edges"])
        df_del_data["Group"] = df_del_data["Group"].map(group_map.__getitem__)
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

                off_margin_ticks(ax_rec, ax_metric_diff, ax_del_edges)
            else:
                gs = plt.GridSpec(6, 6)

                ax_rec = fig.add_subplot(gs[2:, :])
                ax_test = None
                ax_metric_diff = fig.add_subplot(gs[1, :], sharex=ax_rec)
                ax_del_edges = fig.add_subplot(gs[0, :], sharex=ax_rec)

                off_margin_ticks(ax_metric_diff, ax_del_edges)

            df_diff = plot_df.loc[plot_df["Group"] == m_label, ["Epoch"]].copy()
            df_diff[metr_str] = np.abs(
                plot_df.loc[plot_df["Group"] == m_label, metr_str].values -
                plot_df.loc[plot_df["Group"] == f_label, metr_str].values
            )
            df_diff.rename(columns={metr_str: f"{metr_str} Diff"}, inplace=True)
            df_diff["Source Eval Data"] = rec_str

            if metric == "ndcg":
                df_fair_loss_data = []
                for epoch in fair_loss_per_epoch[e_type]:
                    df_fair_loss_data.append([epoch, fair_loss_per_epoch[e_type][epoch]])

                df_fair_loss_df = pd.DataFrame(df_fair_loss_data, columns=["Epoch", f"{metr_str} Diff"])

                df_fair_loss_df["Source Eval Data"] = "ApproxNDCG Loss"

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

                    title_proxy = Rectangle((0, 0), 0, 0, color='w')
                    ls_legend_handles = [
                        Line2D([0], [0], ls='-', color='k'),
                        Line2D([0], [0], ls='--', color='k')
                    ]

                    ls_legend_labels = [
                        data_type,
                        f"{data_type} Original"
                    ]

                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(
                        [title_proxy] + handles + [title_proxy] + ls_legend_handles,
                        ["Group"] + labels + ["Source Eval Data"] + ls_legend_labels
                    )

            diff_groups = df_diff.groupby("Source Eval Data")
            # diff_colors = sns.color_palette("colorblind")[::-1][:diff_groups.ngroups]
            diff_colors = sns.color_palette("colorblind6", n_colors=diff_groups.ngroups)

            sns.lineplot(x="Epoch", y=f"{metr_str} Diff", data=df_diff, palette=diff_colors, hue="Source Eval Data", ax=ax_metric_diff)
            # for i, (eval_type, eval_diff_df) in enumerate(diff_groups):
            #     zorder = 1 if eval_type == "ApproxNDCG Loss" else 2
            #     ax_metric_diff.fill_between(eval_diff_df["Epoch"], eval_diff_df[f"{metr_str} Diff"], color=diff_colors[i], alpha=0.3, zorder=zorder)
            ax_metric_diff.grid(axis='y', ls=':')

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
                x_lim = [plot_df["Epoch"].min(), plot_df["Epoch"].max()]
                ax_rec.plot(x_lim, [orig_m_ndcg, orig_m_ndcg], c=colors[0], ls='--')
                ax_rec.plot(x_lim, [orig_f_ndcg, orig_f_ndcg], c=colors[1], ls='--')

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
            fig.savefig(os.path.join(plots_path, f'{data_info}_lineplot_per_epoch_per_group_{e_type}_{metric}.png'))
            plt.close()


# %%
def create_lineplot_metrics_over_del_edges(_result_all_data, _pref_dfs, orig_result, config_ids, n_bins=10, hist_type="test", test_f="f_oneway"):
    sensitive_map = train_data.dataset.field2id_token[sens_attr]

    order = [model_name, f'{model_name}+DP']

    P_005 = '*'
    P_001 = '^'

    for metric in metrics_names:
        plots_path = os.path.join(get_plots_path(), 'comparison', f"epochs_{epochs}", '_'.join(config_ids), sens_attr)
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)

        # e_type with highest number of explanations
        max_del_edges_e_type = max([(k, len(x)) for k, x in _result_all_data.items()], key=lambda v: v[1])[0]
        del_edges = sorted(list(_result_all_data[max_del_edges_e_type]))

        bin_size = max(del_edges) // n_bins
        bin_map = {i: f"{i * bin_size + 1 if i != 0 else 1}-{(i + 1) * bin_size}" for i in
                   range(max(del_edges) // bin_size + 1)}

        del_edges_map = {x: bin_map[x // bin_size] for x in del_edges}

        d_grs = np.arange(1, len(sensitive_map))

        exp_data, stats_data, final_bins = compute_exp_stats_data(
            _result_all_data, _pref_dfs, orig_result, order, sens_attr, d_grs, del_edges_map, metric, test_f=test_f
        )

        # plot_vals = []
        # for row, stat in zip(exp_data, stats_data):
        #     plot_vals.append([])
        #     for j in range(len(final_bins)):
        #         if row:
        #             plot_vals[-1].append(f"{row[j]:.4f}"
        #                                  f"{P_001 if stat[j] < 0.01 else (P_005 if stat[j] < 0.05 else '')}")
        #         else:
        #             plot_vals[-1].append("-")

        final_bins = ['-'.join(map(lambda x: f"{int(x) / train_data.dataset.inter_num * 100:.2f}%", bin.split('-')))
                      for bin in final_bins]

        df_attr = pd.DataFrame(zip(
            exp_data.flatten(),
            np.repeat(order, len(final_bins)),
            np.tile(final_bins, len(order))
        ), columns=['DP', 'model', '% Del Edges'])

        sns.lineplot(x='% Del Edges', y='DP', hue='model', data=df_attr)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_path, f"{hist_type}_lineplot_over_edges_{sens_attr}_{metric}.png"))
        plt.close()


def compute_exp_stats_data(_result_all_data, _pref_dfs, orig_result, order, attr, d_grs, del_edges_map, metric, test_f="f_oneway"):
    orig_data = []
    orig_stats_data = []
    exp_data = []
    stats_data = []
    final_bins = None
    for e_type in order[1:]:
        exp_data.append([])
        stats_data.append([])
        if e_type in _result_all_data:
            result_data = _result_all_data[e_type]

            e_df = _pref_dfs[e_type]
            e_df_grby = e_df.groupby('n_del_edges')

            ch_bins = []
            temp_exp_data = []
            temp_stats_data = []
            for n_del, bin_del in del_edges_map.items():
                e_d_grs_df = e_df_grby.get_group(n_del).join(user_df.set_index("user_id"), on="user_id")
                masks = {d_gr: e_d_grs_df[attr] == d_gr for d_gr in d_grs}

                if len(ch_bins) == 0:
                    ch_bins.append(bin_del)
                elif bin_del not in ch_bins:  # nanmean over rows is executed only if new bin is met
                    exp_data[-1].append(np.nanmean(temp_exp_data))
                    stats_data[-1].append(np.nanmean(temp_stats_data))
                    temp_exp_data = []
                    temp_stats_data = []

                    ch_bins.append(bin_del)

                if n_del in result_data:
                    n_del_res_data = []
                    d_grs_exp_data = []
                    for d_gr in d_grs:
                        res_gr_data = result_data[n_del][metric][masks[d_gr], -1]
                        n_del_res_data.append(res_gr_data)
                        d_grs_exp_data.append(np.mean(res_gr_data))
                    try:
                        temp_stats_data.append(getattr(scipy.stats, test_f)(*n_del_res_data).pvalue)
                    except ValueError as e:
                        temp_stats_data.append(1)

                    new_d_grs_exp_data = []
                    comb_exp_data = list(itertools.combinations(d_grs_exp_data, 2))
                    for (g1, g2) in comb_exp_data:
                        new_d_grs_exp_data.append(abs(g1 - g2))
                    temp_exp_data.append(np.nansum(new_d_grs_exp_data) / len(comb_exp_data))
                else:
                    temp_exp_data.append(np.nan)

            final_bins = ch_bins
            exp_data[-1].append(np.nanmean(temp_exp_data))
            stats_data[-1].append(np.nanmean(temp_stats_data))

            if not orig_data and not orig_stats_data:
                temp_orig_data = []
                for d_gr in d_grs:
                    val = orig_result[metric][masks[d_gr], -1]
                    orig_stats_data.append(val)
                    temp_orig_data.append(np.nanmean(val))
                try:
                    orig_stats_data = [getattr(scipy.stats, test_f)(*orig_stats_data).pvalue] * len(final_bins)
                except ValueError as e:
                    orig_stats_data = [1] * len(final_bins)

                comb_orig_data = list(itertools.combinations(temp_orig_data, 2))
                for (g1, g2) in comb_orig_data:
                    orig_data.append(abs(g1 - g2))
                orig_data = [sum(orig_data) / len(comb_orig_data)] * len(final_bins)

    exp_data.insert(0, orig_data)
    stats_data.insert(0, orig_stats_data)

    return exp_data, stats_data, final_bins


# %%
def create_table_metrics_over_del_edges(_result_all_data, _pref_dfs, orig_result, config_ids, n_bins=10, hist_type="test", test_f="f_oneway"):
    sensitive_map = train_data.dataset.field2id_token[sens_attr]

    order = [model_name, f'{model_name}+DP']

    P_005 = '*'
    P_001 = '^'

    for metric in metrics_names:
        tables_path = os.path.join(get_plots_path(), 'comparison', f"epochs_{epochs}", '_'.join(config_ids), sens_attr)
        if not os.path.exists(tables_path):
            os.makedirs(tables_path)

        # e_type with highest number of explanations
        max_del_edges_e_type = max([(k, len(x)) for k, x in _result_all_data.items()], key=lambda v: v[1])[0]
        del_edges = sorted(list(_result_all_data[max_del_edges_e_type]))

        bin_size = max(del_edges) // n_bins
        bin_map = {i: f"{i * bin_size + 1 if i != 0 else 1}-{(i + 1) * bin_size}" for i in
                   range(max(del_edges) // bin_size + 1)}

        del_edges_map = {x: bin_map[x // bin_size] for x in del_edges}

        d_grs = np.arange(1, len(sensitive_map))

        exp_data, stats_data, final_bins = compute_exp_stats_data(
            _result_all_data, _pref_dfs, orig_result, order, sens_attr, d_grs, del_edges_map, metric, test_f=test_f
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

        df_attr.to_markdown(os.path.join(tables_path, f"{hist_type}_table_over_edges_{sens_attr}_{metric}_{test_f}.md"), tablefmt="github")
        df_attr.to_latex(os.path.join(tables_path, f"{hist_type}_table_over_edges_{sens_attr}_{metric}_{test_f}.tex"), multirow=True)


# %%
def plot_dist_over_del_edges(_topk_dist_all, bd_all, config_ids, max_del_edges=80):
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
            plots_path = os.path.join(get_plots_path(), 'comparison', f"epochs_{epochs}", '_'.join(config_ids), attr)
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
            plt.savefig(os.path.join(plots_path, f'edit_set_dist_over_del_edges_{e_type}.png'))
            plt.close()


# %%
def plot_del_edges_hops(dfs, _config_ids):
    # td_G = utils.get_nx_biadj_matrix(train_data.dataset)

    inter_matrix = train_data.dataset.inter_matrix(form='csr').astype(np.float32)

    td_G = nx.bipartite.from_biadjacency_matrix(inter_matrix)

    color_0 = "red"
    color_1 = "blue"

    for e_type, e_df in dfs.items():
        _df = e_df[['user_id', 'del_edges']]

        data_df = []
        for _, row in _df.iterrows():
            for edge in zip(*row['del_edges']):
                data_df.append([row['user_id'], edge])

        new_df = pd.DataFrame(data_df, columns=['user_id', 'edge'])
        new_df.drop_duplicates('edge', inplace=True)
        new_df[['node_1', 'node_2']] = list(map(list, new_df['edge'].values))

        new_df = new_df.join(user_df.set_index('user_id'), on='user_id')

        data_hops = []
        for (u_attr, u_id), u_df in new_df.groupby([sens_attr, 'user_id']):
            # if e_type == "GCMC+NDCG":
            #     print(u_attr, u_id)
            #     print(u_df)
            #     input()
            # continue
            u_G = nx.Graph()
            e = np.array(u_df['edge'].tolist())
            if not (e[:, 0] == u_id).all():
                fig = plt.figure(figsize=(14, 14))
                ego = set(nx.ego_graph(td_G, u_id, radius=1, center=True, undirected=True))
                sub = td_G.subgraph(ego | set(np.unique(np.append(e, u_id)))).copy()
                top_nodes = {n for n, d in sub.nodes(data=True) if d['bipartite'] == 0}
                node_color = list({n: 'green' if n == u_id else (color_0 if d['bipartite'] == 0 else color_1)
                                   for n, d in sub.nodes(data=True)}.values())

                tuple_e = list(map(tuple, e))
                sub_edges = sub.edges()
                edge_list = [edge for edge in sub_edges if u_id in edge] + tuple_e
                edge_color = ['red' if edge in tuple_e else 'black' for edge in edge_list]
                nx.draw(
                    sub,
                    nx.bipartite_layout(sub, top_nodes),
                    edgelist=edge_list,
                    node_color=node_color,
                    edge_color=edge_color
                )
                plt.show()
                plt.close("all")
                import pdb; pdb.set_trace()
            # u_G.add_edges_from(e)

            # max_hop = 0
            # while True:
            #     try:
            #         if len(nx.descendants_at_distance(u_G, u_id, max_hop + 1)) > 0:
            #             max_hop += 1
            #         else:
            #             break
            #     except Exception as e:
            #         import pdb; pdb.set_trace()
            #         break
            #
            # data_hops.append([u_id, max_hop])

        df_hops = pd.DataFrame(data_hops, columns=['user_id', 'max_hops'])

        print(df_hops)

        del new_df['edge']

        print(new_df)


def draw_graph3(networkx_graph, notebook=True, output_filename='graph.html', show_buttons=False, only_physics_buttons=False):
    """
    This function accepts a networkx graph object,
    converts it to a pyvis network object preserving its node and edge attributes,
    and both returns and saves a dynamic network visualization.

    Valid node attributes include:
        "size", "value", "title", "x", "y", "label", "color".

        (For more info: https://pyvis.readthedocs.io/en/latest/documentation.html#pyvis.network.Network.add_node)

    Valid edge attributes include:
        "arrowStrikethrough", "hidden", "physics", "title", "value", "width"

        (For more info: https://pyvis.readthedocs.io/en/latest/documentation.html#pyvis.network.Network.add_edge)


    Args:
        networkx_graph: The graph to convert and display
        notebook: Display in Jupyter?
        output_filename: Where to save the converted network
        show_buttons: Show buttons in saved version of network?
        only_physics_buttons: Show only buttons controlling physics of network?
    """

    # import
    from pyvis import network as net

    # make a pyvis network
    pyvis_graph = net.Network(notebook=notebook)
    pyvis_graph.width = '1000px'
    # for each node and its attributes in the networkx graph
    for node, node_attrs in networkx_graph.nodes(data=True):
        pyvis_graph.add_node(int(node), **node_attrs)
    #         print(node,node_attrs)

    # for each edge and its attributes in the networkx graph
    for source, target, edge_attrs in networkx_graph.edges(data=True):
        # if value/width not specified directly, and weight is specified, set 'value' to 'weight'
        if 'value' not in edge_attrs and 'width' not in edge_attrs and 'weight' in edge_attrs:
            # place at key 'value' the weight of the edge
            edge_attrs['value'] = float(edge_attrs['weight'])
            edge_attrs['weight'] = float(edge_attrs['weight'])

        # add the edge
        pyvis_graph.add_edge(int(source), int(target), **edge_attrs)

    # turn buttons on
    if show_buttons:
        if only_physics_buttons:
            pyvis_graph.show_buttons(filter_=['physics'])
        else:
            pyvis_graph.show_buttons()

    # return and also save
    return pyvis_graph.show(output_filename)


# %%
parser = argparse.ArgumentParser()
parser.add_argument('--model_file', required=True)
parser.add_argument('--explainer_config_file', default=os.path.join("config", "explainer.yaml"))
parser.add_argument('--load_config_ids', nargs="+", type=int, default=[1, 1],
                    help="follows the order ['Silvestri et al.', model_name + '+DP'], set -1 to skip")
parser.add_argument('--best_exp_col', nargs='+', default=["loss_total"])

args = parser.parse_args() # r"--model_file src\saved\GCMC-ML100K-Jun-01-2022_13-28-01.pth --explainer_config_file config\gcmc_explainer.yaml --load_config_ids -1 3".split())

script_path = os.path.abspath(os.path.dirname(inspect.getsourcefile(lambda: 0)))
script_path = os.path.join(script_path, 'src') if 'src' not in script_path else script_path

cleaned_config_ids = list(map(str, args.load_config_ids))

if args.model_file is None:
    raise FileNotFoundError("need to specify a saved file with `--model_file`")

print(args)

config, model, dataset, train_data, valid_data, test_data = utils.load_data_and_model(args.model_file,
                                                                                      args.explainer_config_file)

model_name = model.__class__.__name__
sens_attr, epochs, batch_exp = config['sensitive_attribute'], config['cf_epochs'], config['user_batch_exp']

attr_map = dataset.field2id_token[sens_attr]
f_idx, m_idx = (attr_map == 'F').nonzero()[0][0], (attr_map == 'M').nonzero()[0][0]
user_num, item_num = dataset.user_num, dataset.item_num
evaluator = Evaluator(config)

metrics_names = evaluator.metrics

# %%
# G=utils.get_nx_biadj_matrix(train_data.dataset)
# # top = nx.bipartite.sets(G)[0]
# top_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
# bottom_nodes = set(G) - top_nodes
#
# # color_G = nx.bipartite.color(G)
# nx.set_node_attributes(G, dict.fromkeys(top_nodes, "blue"), "color")
# nx.set_node_attributes(G, dict.fromkeys(bottom_nodes, "red"), "color")
# # nx.draw(G, nx.bipartite_layout(G, top_nodes), node_color=list({n: d['color'] for n, d in G.nodes(data=True)}.values()))
# draw_graph3(G, notebook=False)
# nt = pyvis.network.Network('500px', '500px')
# exit()

# %%

exp_paths = {}
for c_id, exp_t, old_exp_t in zip(
    args.load_config_ids,
    ['Silvestri et al.', f'{model_name}+DP'],
    ['pred_explain', 'FairDP']
):
    exp_paths[exp_t] = None
    if c_id != -1:
        exp_paths[exp_t] = os.path.join(script_path, 'dp_ndcg_explanations', dataset.dataset_name, model_name,
                                        old_exp_t, sens_attr, f"epochs_{epochs}", str(c_id))

with open(os.path.join(exp_paths[f'{model_name}+DP'], 'config.pkl'), 'rb') as f:
    exp_config = pickle.load(f)

edge_additions = exp_config['edge_additions']
exp_rec_data = exp_config['exp_rec_data']
delete_adv_group = exp_config['delete_adv_group']
rec_data = locals()[f"{exp_rec_data}_data"]

user_df = pd.DataFrame({
    'user_id': train_data.dataset.user_feat['user_id'].numpy(),
    sens_attr: train_data.dataset.user_feat[sens_attr].numpy()
})

train_df = pd.DataFrame(train_data.dataset.inter_feat.numpy())[["user_id", "item_id"]]

user_hist_matrix, _, user_hist_len = train_data.dataset.history_item_matrix()
item_hist_matrix, _, item_hist_len = train_data.dataset.history_user_matrix()

# %%
args.best_exp_col = args.best_exp_col[0] if len(args.best_exp_col) == 1 else args.best_exp_col
args.best_exp_col = {f'{model_name}+DP': 1}

# %%
best_test_pref_data, best_test_result = extract_best_metrics(exp_paths, args.best_exp_col, data=test_data.dataset)
best_rec_pref_data, best_rec_result = extract_best_metrics(exp_paths, args.best_exp_col, data=rec_data.dataset)

# %%
all_exp_test_dfs, test_result_all_data, test_n_users_data_all, test_topk_dist_all = extract_all_exp_metrics_data(
    exp_paths, data=test_data.dataset, rec=False
)
all_exp_rec_dfs, rec_result_all_data, rec_n_users_data_all, rec_topk_dist_all = extract_all_exp_metrics_data(
    exp_paths, data=rec_data.dataset, rec=True
)

test_orig_total_ndcg = utils.compute_metric(evaluator, test_data.dataset, all_exp_test_dfs[f'{model_name}+DP'], 'topk_pred', 'ndcg')
rec_orig_total_ndcg = utils.compute_metric(evaluator, rec_data.dataset, all_exp_rec_dfs[f'{model_name}+DP'], 'topk_pred', 'ndcg')

test_orig_m_ndcg = utils.compute_metric(
    evaluator,
    test_data.dataset,
    all_exp_test_dfs[f'{model_name}+DP'].set_index('user_id').loc[user_df.loc[user_df[sens_attr] == m_idx, 'user_id']].reset_index(),
    'topk_pred',
    'ndcg'
)[:, -1].mean()
test_orig_f_ndcg = utils.compute_metric(
    evaluator,
    test_data.dataset,
    all_exp_test_dfs[f'{model_name}+DP'].set_index('user_id').loc[user_df.loc[user_df[sens_attr] == f_idx, 'user_id']].reset_index(),
    'topk_pred',
    'ndcg'
)[:, -1].mean()

rec_orig_m_ndcg = utils.compute_metric(
    evaluator,
    rec_data.dataset,
    all_exp_rec_dfs[f'{model_name}+DP'].set_index('user_id').loc[user_df.loc[user_df[sens_attr] == m_idx, 'user_id']].reset_index(),
    'topk_pred',
    'ndcg'
)[:, -1].mean()
rec_orig_f_ndcg = utils.compute_metric(
    evaluator,
    rec_data.dataset,
    all_exp_rec_dfs[f'{model_name}+DP'].set_index('user_id').loc[user_df.loc[user_df[sens_attr] == f_idx, 'user_id']].reset_index(),
    'topk_pred',
    'ndcg'
)[:, -1].mean()

if rec_orig_m_ndcg >= rec_orig_f_ndcg:
    if delete_adv_group is not None:
        group_edge_del = m_idx if delete_adv_group else f_idx
    else:
        group_edge_del = m_idx
else:
    if delete_adv_group is not None:
        group_edge_del = f_idx if delete_adv_group else m_idx
    else:
        group_edge_del = f_idx

test_result_per_epoch_per_group, test_del_edges_per_epoch, test_fair_loss_per_epoch = result_data_per_epoch_per_group(all_exp_test_dfs, data=test_data.dataset)
rec_result_per_epoch_per_group, rec_del_edges_per_epoch, rec_fair_loss_per_epoch = result_data_per_epoch_per_group(all_exp_rec_dfs, data=rec_data.dataset)

# %%
plot_lineplot_per_epoch_per_group(
    test_result_per_epoch_per_group,
    test_del_edges_per_epoch,
    test_fair_loss_per_epoch,
    (test_orig_m_ndcg, test_orig_f_ndcg),
    data_info="test"
)
plot_lineplot_per_epoch_per_group(
    rec_result_per_epoch_per_group,
    rec_del_edges_per_epoch,
    rec_fair_loss_per_epoch,
    (rec_orig_m_ndcg, rec_orig_f_ndcg),
    test_orig_ndcg=(test_orig_m_ndcg, test_orig_f_ndcg),
    data_info=exp_rec_data
)

# %%
create_table_metrics_over_del_edges(
    test_result_all_data,
    all_exp_test_dfs,
    best_test_result[model_name],
    cleaned_config_ids,
    n_bins=100,
    hist_type="test",
    test_f="f_oneway"
)
create_table_metrics_over_del_edges(
    rec_result_all_data,
    all_exp_rec_dfs,
    best_rec_result[model_name],
    cleaned_config_ids,
    n_bins=10,
    hist_type=exp_rec_data,
    test_f="f_oneway"
)

# %%
# plot_del_edges_hops(all_exp_dfs, cleaned_config_ids)

