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
import matplotlib as mpl
import matplotlib.ticker as mpl_tick
import matplotlib.pyplot as plt
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
        config['dataset'],
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

            pref_data.extend(list(zip(_exp[0], _exp[1], _exp[2])))

        pref_data = pd.DataFrame(pref_data, columns=['user_id', 'topk_pred', 'cf_topk_pred'])
        pref_data_all[e_type] = pref_data

        if not pref_data.empty:
            result_all[e_type] = {}
            for metric in evaluator.metrics:
                result_all[e_type][metric] = utils.compute_metric(evaluator, data, pref_data, 'cf_topk_pred', metric)

                if 'GCMC' not in result_all:
                    result_all['GCMC'] = {}

                if metric not in result_all['GCMC']:
                    result_all['GCMC'][metric] = utils.compute_metric(evaluator, data, pref_data, 'topk_pred', metric)
        else:
            print("Pref Data is empty!")

    return pref_data_all, result_all


def extract_all_exp_metrics_data(_exp_paths, data=None):
    sens_attributes = config['sensitive_attributes']
    sensitive_maps = {sens_attr: train_data.dataset.field2id_token[sens_attr] for sens_attr in sens_attributes}

    data = test_data.dataset if data is None else data

    cols = [1, 2, 6, 3, "set", 8, 10]
    col_names = ['user_id', 'topk_pred', 'cf_topk_pred', 'dist_loss', 'topk_dist', 'topk_set_dist', 'del_edges', 'epoch']

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
                    if col in [1, 2]:
                        exp_row_data.append(_exp[col])
                    elif col in [6]:
                        if _exp[col] == int(_exp[col]):
                            exp_row_data.append([int(_exp[col])] * len(exp_row_data[0]))
                        else:
                            exp_row_data.append([int(_exp[8].shape[1])] * len(exp_row_data[0]))
                    elif col in [3]:
                        if len(_exp[col]) == len(exp_row_data[0]):
                            exp_row_data.append(_exp[col])
                        else:
                            exp_row_data.append(
                                [utils.damerau_levenshtein_distance(_pred, _topk_idx)
                                 for _pred, _topk_idx in zip(_exp[1], _exp[2])]
                            )
                    elif col == "set":
                        comm_items = np.array([len(set(orig) & set(pred)) for orig, pred in zip(_exp[1], _exp[2])])
                        exp_row_data.append(len(_exp[1][0]) - comm_items)
                    else:
                        exp_row_data.append([_exp[col]] * len(exp_row_data[0]))

                exp_data.extend(list(zip(*exp_row_data)))

        data_df = pd.DataFrame(exp_data, columns=col_names)
        data_df['n_del_edges'] = data_df['del_edges'].map(lambda x: x.shape[0])
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
                zip([n_del] * len(t_dist), t_dist / len(t_dist), gr_df['topk_set_dist'].to_numpy() / len(t_dist))
            ))

            gr_df_attr = gr_df['user_id'].drop_duplicates().to_frame().join(user_df.set_index('user_id'), on='user_id')
            n_users_data[e_type][n_del] = {attr: gr_df_attr[attr].value_counts().to_dict() for attr in sens_attributes}
            for attr in sens_attributes:
                n_users_del = n_users_data[e_type][n_del][attr]
                n_users_data[e_type][n_del][attr] = {sensitive_maps[attr][dg]: n_users_del[dg] for dg in n_users_del}

    return exp_dfs, result_data, n_users_data, topk_dist


def result_data_per_epoch_per_group(exp_dfs, data=None):
    data = test_data.dataset if data is None else data

    u_df = user_df.set_index('user_id')

    result_per_epoch = {}
    del_edges_per_epoch = {}
    for e_type, e_df in exp_dfs.items():
        result_per_epoch[e_type] = {}
        del_edges_per_epoch[e_type] = {}
        for epoch, epoch_df in e_df.groupby("epoch"):
            result_per_epoch[e_type][epoch] = {}
            del_edges_per_epoch[e_type][epoch] = {}
            for attr in sens_attrs:
                if attr == "gender":
                    uid = epoch_df['user_id']

                    males_mask = (u_df.loc[uid, attr] == male_idx).values
                    females_mask = ~males_mask
                    males_df = epoch_df[males_mask]
                    females_df = epoch_df[females_mask]

                    result_per_epoch[e_type][epoch][1], result_per_epoch[e_type][epoch][2] = {}, {}
                    for metric in evaluator.metrics:
                        result_per_epoch[e_type][epoch][male_idx][metric] = utils.compute_metric(evaluator, data, males_df, 'cf_topk_pred', metric)[:, -1].mean()
                        result_per_epoch[e_type][epoch][female_idx][metric] = utils.compute_metric(evaluator, data, females_df, 'cf_topk_pred', metric)[:, -1].mean()

                    del_edges = epoch_df.iloc[0]['del_edges']
                    del_edges_per_epoch[e_type][epoch][male_idx] = del_edges[:, (epoch_df.loc[males_mask].user_id.values[:, None] == del_edges[0]).nonzero()[1]]
                    del_edges_per_epoch[e_type][epoch][female_idx] = del_edges[:, (epoch_df.loc[females_mask].user_id.values[:, None] == del_edges[0]).nonzero()[1]]

    return result_per_epoch, del_edges_per_epoch


# %%
def off_margin_ticks(ax_marg_x1, ax_marg_x2):
    # Turn off tick visibility for the measure axis on the marginal plots
    plt.setp(ax_marg_x1.get_xticklabels(), visible=False)
    plt.setp(ax_marg_x2.get_xticklabels(), visible=False)
    plt.setp(ax_marg_x1.get_xticklabels(minor=True), visible=False)
    plt.setp(ax_marg_x2.get_xticklabels(minor=True), visible=False)


# %%
def plot_lineplot_per_epoch_per_group(res_epoch_group, del_edges_epoch, annot_offset=0.005):
    columns = ["Epoch", "Group", "metric", "value"]

    for e_type in res_epoch_group:
        for attr in sens_attrs:
            plots_path = os.path.join(get_plots_path(), 'comparison', '_'.join(cleaned_config_ids), attr)
            if not os.path.exists(plots_path):
                os.makedirs(plots_path)

            group_map = train_data.dataset.field2id_token[attr]
            df_data = []
            del_edge_data = []
            for epoch in res_epoch_group[e_type]:
                for gr, gr_res in res_epoch_group[e_type][epoch].items():
                    for metric in gr_res:
                        df_data.append([epoch, gr, metric, gr_res[metric]])
                    del_edge_data.append([epoch, gr, del_edges_epoch[e_type][epoch][gr]])

            df = pd.DataFrame(df_data, columns=columns)
            df["Group"] = df["Group"].map(group_map.__getitem__)

            df_del_data = pd.DataFrame(del_edge_data, columns=["Epoch", "Group", "del_edges"])
            df_del_data["Group"] = df_del_data["Group"].map(group_map.__getitem__)
            df_del_data["# Del Edges"] = df_del_data["del_edges"].map(lambda x: x.shape[1])
            df_del_data["# Del Edges Lab"] = (df_del_data["# Del Edges"] / train_data.dataset.inter_num * 100).map("{:.2f}%".format)
            df_del_data.sort_values("# Del Edges", inplace=True)

            for metric in evaluator.metrics:
                metr_str = metric.upper()

                fig = plt.figure(figsize=(10, 10))
                plot_df = df[df["metric"] == metric].copy()
                plot_df.rename(columns={"value": metr_str}, inplace=True)

                male_val = plot_df.loc[plot_df['Group'] == "M"].sort_values("Epoch")[metr_str].to_numpy()
                female_val = plot_df.loc[plot_df['Group'] == "F"].sort_values("Epoch")[metr_str].to_numpy()
                intersects = np.argwhere(np.diff(np.sign(male_val - female_val))).flatten()
                x_intersects = plot_df.loc[plot_df['Group'] == "M"].sort_values("Epoch")["Epoch"].iloc[intersects].to_numpy()

                colors = sns.color_palette("colorblind", n_colors=2)
                gs = plt.GridSpec(6, 6)

                ax = fig.add_subplot(gs[2:, :])
                ax_metric_diff = fig.add_subplot(gs[1, :], sharex=ax)
                ax_del_edges = fig.add_subplot(gs[0, :], sharex=ax)

                off_margin_ticks(ax_metric_diff, ax_del_edges)

                df_diff = plot_df.loc[plot_df["Group"] == "M", ["Epoch", "Group"]].copy()
                df_diff[metr_str] = np.abs(
                    plot_df.loc[plot_df["Group"] == "M", metr_str].values -
                    plot_df.loc[plot_df["Group"] == "F", metr_str].values
                )
                df_diff.rename(columns={metr_str: f"{metr_str} Diff"}, inplace=True)

                lines = sns.lineplot(x="Epoch", y=metr_str, data=plot_df, hue="Group", palette=colors, hue_order=["M", "F"], ax=ax)
                sns.lineplot(x="Epoch", y=f"{metr_str} Diff", data=df_diff, color="blue", ax=ax_metric_diff)
                ax_metric_diff.fill_between(df_diff["Epoch"], df_diff[f"{metr_str} Diff"], color="blue", alpha=0.3)
                sns.lineplot(x="Epoch", y="# Del Edges", hue="Group", data=df_del_data, palette=colors, hue_order=["M", "F"], ax=ax_del_edges)

                df_del_data_epoch_group = df_del_data.set_index(["Epoch", "Group"])
                max_epoch = df_del_data["Epoch"].max()
                ax_del_edges.annotate(
                    df_del_data_epoch_group.loc[(max_epoch, "F"), "# Del Edges Lab"],
                    (max_epoch, df_del_data_epoch_group.loc[(max_epoch, "F"), "# Del Edges"])
                )

                ax_del_edges.yaxis.set_major_formatter(mpl_tick.FuncFormatter(lambda x, pos: f"{x / train_data.dataset.inter_num * 100:.2f}%"))

                y_scatter = []
                texts = []
                for i, (x_cross, cross) in enumerate(zip(x_intersects, intersects)):
                    mean_metric = (male_val[cross] + female_val[cross]) / 2
                    diff_metric = abs(male_val[cross] - female_val[cross])
                    texts.append(ax.text(x_cross, mean_metric, f"{mean_metric:.4f} ({diff_metric:.4f})"))
                    y_scatter.append(mean_metric)

                if len(intersects) > 0:
                    ax.scatter(x_intersects, y_scatter, marker="X", c="k")

                if metric == "ndcg":
                    x_lim = [plot_df["Epoch"].min(), plot_df["Epoch"].max()]
                    ax.plot(x_lim, [orig_males_ndcg, orig_males_ndcg], c=colors[0], ls='--')
                    ax.plot(x_lim, [orig_females_ndcg, orig_females_ndcg], c=colors[1], ls='--')

                    global_idx_inters = (
                            plot_df.loc[plot_df["Group"] == "M"].sort_values("Epoch")[metr_str] - orig_females_ndcg
                    ).abs().sort_values().index[0]
                    inters_val, y_global_idx_inters = plot_df.loc[global_idx_inters, [metr_str, "Epoch"]]
                    ax.annotate(
                        f"{orig_females_ndcg:.4f} ({abs(orig_females_ndcg - inters_val):.4f})",
                        (y_global_idx_inters + y_global_idx_inters * annot_offset, orig_females_ndcg + orig_females_ndcg * annot_offset)
                    )
                    ax.scatter([y_global_idx_inters], [orig_females_ndcg], c="k")

                ax.minorticks_on()
                ax.grid(True, which="both", ls=':')

                adjust_text(
                    texts,
                    only_move={'points': 'xy', 'text': 'xy'},
                    arrowprops=dict(arrowstyle="->", color='k', lw=0.5),
                    ax=ax,
                    expand_text=(1.1, 1.2),
                    expand_points=(1.1, 2.5),
                    expand_objects=(0.3, 0.2),
                    add_objects=lines.lines[:2]
                )

                plt.tight_layout()
                fig.savefig(os.path.join(plots_path, f'lineplot_per_epoch_per_group_{e_type}_{metric}.png'))
                plt.close()


# %%
def create_lineplot_metrics_over_del_edges(_result_all_data, _pref_dfs, orig_result, config_ids, n_bins=10, hist_type="test", test_f="f_oneway"):
    sens_attributes = config['sensitive_attributes']
    sensitive_maps = {sens_attr: train_data.dataset.field2id_token[sens_attr] for sens_attr in sens_attributes}

    order = ['GCMC', 'GCMC+DP']

    P_005 = '*'
    P_001 = '^'

    for metric in metrics_names:
        for attr in sens_attributes:
            sens_map = sensitive_maps[attr]
            plots_path = os.path.join(get_plots_path(), 'comparison', '_'.join(config_ids), attr)
            if not os.path.exists(plots_path):
                os.makedirs(plots_path)

            # e_type with highest number of explanations
            max_del_edges_e_type = max([(k, len(x)) for k, x in _result_all_data.items()], key=lambda v: v[1])[0]
            del_edges = sorted(list(_result_all_data[max_del_edges_e_type]))

            bin_size = max(del_edges) // n_bins
            bin_map = {i: f"{i * bin_size + 1 if i != 0 else 1}-{(i + 1) * bin_size}" for i in
                       range(max(del_edges) // bin_size + 1)}

            del_edges_map = {x: bin_map[x // bin_size] for x in del_edges}

            d_grs = np.arange(1, len(sens_map))

            exp_data, stats_data, final_bins = compute_exp_stats_data(
                _result_all_data, _pref_dfs, orig_result, order, attr, d_grs, del_edges_map, metric, test_f=test_f
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
            plt.savefig(os.path.join(plots_path, f"{hist_type}_lineplot_over_edges_{attr}_{metric}.png"))
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
    sens_attributes = config['sensitive_attributes']
    sensitive_maps = {sens_attr: train_data.dataset.field2id_token[sens_attr] for sens_attr in sens_attributes}

    order = ['GCMC', 'GCMC+DP']

    P_005 = '*'
    P_001 = '^'

    for metric in metrics_names:
        for attr in sens_attributes:
            sens_map = sensitive_maps[attr]
            tables_path = os.path.join(get_plots_path(), 'comparison', '_'.join(config_ids), attr)
            if not os.path.exists(tables_path):
                os.makedirs(tables_path)

            # e_type with highest number of explanations
            max_del_edges_e_type = max([(k, len(x)) for k, x in _result_all_data.items()], key=lambda v: v[1])[0]
            del_edges = sorted(list(_result_all_data[max_del_edges_e_type]))

            bin_size = max(del_edges) // n_bins
            bin_map = {i: f"{i * bin_size + 1 if i != 0 else 1}-{(i + 1) * bin_size}" for i in
                       range(max(del_edges) // bin_size + 1)}

            del_edges_map = {x: bin_map[x // bin_size] for x in del_edges}

            d_grs = np.arange(1, len(sens_map))

            exp_data, stats_data, final_bins = compute_exp_stats_data(
                _result_all_data, _pref_dfs, orig_result, order, attr, d_grs, del_edges_map, metric, test_f=test_f
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

            df_attr.to_markdown(os.path.join(tables_path, f"{hist_type}_table_over_edges_{attr}_{metric}_{test_f}.md"), tablefmt="github")
            df_attr.to_latex(os.path.join(tables_path, f"{hist_type}_table_over_edges_{attr}_{metric}_{test_f}.tex"), multirow=True)


# %%
def plot_dist_over_del_edges(_topk_dist_all, bd_all, config_ids, max_del_edges=80):
    for e_type, _topk_dist in _topk_dist_all.items():
        bd = bd_all[e_type]
        bd_attrs = defaultdict(list)
        for n_del in bd:
            for attr in bd[n_del]:
                for d_gr, bd_d_gr in bd[n_del][attr].items():
                    if bd_d_gr is not None:
                        bd_attrs[attr].extend(list(zip([n_del] * len(bd_d_gr), bd_d_gr.numpy(), [d_gr] * len(bd_d_gr))))

        bd_attrs = dict(bd_attrs)

        topk_dist_df = pd.DataFrame(_topk_dist, columns=['# Del Edges', 'Edit Dist', 'Set Dist'])

        for attr, bd_attr_data in bd_attrs.items():
            plots_path = os.path.join(get_plots_path(), 'comparison', '_'.join(config_ids), attr)
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
    sens_attrs = config['sensitive_attributes']
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

        for attr in sens_attrs:
            data_hops = []
            for (u_attr, u_id), u_df in new_df.groupby([attr, 'user_id']):
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
parser.add_argument('--explainer_config_file', default=os.path.join("config", "gcmc_explainer.yaml"))
parser.add_argument('--load_config_ids', nargs="+", type=int, default=[1, 1],
                    help="follows the order ['Silvestri et al.', 'GCMC+DP'], set -1 to skip")
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

gender_map = dataset.field2id_token['gender']
female_idx, male_idx = (gender_map == 'F').nonzero()[0][0], (gender_map == 'M').nonzero()[0][0]
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
sens_attrs, epochs, batch_exp = config['sensitive_attributes'], config['cf_epochs'], config['user_batch_exp']

exp_paths = {}
for c_id, exp_t, old_exp_t in zip(
    args.load_config_ids,
    ['Silvestri et al.', 'GCMC+DP'],
    ['pred_explain', 'FairDP']
):
    exp_paths[exp_t] = None
    if c_id != -1:
        exp_paths[exp_t] = os.path.join(script_path, 'dp_ndcg_explanations', dataset.dataset_name,
                                        old_exp_t, '_'.join(sens_attrs), f"epochs_{epochs}", str(c_id))

user_df = pd.DataFrame({
    'user_id': train_data.dataset.user_feat['user_id'].numpy(),
    **{sens_attr: train_data.dataset.user_feat[sens_attr].numpy() for sens_attr in sens_attrs}
})

train_df = pd.DataFrame(train_data.dataset.inter_feat.numpy())[["user_id", "item_id"]]

user_hist_matrix, _, user_hist_len = train_data.dataset.history_item_matrix()
item_hist_matrix, _, item_hist_len = train_data.dataset.history_user_matrix()

# %%
args.best_exp_col = args.best_exp_col[0] if len(args.best_exp_col) == 1 else args.best_exp_col
args.best_exp_col = {"GCMC+DP": 20}

# %%
best_test_pref_data, best_test_result = extract_best_metrics(exp_paths, args.best_exp_col, data=test_data.dataset)
best_val_pref_data, best_val_result = extract_best_metrics(exp_paths, args.best_exp_col, data=valid_data.dataset)

# %%
all_exp_test_dfs, test_result_all_data, test_n_users_data_all, test_topk_dist_all = extract_all_exp_metrics_data(
    exp_paths, data=test_data.dataset
)
all_exp_val_dfs, val_result_all_data, val_n_users_data_all, val_topk_dist_all = extract_all_exp_metrics_data(
    exp_paths, data=valid_data.dataset
)

orig_total_ndcg = utils.compute_metric(evaluator, test_data.dataset, all_exp_test_dfs["GCMC+DP"], 'topk_pred', 'ndcg')

orig_males_ndcg = utils.compute_metric(
    evaluator,
    test_data.dataset,
    all_exp_test_dfs["GCMC+DP"].set_index('user_id').loc[user_df.loc[user_df['gender'] == male_idx, 'user_id']].reset_index(),
    'topk_pred',
    'ndcg'
)[:, -1].mean()
orig_females_ndcg = utils.compute_metric(
    evaluator,
    test_data.dataset,
    all_exp_test_dfs["GCMC+DP"].set_index('user_id').loc[user_df.loc[user_df['gender'] == female_idx, 'user_id']].reset_index(),
    'topk_pred',
    'ndcg'
)[:, -1].mean()

result_per_epoch_per_group, del_edges_per_epoch = result_data_per_epoch_per_group(all_exp_test_dfs, data=test_data.dataset)

# %%
plot_lineplot_per_epoch_per_group(result_per_epoch_per_group, del_edges_per_epoch)

# %%
create_table_metrics_over_del_edges(
    test_result_all_data,
    all_exp_test_dfs,
    best_test_result['GCMC'],
    cleaned_config_ids,
    n_bins=100,
    hist_type="test",
    test_f="f_oneway"
)
create_table_metrics_over_del_edges(
    val_result_all_data,
    all_exp_test_dfs,
    best_val_result['GCMC'],
    cleaned_config_ids,
    n_bins=10,
    hist_type="val",
    test_f="f_oneway"
)

# %%
# plot_del_edges_hops(all_exp_dfs, cleaned_config_ids)

