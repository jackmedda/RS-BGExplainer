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
import recbole.evaluator.collector as recb_collector
from recbole.evaluator import Evaluator
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


def plot_bias_analysis_disparity(train_bias, rec_bias, _train_data, item_categories=None):
    plots_path = get_plots_path()

    bias_disparity = dict.fromkeys(list(train_bias.keys()))
    for attr in train_bias:
        group_map = _train_data.dataset.field2id_token[attr]
        bias_disparity[attr] = dict.fromkeys(list(train_bias[attr].keys()))
        for demo_group in train_bias[attr]:
            if train_bias[attr][demo_group] is None or rec_bias[attr][demo_group] is None:
                bias_disparity[attr][group_map[demo_group]] = None
                continue

            bias_r = rec_bias[attr][demo_group]
            bias_s = train_bias[attr][demo_group]
            bias_disparity[attr][group_map[demo_group]] = (bias_r - bias_s) / bias_s

        # fig, axs = plt.subplots(len(bias_disparity[attr]) - 1, 1, figsize=(10, 8))
        df = pd.DataFrame(bias_disparity[attr])[1:].dropna(axis=1).T
        if item_categories is not None:
            item_categories_map = dict(zip(range(len(item_categories)), item_categories))
            df.rename(columns=item_categories_map, inplace=True)
        ax = sns.heatmap(df, vmin=-2.0, vmax=2.0)
        ax.set_title(attr.title())
        plt.tight_layout()
        plt.savefig(os.path.join(plots_path, f'{attr}.png'))
        plt.close()

    return plots_path


def clean_history_matrix(hist_m):
    for col in ['topk_pred', 'cf_topk_pred']:
        if isinstance(hist_m.iloc[0][col], str):
            hist_m[col] = hist_m[col].map(lambda x: np.array(x[1:-1].strip().split(), int))


def compute_result(pref_data, pred_col, metric, hist_matrix):
    dataobject = recb_collector.DataStruct()
    pos_matrix = np.zeros((user_num, item_num), dtype=int)
    pos_matrix[pref_data['user_id'].to_numpy()[:, None], hist_matrix[pref_data['user_id'], :]] = 1
    pos_matrix[:, 0] = 0
    pos_len_list = torch.tensor(pos_matrix.sum(axis=1, keepdims=True)[1:])
    pos_idx = torch.tensor(pos_matrix[pref_data['user_id'].to_numpy()[:, None], np.stack(pref_data[pred_col].values)])
    pos_data = torch.cat((pos_idx, pos_len_list), dim=1)
    dataobject.set('rec.topk', pos_data)

    pos_index, pos_len = evaluator.metric_class[metric].used_info(dataobject)
    result = evaluator.metric_class[metric].metric_info(pos_index, pos_len)

    return result


def extract_best_metrics(_exp_paths, best_exp_col, hist_matrix=None):
    hist_matrix = test_hist_matrix if hist_matrix is None else hist_matrix

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
                result_all[e_type][metric] = compute_result(pref_data, 'cf_topk_pred', metric, hist_matrix)

                if 'GCMC' not in result_all:
                    result_all['GCMC'] = {}

                if metric not in result_all['GCMC']:
                    result_all['GCMC'][metric] = compute_result(pref_data, 'topk_pred', metric, hist_matrix)
        else:
            print("Pref Data is empty!")

    return pref_data_all, result_all


def plot_bias_disparity_diff_dumbbell(bd, sens_attrs, config_ids, sort="dots"):
    plots_path = os.path.join(get_plots_path(), 'comparison', '_'.join(config_ids))
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    if sort not in ["dots", "barplot_side"]:
        raise NotImplementedError(f"sort = {sort} not supported for dumbbell")

    x, y = 'Bias Disparity', 'Category'
    item_cats = train_data.dataset.field2id_token['class']

    sensitive_maps = {sens_attr: train_data.dataset.field2id_token[sens_attr] for sens_attr in sens_attrs}

    for attr in sens_attrs:
        attr_path = os.path.join(plots_path, attr)
        if not os.path.exists(attr_path):
            os.makedirs(attr_path)

        cats_inter_counts_attr = {}
        for demo_gr, demo_df in joint_df.groupby(attr):
            class_counts = demo_df.explode('class').value_counts('class')
            class_counts.index = class_counts.index.map(item_cats.__getitem__)
            cats_inter_counts_attr[sensitive_maps[attr][demo_gr]] = class_counts.to_dict()

        for demo_gr in bd['GCMC'][attr]:
            if bd['GCMC'][attr][demo_gr] is not None:
                for exp_type in bd:
                    if exp_type == 'GCMC' or bd[exp_type][attr][demo_gr] is None:
                        continue

                    orig_data = dict(zip(item_cats, bd['GCMC'][attr][demo_gr].numpy()))
                    exp_data = dict(zip(item_cats, bd[exp_type][attr][demo_gr].numpy()))

                    df_orig = pd.DataFrame.from_dict(orig_data, orient='index', columns=[x]).reset_index().dropna()
                    df_orig = df_orig.rename(columns={'index': y})
                    df_exp = pd.DataFrame.from_dict(exp_data, orient='index', columns=[x]).reset_index().dropna()
                    df_exp = df_exp.rename(columns={'index': y})

                    if sort == "dots":
                        order = df_orig.sort_values(x)[y].to_list()
                        bar_data = pd.DataFrame(cats_inter_counts_attr[demo_gr].items(), columns=[y, x])
                    elif sort == "barplot_side":
                        order, vals = map(list, zip(*(sorted(cats_inter_counts_attr[demo_gr].items(), key=lambda x: x[1])[::-1])))
                        bar_data = pd.DataFrame(zip(order, vals), columns=[y, x])

                    g = sns.JointGrid(height=12, space=0.5)
                    g.ax_marg_x.remove()
                    sns.barplot(x=x, y=y, data=bar_data, ax=g.ax_marg_y, color="black", order=order)

                    sns.stripplot(x=x, y=y, color='#F5793A', data=df_exp, ax=g.ax_joint, jitter=False, s=10, label=exp_type, zorder=2, order=order)
                    sns.stripplot(x=x, y=y, color='#A95AA1', data=df_orig, ax=g.ax_joint, jitter=False, s=10, label='GCMC', zorder=2, order=order)

                    lines_df = df_orig.set_index('Category').join(df_exp.set_index('Category'), lsuffix='_orig').loc[order]
                    lines_styles = ((lines_df[x + '_orig'].abs() - lines_df[x].abs()) < 0).map(lambda x: ':' if x else '-').values.tolist()

                    lines_df['diff%'] = (((lines_df[x + '_orig'].abs() - lines_df[x].abs()) / lines_df[x + '_orig'].abs()) * 100).round(1)
                    lines_df['abs_diff'] = (lines_df[x + '_orig'] - lines_df[x]).abs().round(1)
                    lines_df['diff%'] = lines_df[['diff%', 'abs_diff']].apply(lambda row: f"{row['diff%']}% ({row['abs_diff']})", axis=1)
                    del lines_df['abs_diff']
                    for i, c in enumerate(lines_df.index):
                        g.ax_joint.plot(lines_df.loc[c, [x + '_orig', x]], [c, c], 'k', zorder=1, ls=lines_styles[i], label="")
                        g.ax_joint.text(lines_df.loc[c, [x + '_orig', x]].mean(), i - 0.3, lines_df.loc[c, 'diff%'], ha='center')

                    utils.legend_without_duplicate_labels(g.ax_joint)

                    g.ax_joint.plot([0., 0.], g.ax_joint.get_ylim(), 'k--', zorder=1)
                    g.ax_joint.set_title(f"{attr.title()}: {demo_gr}")

                    plt.tight_layout()
                    plt.savefig(os.path.join(attr_path, f"{demo_gr}#dumbbell_orig__{exp_type}.png"))
                    plt.close()


def plot_bias_disparity_boxplot(bd, pref_topk_all, sens_attrs, config_ids):
    plots_path = os.path.join(get_plots_path(), 'comparison', '_'.join(config_ids))
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    order = ['GCMC', 'GCMC+BD', 'GCMC+NDCG']

    for attr in sens_attrs:
        attr_path = os.path.join(plots_path, attr)
        if not os.path.exists(attr_path):
            os.makedirs(attr_path)

        rows = int(np.ceil((len(bd['GCMC'][attr]) - 1) / 3))
        cols = min(3, len(bd['GCMC'][attr]) - 1)

        fig, axs = plt.subplots(rows, cols, sharey=True, figsize=(12, 12))
        axs = axs.ravel()
        i = 0
        for demo_gr in bd['GCMC'][attr]:
            if bd['GCMC'][attr][demo_gr] is not None:
                plot_data = [[], []]
                exp_size = {}
                for exp_type in bd:
                    if bd[exp_type][attr][demo_gr] is None:
                        continue

                    exp_data = bd[exp_type][attr][demo_gr].numpy().tolist()
                    plot_data[0].extend(exp_data)
                    plot_data[1].extend([exp_type] * len(exp_data))

                    exp_size[exp_type] = f"{len(pref_topk_all[exp_type]) / (train_data.dataset.user_num - 1) * 100:.1f}%"

                df = pd.DataFrame(plot_data, index=['Bias Disparity', 'Recommendations Type']).T.dropna()

                sns.boxplot(x='Recommendations Type', y='Bias Disparity', data=df, order=order, ax=axs[i])
                axs[i].plot(axs[i].get_xlim(), [0., 0.], 'k--')
                axs[i].set_xlabel("")

                axs[i].set_title(f"{attr.title()}: {demo_gr.title()}")

                axs[i].set_xticklabels([f"{x.get_text()} \n {exp_size.get(x.get_text(), '-')}" for x in axs[i].get_xticklabels()])
                i += 1

        fig.savefig(os.path.join(attr_path, f"boxplot.png"))
        plt.close()


# %%
def extract_all_exp_metrics_data(_exp_paths, hist_matrix=None):
    sens_attributes = config['sensitive_attributes']
    sensitive_maps = {sens_attr: train_data.dataset.field2id_token[sens_attr] for sens_attr in sens_attributes}

    hist_matrix = test_hist_matrix if hist_matrix is None else hist_matrix

    cols = [1, 2, 6, 3, "set", 8]
    col_names = ['user_id', 'topk_pred', 'cf_topk_pred', 'n_del_edges', 'topk_dist', 'topk_set_dist', 'del_edges']

    exp_dfs = {}
    result_data = {}
    n_users_data = {}
    topk_dist = {}
    for e_type, e_path in _exp_paths.items():
        if e_path is None:
            continue
        exps_data = utils.load_dp_exps_file(e_path)

        data = []
        for exp_entry in exps_data:
            for _exp in exp_entry:
                exp_row_data = [_exp[0]]
                for col in cols:
                    if col in [1, 2]:
                        exp_row_data.append(_exp[col])
                    elif col in [6]:
                        exp_row_data.append([int(_exp[col])] * len(exp_row_data[0]))
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

                data.extend(list(zip(*exp_row_data)))

        data_df = pd.DataFrame(data, columns=col_names)
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
                result_data[e_type][n_del][metric] = compute_result(gr_df, 'cf_topk_pred', metric, hist_matrix)

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


def plot_explanations_fairness_trend(_bd_data_all, _n_users_data_all, orig_disparity, config_ids, filter_cats=None):
    n_item_cats = train_data.dataset.field2id_token['class'].shape[0]
    if filter_cats is not None:
        n_item_cats -= len(filter_cats)
    sens_attributes = config['sensitive_attributes']
    sensitive_maps = {sens_attr: train_data.dataset.field2id_token[sens_attr] for sens_attr in sens_attributes}

    dg_counts = {}
    for attr in config['sensitive_attributes']:
        dg_counts[attr] = user_df[attr].value_counts().to_dict()
        dg_counts[attr] = {sensitive_maps[attr][dg]: dg_counts[attr][dg] for dg in dg_counts[attr]}

    for e_type in _bd_data_all:
        bd_data = _bd_data_all[e_type]
        for attr in sens_attributes:
            sens_map = sensitive_maps[attr]
            plots_path = os.path.join(get_plots_path(), 'comparison', '_'.join(config_ids), attr)
            if not os.path.exists(plots_path):
                os.makedirs(plots_path)
            for d_gr in sens_map:
                if orig_disparity[attr][d_gr] is None:
                    continue

                n_users_data = {k: f"{v[attr][d_gr] / dg_counts[attr][d_gr] * 100:.1f}"
                                for k, v in _n_users_data_all[e_type].items() if d_gr in v[attr]}

                plot_data = []
                plot_bd_keys = []
                for n_del in bd_data:
                    bd_gr_data = bd_data[n_del][attr][d_gr].numpy()
                    if filter_cats is not None:
                        bd_gr_data = bd_gr_data[~np.in1d(np.arange(bd_gr_data.shape[0]), filter_cats)]
                    if not np.isnan(bd_gr_data).all():
                        plot_data.extend(list(zip([n_del] * n_item_cats, [e_type] * n_item_cats, bd_gr_data)))
                        plot_bd_keys.append(n_del)

                if filter_cats is not None:
                    orig_disp = orig_disparity[attr][d_gr].numpy()[~np.in1d(np.arange(orig_disparity[attr][d_gr].shape[0]), filter_cats)]
                else:
                    orig_disp = orig_disparity[attr][d_gr].numpy()
                plot_data.extend(list(zip(
                    np.repeat(plot_bd_keys, n_item_cats),
                    np.repeat([f'GCMC'] * n_item_cats, len(plot_bd_keys)),
                    np.tile(orig_disp, len(plot_bd_keys))
                )))

                plot_df = pd.DataFrame(plot_data, columns=['# Del Edges', 'Attribute', 'Bias Disparity']).dropna()

                ax = sns.lineplot(x='# Del Edges', y='Bias Disparity', hue='Attribute', data=plot_df, ci="sd")
                n_ticks = len(ax.get_xticks())
                nud_keys = list(n_users_data.keys())
                xticks = np.linspace(1, max(nud_keys), n_ticks, dtype=int)
                xtick_keys = np.linspace(1, len(nud_keys) - 1, n_ticks, dtype=int)
                xtick_labels = [f"{xticks[i]} \n ({n_users_data[nud_keys[x - 1]]})" for i, x in enumerate(xtick_keys)]
                ax.set_xticks(xticks)
                ax.set_xticklabels(xtick_labels)
                ax.set_title(f"{attr.title()}: {d_gr}")

                ax.plot(ax.get_xlim(), [0., 0.], 'k--')

                filter_str = f"_filter_{','.join(map(str, filter_cats))}" if filter_cats is not None else ""

                plt.tight_layout()
                plt.savefig(os.path.join(plots_path, f'{d_gr}#lineplot_over_del_edges_{e_type}{filter_str}.png'))
                plt.close()


# %%
def plot_explanations_fairness_trend_dumbbell(_bd_all_data, orig_disparity, config_ids, sort="dots", n_bins=10):
    sens_attributes = config["sensitive_attributes"]
    sensitive_maps = {sens_attr: train_data.dataset.field2id_token[sens_attr] for sens_attr in sens_attributes}
    item_cats = train_data.dataset.field2id_token['class']

    x, y = 'Bias Disparity', 'Category'

    dot_size = 50

    if sort not in ["dots", "barplot_side"]:
        raise NotImplementedError(f"sort = {sort} not supported for dumbbell")

    for e_type in _bd_all_data:
        bd_data = _bd_all_data[e_type]
        for attr in sens_attributes:
            sens_map = sensitive_maps[attr]
            plots_path = os.path.join(get_plots_path(), 'comparison', '_'.join(config_ids), attr)
            if not os.path.exists(plots_path):
                os.makedirs(plots_path)

            exp_cats = item_cats
            if map_cats_exp[exp_t] is not None:
                exp_cats = [exp_cats[0]] + exp_cats[list(map_cats_exp[exp_t])[:-1]].tolist() + ["Other"]

            cats_inter_counts_attr = {}
            for demo_gr, demo_df in joint_df.groupby(attr):
                class_counts = demo_df.explode('class').value_counts('class')
                class_counts.index = class_counts.index.map(exp_cats.__getitem__)
                cats_inter_counts_attr[sensitive_maps[attr][demo_gr]] = class_counts.to_dict()

            for d_gr in sens_map:
                if orig_disparity[attr][d_gr] is None:
                    continue

                exp_data = []
                plot_bd_keys = []
                for n_del in bd_data:
                    bd_gr_data = bd_data[n_del][attr][d_gr].numpy()
                    if map_cats_exp[exp_t] is not None:
                        bd_gr_data = bd_gr_data[list(range(len(exp_cats)))]
                    if not np.isnan(bd_gr_data).all():
                        l = len(exp_cats)
                        exp_data.extend(list(zip([n_del] * l, bd_gr_data, exp_cats)))
                        plot_bd_keys.append(n_del)

                orig_disp = orig_disparity[attr][d_gr].numpy()
                orig_disp = orig_disp[list(range(len(exp_cats)))]
                orig_data = list(zip(['GCMC'] * len(exp_cats), orig_disp, exp_cats))

                df_orig = pd.DataFrame(orig_data, columns=['Attribute', x, y]).dropna()
                df_exp = pd.DataFrame(exp_data, columns=['# Del Edges', x, y]).dropna()

                max_del_edges = max(bd_data)
                bin_size = max_del_edges // n_bins
                bin_map = {i: f"{e_type}: {i * bin_size + 1 if i != 0 else 1}-{(i + 1) * bin_size}" for i in
                           range(max_del_edges // bin_size + 1)}

                df_exp['# Del Edges lab'] = df_exp['# Del Edges'].map(lambda x: bin_map[x // bin_size])
                df_exp['# Del Edges'] = df_exp['# Del Edges'].map(lambda x: (x // bin_size))  # * dot_size

                if sort == "dots":
                    order = df_orig.sort_values(x)[y].to_list()
                    bar_data = pd.DataFrame(cats_inter_counts_attr[d_gr].items(), columns=[y, x])
                elif sort == "barplot_side":
                    order, vals = map(list, zip(*(sorted(cats_inter_counts_attr[d_gr].items(), key=lambda x: x[1])[::-1])))
                    bar_data = pd.DataFrame(zip(order, vals), columns=[y, x])

                g = sns.JointGrid(height=12, space=0.5)
                # g.ax_marg_x.remove()
                sns.barplot(x=x, y=y, data=bar_data, ax=g.ax_marg_y, color="black", order=order)

                df_exp_plot = df_exp.groupby(['Category', '# Del Edges']).agg(**{
                    "Bias Disparity": pd.NamedAgg(column='Bias Disparity', aggfunc='mean'),
                    "# Del Edges lab": pd.NamedAgg(column='# Del Edges lab', aggfunc='first'),
                }).reset_index()

                norm = mpl.colors.Normalize(vmin=0, vmax=df_exp_plot['# Del Edges'].max())

                df_exp_plot_gb, df_orig_gb = df_exp_plot.groupby(y), df_orig.groupby(y)
                df_exp_plot = pd.concat([df_exp_plot_gb.get_group(g) for g in order])
                df_orig = pd.concat([df_orig_gb.get_group(g) for g in order])

                print(df_exp_plot)

                palette = sns.color_palette("Blues_d", as_cmap=True)
                # plot_palette = list(df_exp_plot['# Del Edges'].map(palette))

                sns.stripplot(x=x, y=y, color='#780808', data=df_orig, ax=g.ax_joint, s=20, marker="X", jitter=False,
                              label='GCMC', zorder=2) #, order=order)
                sns.scatterplot(x=x, y=y, hue="# Del Edges", size="# Del Edges", palette=palette, sizes=(50, 230), hue_norm=norm,
                                data=df_exp_plot, ax=g.ax_joint, zorder=2, legend="full")  # , jitter=False, order=order)

                handles, labels = zip(*utils.legend_without_duplicate_labels(g.ax_joint))
                df_exp_plot_sizes = df_exp_plot.set_index("# Del Edges")
                labels = [df_exp_plot_sizes.loc[int(l), "# Del Edges lab"].iloc[0] if l.isnumeric() else l for l in labels]
                g.ax_joint.legend(handles, labels)

                g.ax_joint.plot([0., 0.], g.ax_joint.get_ylim(), 'k--', zorder=1)
                g.ax_joint.set_title(f"{attr.title()}: {d_gr}")
                g.ax_joint.grid(axis='y', ls=(0, (1, 3)))

                _ax_j = g.ax_joint.twinx()
                _ax_j.tick_params(right=False)
                _ax_j.set_yticklabels([])
                _ax_j.set_ylabel('# Interactions for each category of items', rotation=270, labelpad=15)
                # g.ax_marg_x.set_title('# Interactions for each category of items')
                g.ax_joint.legend().remove()

                # g.ax_marg_x.get_shared_x_axes().remove(g.ax_joint)
                gs = plt.GridSpec(12, 6)

                ax_marg_x = g.fig.add_subplot(gs[1, :-1])
                fake_ax = g.fig.add_subplot(gs[0, :-1])
                fake_ax.set_visible(False)

                sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
                plt.colorbar(cax=ax_marg_x, mappable=sm, orientation='horizontal')

                cbar_xticks = [f"{x * 100:.2f}%" for x in np.linspace(
                    0.,
                    max_del_edges / train_data.dataset.inter_num,
                    len(ax_marg_x.get_xticklabels())
                )]
                ax_marg_x.set_xticklabels(cbar_xticks)

                plt.tight_layout()
                plt.savefig(os.path.join(plots_path, f'{d_gr}#dumbbell_over_del_edges_{e_type}.png'))
                plt.close()


# %%
def plot_explanations_fairness_trend_dumbbell_individual(_bd_all_data, orig_disparity, config_ids, sort="dots", n_bins=10):
    item_cats = train_data.dataset.field2id_token['class']

    x, y = 'Bias Disparity', 'Category'

    dot_size = 50

    if sort not in ["dots", "barplot_side"]:
        raise NotImplementedError(f"sort = {sort} not supported for dumbbell")

    for e_type in _bd_all_data:
        bd_data = _bd_all_data[e_type]
        plots_path = os.path.join(get_plots_path(), 'comparison', '_'.join(config_ids), 'individual')
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)

        class_counts = joint_df.explode('class').value_counts('class')
        class_counts.index = class_counts.index.map(item_cats.__getitem__)
        cats_inter_counts = class_counts.to_dict()

        exp_data = []
        plot_bd_keys = []
        for n_del in bd_data:
            bd_n_del_data = bd_data[n_del].numpy()
            if not np.isnan(bd_n_del_data).all():
                l = len(item_cats)
                exp_data.extend(list(zip([n_del] * l, bd_n_del_data, item_cats)))
                plot_bd_keys.append(n_del)

        orig_data = list(zip(['GCMC'] * len(item_cats), orig_disparity.numpy(), item_cats))

        df_orig = pd.DataFrame(orig_data, columns=['Attribute', x, y]).dropna()
        df_exp = pd.DataFrame(exp_data, columns=['# Del Edges', x, y]).dropna()

        max_del_edges = max(bd_data)
        bin_size = max_del_edges // n_bins
        bin_map = {i: f"{e_type}: {i * bin_size + 1 if i != 0 else 1}-{(i + 1) * bin_size}" for i in
                   range(max_del_edges // bin_size + 1)}

        df_exp['# Del Edges lab'] = df_exp['# Del Edges'].map(lambda x: bin_map[x // bin_size])
        df_exp['# Del Edges'] = df_exp['# Del Edges'].map(lambda x: (x // bin_size))  # * dot_size

        if sort == "dots":
            order = df_orig.sort_values(x)[y].to_list()
            bar_data = pd.DataFrame(cats_inter_counts.items(), columns=[y, x])
        elif sort == "barplot_side":
            order, vals = map(list, zip(*(sorted(cats_inter_counts.items(), key=lambda x: x[1])[::-1])))
            bar_data = pd.DataFrame(zip(order, vals), columns=[y, x])

        g = sns.JointGrid(height=12, space=0.5)
        # g.ax_marg_x.remove()
        sns.barplot(x=x, y=y, data=bar_data, ax=g.ax_marg_y, color="black", order=order)

        df_exp_plot = df_exp.groupby(['Category', '# Del Edges']).agg(**{
            "Bias Disparity": pd.NamedAgg(column='Bias Disparity', aggfunc='mean'),
            "# Del Edges lab": pd.NamedAgg(column='# Del Edges lab', aggfunc='first'),
        }).reset_index()

        norm = mpl.colors.Normalize(vmin=0, vmax=df_exp_plot['# Del Edges'].max())

        df_exp_plot_gb, df_orig_gb = df_exp_plot.groupby(y), df_orig.groupby(y)
        df_exp_plot = pd.concat([df_exp_plot_gb.get_group(g) for g in order])
        df_orig = pd.concat([df_orig_gb.get_group(g) for g in order])

        print(df_exp_plot)

        palette = sns.color_palette("Blues_d", as_cmap=True)
        # plot_palette = list(df_exp_plot['# Del Edges'].map(palette))

        sns.stripplot(x=x, y=y, color='#780808', data=df_orig, ax=g.ax_joint, s=25, marker="X", jitter=False,
                      label='GCMC', zorder=2) #, order=order)
        sns.scatterplot(x=x, y=y, hue="# Del Edges", size="# Del Edges", palette=palette, sizes=(50, 230), hue_norm=norm,
                        data=df_exp_plot, ax=g.ax_joint, zorder=2, legend="full")  # , jitter=False, order=order)

        handles, labels = zip(*utils.legend_without_duplicate_labels(g.ax_joint))
        df_exp_plot_sizes = df_exp_plot.set_index("# Del Edges")
        labels = [df_exp_plot_sizes.loc[int(l), "# Del Edges lab"].iloc[0] if l.isnumeric() else l for l in labels]
        g.ax_joint.legend(handles, labels)

        g.ax_joint.plot([0., 0.], g.ax_joint.get_ylim(), 'k--', zorder=1)
        g.ax_joint.grid(axis='y', ls=(0, (1, 3)))

        _ax_j = g.ax_joint.twinx()
        _ax_j.tick_params(right=False)
        _ax_j.set_yticklabels([])
        _ax_j.set_ylabel('# Interactions for each category of items', rotation=270, labelpad=15)
        # g.ax_marg_x.set_title('# Interactions for each category of items')
        g.ax_joint.legend().remove()

        # g.ax_marg_x.get_shared_x_axes().remove(g.ax_joint)
        gs = plt.GridSpec(12, 6)

        ax_marg_x = g.fig.add_subplot(gs[1, :-1])
        fake_ax = g.fig.add_subplot(gs[0, :-1])
        fake_ax.set_visible(False)

        sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
        plt.colorbar(cax=ax_marg_x, mappable=sm, orientation='horizontal')

        cbar_xticks = [f"{x * 100:.2f}%" for x in np.linspace(
            0.,
            max_del_edges / train_data.dataset.inter_num,
            len(ax_marg_x.get_xticklabels())
        )]
        ax_marg_x.set_xticklabels(cbar_xticks)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_path, f'dumbbell_over_del_edges_{e_type}.png'))
        plt.close()


# %%
def plot_bias_disparity_scatterplot(_bd_all_data, orig_disparity, config_ids, n_bins=10, filter_cats=None):
    sens_attributes = config["sensitive_attributes"]
    sensitive_maps = {sens_attr: train_data.dataset.field2id_token[sens_attr] for sens_attr in sens_attributes}
    item_cats = train_data.dataset.field2id_token['class']
    if filter_cats is not None:
        item_cats = item_cats[~np.in1d(np.arange(item_cats.shape[0]), filter_cats)]

    x, y = 'Bias Disparity', 'Category'

    for e_type in _bd_all_data:
        bd_data = _bd_all_data[e_type]
        for attr in sens_attributes:
            sens_map = sensitive_maps[attr]
            plots_path = os.path.join(get_plots_path(), 'comparison', '_'.join(config_ids), attr)
            if not os.path.exists(plots_path):
                os.makedirs(plots_path)

            for d_gr in sens_map:
                if orig_disparity[attr][d_gr] is None:
                    continue

                exp_data = []
                plot_bd_keys = []
                l = len(item_cats)
                for n_del in bd_data:
                    bd_gr_data = bd_data[n_del][attr][d_gr].numpy()
                    if not np.isnan(bd_gr_data).all():
                        exp_data.extend(list(zip([n_del] * l, bd_gr_data[~np.in1d(np.arange(bd_gr_data.shape[0]), filter_cats)], item_cats)))
                        plot_bd_keys.append(n_del)

                orig_disp = orig_disparity[attr][d_gr].numpy()
                orig_data = list(zip(['GCMC'] * len(item_cats), orig_disp[~np.in1d(np.arange(orig_disp.shape[0]), filter_cats)], item_cats))

                df_orig = pd.DataFrame(orig_data, columns=['Attribute', x, y]).dropna()
                df_exp = pd.DataFrame(exp_data, columns=['# Del Edges', x, y]).dropna()

                max_del_edges = max(bd_data)
                bin_size = max_del_edges // n_bins
                bin_map = {i: f"{e_type}: {i * bin_size + 1 if i != 0 else 1}-{(i + 1) * bin_size}" for i in
                           range(max_del_edges // bin_size + 1)}

                df_exp['# Del Edges lab'] = df_exp['# Del Edges'].map(lambda x: bin_map[x // bin_size])
                df_exp['# Del Edges'] = df_exp['# Del Edges'].map(lambda x: (x // bin_size))  # * dot_size

                # g = sns.JointGrid(height=12, ratio=11, space=0.5)
                # g.ax_marg_y.remove()
                # g.ax_marg_x.get_shared_x_axes().remove(g.ax_joint)

                fig = plt.figure(figsize=(12, 12))
                gs = plt.GridSpec(12, 6)
                ax_marg_x = fig.add_subplot(gs[0, :])
                ax_joint = fig.add_subplot(gs[1:, :])
                fig.subplots_adjust(hspace=1.0, wspace=1.0)

                ax_joint.spines['left'].set_position('zero')
                ax_joint.spines['bottom'].set_position('zero')

                # Eliminate upper and right axes
                ax_joint.spines['right'].set_color('none')
                ax_joint.spines['top'].set_color('none')

                # Show ticks in the left and lower axes only
                ax_joint.xaxis.set_ticks_position('bottom')
                ax_joint.yaxis.set_ticks_position('left')

                df_exp_plot = df_exp.groupby(['# Del Edges']).agg(**{
                    "Bias Disparity Mean": pd.NamedAgg(column='Bias Disparity', aggfunc='mean'),
                    "Bias Disparity Std": pd.NamedAgg(column='Bias Disparity', aggfunc='std'),
                    "# Del Edges lab": pd.NamedAgg(column='# Del Edges lab', aggfunc='first'),
                }).reset_index()

                norm = mpl.colors.Normalize(vmin=0, vmax=df_exp_plot['# Del Edges'].max())

                print(df_exp_plot)

                palette = sns.color_palette("Blues_d", as_cmap=True)

                sns.scatterplot(x="Bias Disparity Std", y="Bias Disparity Mean", hue="# Del Edges", size="# Del Edges",
                                palette=palette, sizes=(50, 230), hue_norm=norm,
                                data=df_exp_plot, ax=ax_joint, zorder=2, legend="full")  # , jitter=False, order=order)
                ax_joint.plot(df_orig['Bias Disparity'].std(), df_orig['Bias Disparity'].mean(),
                                color='#780808', ms=25, marker="X", zorder=2)
                ax_joint.plot(0, 0, ms=1)

                handles, labels = zip(*utils.legend_without_duplicate_labels(ax_joint))
                df_exp_plot_sizes = df_exp_plot.set_index("# Del Edges")
                labels = [df_exp_plot_sizes.loc[int(l), "# Del Edges lab"] if l.isnumeric() else l for l in labels]
                ax_joint.legend(handles, labels)

                ax_joint.set_title(f"{attr.title()}: {d_gr}")

                ax_joint.legend().remove()

                sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
                plt.colorbar(cax=ax_marg_x, mappable=sm, orientation='horizontal')

                cbar_xticks = [f"{x * 100:.2f}%" for x in np.linspace(
                    0.,
                    max_del_edges / train_data.dataset.inter_num,
                    len(ax_marg_x.get_xticklabels())
                )]
                ax_marg_x.set_xticklabels(cbar_xticks)

                filter_str = f"_filter_{','.join(map(str, filter_cats))}" if filter_cats is not None else ""

                plt.tight_layout()
                plt.savefig(os.path.join(plots_path, f'{d_gr}#scatterplot_bd_over_del_edges_{e_type}{filter_str}.png'))
                plt.close()


def create_table_bias_disparity(bd, config_ids):
    sens_attributes = config['sensitive_attributes']
    sensitive_maps = {sens_attr: train_data.dataset.field2id_token[sens_attr] for sens_attr in sens_attributes}

    order = ['GCMC', 'GCMC+BD', 'GCMC+NDCG']

    for attr, sens_map in sensitive_maps.items():
        tables_path = os.path.join(get_plots_path(), 'comparison', '_'.join(config_ids), attr)
        if not os.path.exists(tables_path):
            os.makedirs(tables_path)

        vals = []
        for exp_type in order:
            vals.append([])
            for demo_gr in sens_map:
                if exp_type != 'pred_explain' and exp_type in bd:
                    data = bd[exp_type][attr][demo_gr]
                    if data is None:
                        continue

                    vals[-1].extend([np.nanmean(data.numpy()), np.nanstd(data.numpy())])

        d_grs = [x for x in sens_map if bd['GCMC'][attr][x] is not None]

        plot_vals = []
        for row in vals:
            plot_vals.append([])
            for i in range(len(d_grs)):
                if row:
                    plot_vals[-1].append(f"{row[2 * i]:.2f} ({row[2 * i + 1]:.2f})")
                else:
                    plot_vals[-1].append("- (-)")

        df_attr = pd.DataFrame(plot_vals, columns=d_grs, index=order)
        df_attr.to_markdown(os.path.join(tables_path, f"bias_disparity_table_{attr}.md"), tablefmt="github")


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
            e_d_grs_df = e_df.join(user_df.set_index("user_id"), on="user_id")
            masks = {d_gr: e_d_grs_df[attr] == d_gr for d_gr in d_grs}

            ch_bins = []
            temp_exp_data = []
            temp_stats_data = []
            for n_del, bin_del in del_edges_map.items():
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
                    for (g1, g2) in itertools.combinations(d_grs_exp_data, 2):
                        new_d_grs_exp_data.append(abs(g1 - g2))
                    temp_exp_data.append(np.nansum(new_d_grs_exp_data))
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

                for (g1, g2) in itertools.combinations(temp_orig_data, 2):
                    orig_data.append(abs(g1 - g2))
                orig_data = [sum(orig_data)] * len(final_bins)

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

args = parser.parse_args() #  r"--model_file src\saved\GCMC-ML100K-Jun-01-2022_13-28-01.pth --explainer_config_file config\gcmc_explainer.yaml --load_config_ids -1 -1 82".split())

script_path = os.path.abspath(os.path.dirname(inspect.getsourcefile(lambda: 0)))
script_path = os.path.join(script_path, 'src') if 'src' not in script_path else script_path

if args.model_file is None:
    raise FileNotFoundError("need to specify a saved file with `--model_file`")

print(args)

config, model, dataset, train_data, valid_data, test_data = utils.load_data_and_model(args.model_file,
                                                                                      args.explainer_config_file)

user_num, item_num = dataset.user_num, dataset.item_num
test_hist_matrix, _, test_hist_len = test_data.dataset.history_item_matrix()
test_hist_matrix, test_hist_len = test_hist_matrix.numpy(), test_hist_len.numpy()
evaluator = Evaluator(config)

val_hist_matrix, _, val_hist_len = valid_data.dataset.history_item_matrix()
val_hist_matrix, val_hist_len = val_hist_matrix.numpy(), val_hist_len.numpy()

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
best_test_pref_data, best_test_result = extract_best_metrics(exp_paths, args.best_exp_col, hist_matrix=test_hist_matrix)
best_val_pref_data, best_val_result = extract_best_metrics(exp_paths, args.best_exp_col, hist_matrix=val_hist_matrix)

# %%
all_exp_test_dfs, test_result_all_data, test_n_users_data_all, test_topk_dist_all = extract_all_exp_metrics_data(
    exp_paths, hist_matrix=test_hist_matrix
)
all_exp_val_dfs, val_result_all_data, val_n_users_data_all, val_topk_dist_all = extract_all_exp_metrics_data(
    exp_paths, hist_matrix=val_hist_matrix
)

cleaned_config_ids = list(map(str, args.load_config_ids))

# %%
# create_table_bias_disparity(bias_disparity, cleaned_config_ids)

# plot_bias_disparity_diff_dumbbell(bias_disparity, sens_attrs, cleaned_config_ids, sort="barplot_side")

# %%
# plot_bias_disparity_boxplot(bias_disparity, pref_data_topk_all, sens_attrs, cleaned_config_ids)

# %%
# plot_bias_disparity_scatterplot(bd_all_data, bias_disparity['GCMC'], cleaned_config_ids, n_bins=10, filter_cats=[14, 16, 19])

# %%
# plot_explanations_fairness_trend(bd_all_data, n_users_data_all, bias_disparity['GCMC'], cleaned_config_ids, filter_cats=[14, 16, 19])

# %%
# plot_explanations_fairness_trend_dumbbell(bd_all_data, bias_disparity['GCMC'], cleaned_config_ids, sort="barplot_side", n_bins=20)

# %%
# plot_dist_over_del_edges(topk_dist_all, bd_all_data, cleaned_config_ids, max_del_edges=80 if batch_exp_s == 'individual' else np.inf)

# %%
create_table_metrics_over_del_edges(
    test_result_all_data,
    best_test_pref_data,
    best_test_result['GCMC'],
    cleaned_config_ids,
    n_bins=100,
    hist_type="test",
    test_f="f_oneway"
)
create_table_metrics_over_del_edges(
    val_result_all_data,
    best_val_pref_data,
    best_val_result['GCMC'],
    cleaned_config_ids,
    n_bins=10,
    hist_type="val",
    test_f="f_oneway"
)

# %%
# plot_del_edges_hops(all_exp_dfs, cleaned_config_ids)

