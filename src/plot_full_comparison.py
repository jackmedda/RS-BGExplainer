# %%
import os
import pickle
import argparse
import inspect
import itertools

import tqdm
import scipy
import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cbook as mpl_cbook
import matplotlib.ticker as mpl_tick
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpl_path_eff
import sklearn.feature_selection as sk_feats
from recbole.evaluator import Evaluator

import src.utils as utils
import src.utils.plot_utils as plot_utils


# %%
def get_plots_path(datasets_names, model_names):
    plots_path = os.path.join(
        script_path,
        os.pardir,
        f'dp_plots',
        datasets_names,
        model_names
    )

    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    return plots_path


def update_plot_data(_test_df_data, _rec_df_data, additional_best_cols=None):
    test_orig_total_metric = best_test_exp_result[model_name][metric] if best_test_exp_result is not None else None
    rec_orig_total_metric = best_rec_exp_result[model_name][metric]

    test_pert_total_metric = best_test_exp_result[model_dp_s][metric] if best_test_exp_result is not None else None
    rec_pert_total_metric = best_rec_exp_result[model_dp_s][metric]

    m_group_mask = best_rec_exp_df[model_dp_s].user_id.isin(user_df.loc[user_df[sens_attr] == m_idx, uid_field])
    f_group_mask = best_rec_exp_df[model_dp_s].user_id.isin(user_df.loc[user_df[sens_attr] == f_idx, uid_field])

    rec_orig_m_metric = rec_orig_total_metric[m_group_mask, -1].mean()
    rec_orig_f_metric = rec_orig_total_metric[f_group_mask, -1].mean()

    if rec_orig_m_metric >= rec_orig_f_metric:
        if delete_adv_group is not None:
            _group_edge_del = m_idx if delete_adv_group else f_idx
        else:
            _group_edge_del = m_idx
    else:
        if delete_adv_group is not None:
            _group_edge_del = f_idx if delete_adv_group else m_idx
        else:
            _group_edge_del = f_idx

    if best_test_exp_result is not None:
        # Adding data from test results
        test_zip_data = [
            test_uid,
            [sens_attr.title().replace('_', ' ')] * len(test_uid),
            user_df.set_index(uid_field).loc[test_uid, sens_attr].map(attr_map.__getitem__).map(real_group_map[sens_attr]),
            [model_name] * len(test_uid),
            [dataset_name] * len(test_uid),
            [metric.upper()] * len(test_uid)
        ]
        if (dataset_name, model_name, sens_attr) not in no_policies:
            zip_data = []
            zip_data.extend(test_zip_data)
            zip_data.append(test_orig_total_metric[:, -1])
            zip_data.append([no_pert_col] * len(test_uid))
            for add_col in additional_best_cols:
                if add_col in best_test_exp_df[model_dp_s]:
                    zip_data.append(best_test_exp_df[model_dp_s][add_col].to_numpy())

            _test_df_data.extend(list(zip(*zip_data)))

        test_zip_data.append(test_pert_total_metric[:, -1])
        test_zip_data.append([policy] * len(test_uid))
        for add_col in additional_best_cols:
            if add_col in best_test_exp_df[model_dp_s]:
                test_zip_data.append(best_test_exp_df[model_dp_s][add_col].to_numpy())

        _test_df_data.extend(list(zip(*test_zip_data)))

    # Adding data from rec results
    rec_zip_data = [
        rec_uid,
        [sens_attr.title().replace('_', ' ')] * len(rec_uid),
        user_df.set_index(uid_field).loc[rec_uid, sens_attr].map(attr_map.__getitem__).map(real_group_map[sens_attr]),
        [model_name] * len(rec_uid),
        [dataset_name] * len(rec_uid),
        [metric.upper()] * len(rec_uid)
    ]
    if (dataset_name, model_name, sens_attr) not in no_policies:
        zip_data = []
        zip_data.extend(rec_zip_data)
        zip_data.append(rec_orig_total_metric[:, -1])
        zip_data.append([no_pert_col] * len(rec_uid))
        for add_col in additional_best_cols:
            if add_col in best_rec_exp_df[model_dp_s]:
                zip_data.append(best_rec_exp_df[model_dp_s][add_col].to_numpy())

        _rec_df_data.extend(list(zip(*zip_data)))

    rec_zip_data.append(rec_pert_total_metric[:, -1])
    rec_zip_data.append([policy] * len(rec_uid))
    for add_col in additional_best_cols:
        if add_col in best_rec_exp_df[model_dp_s]:
            rec_zip_data.append(best_rec_exp_df[model_dp_s][add_col].to_numpy())

    _rec_df_data.extend(list(zip(*rec_zip_data)))

    return _group_edge_del


def update_plot_del_data(_test_df_del_data, _rec_df_del_data):
    filter_cols = ['user_id', 'epoch', 'n_del_edges', 'fair_loss']

    exp_test_df = all_exp_test_dfs[model_dp_s][filter_cols] if all_exp_test_dfs is not None else None
    exp_rec_df = all_exp_rec_dfs[model_dp_s][filter_cols]
    uid_list_test = next(exp_test_df.groupby('n_del_edges').__iter__())[1].user_id if exp_test_df is not None else None
    uid_list_rec = next(exp_rec_df.groupby('n_del_edges').__iter__())[1].user_id

    result_test_df_data, result_rec_df_data = [], []
    if test_result_all_data is not None:
        for n_del, res in test_result_all_data[model_dp_s].items():
            result_test_df_data.extend(list(zip(
                [n_del] * len(uid_list_test),
                uid_list_test,
                res[metric][:, -1],
                [metric.upper()] * len(uid_list_test)
            )))
    for n_del, res in rec_result_all_data[model_dp_s].items():
        result_rec_df_data.extend(list(zip(
            [n_del] * len(uid_list_rec),
            uid_list_rec,
            res[metric][:, -1],
            [metric.upper()] * len(uid_list_rec)
        )))

    if exp_test_df is not None:
        exp_test_df = exp_test_df.join(
            pd.DataFrame(
                result_test_df_data, columns=['n_del_edges', 'user_id', 'Value', 'Metric']
            ).set_index(['n_del_edges', 'user_id']),
            on=['n_del_edges', 'user_id']
        ).join(user_df.set_index(uid_field), on='user_id')
    exp_rec_df = exp_rec_df.join(
        pd.DataFrame(
            result_rec_df_data, columns=['n_del_edges', 'user_id', 'Value', 'Metric']
        ).set_index(['n_del_edges', 'user_id']),
        on=['n_del_edges', 'user_id']
    ).join(user_df.set_index(uid_field), on='user_id')

    _test_result = exp_test_df.pop("Value") if exp_test_df is not None else None
    _rec_result = exp_rec_df.pop("Value")

    test_orig_total_metric = best_test_exp_result[model_name][metric][:, -1] if best_test_exp_result is not None else None
    rec_orig_total_metric = best_rec_exp_result[model_name][metric][:, -1]

    unique_test_del_edges = len(test_result_all_data[model_dp_s]) if test_result_all_data is not None else None
    unique_rec_del_edges = len(rec_result_all_data[model_dp_s])

    if exp_test_df is not None:
        if (dataset_name, model_name, sens_attr) not in no_policies:
            _test_df_del_data.extend(
                np.c_[
                    exp_test_df.values,
                    [sens_attr.title().replace('_', ' ')] * len(exp_test_df),
                    [model_name] * len(exp_test_df),
                    [dataset_name] * len(exp_test_df),
                    np.tile(test_orig_total_metric, unique_test_del_edges),
                    [no_pert_col] * len(exp_test_df)
                ].tolist()
            )
        _test_df_del_data.extend(
            np.c_[
                exp_test_df.values,
                [sens_attr.title().replace('_', ' ')] * len(exp_test_df),
                [model_name] * len(exp_test_df),
                [dataset_name] * len(exp_test_df),
                _test_result.to_numpy(),
                [policy] * len(exp_test_df)
            ].tolist()
        )

    if (dataset_name, model_name, sens_attr) not in no_policies:
        _rec_df_del_data.extend(
            np.c_[
                exp_rec_df.values,
                [sens_attr.title().replace('_', ' ')] * len(exp_rec_df),
                [model_name] * len(exp_rec_df),
                [dataset_name] * len(exp_rec_df),
                np.tile(rec_orig_total_metric, unique_rec_del_edges),
                [no_pert_col] * len(exp_rec_df)
            ].tolist()
        )
    _rec_df_del_data.extend(
        np.c_[
            exp_rec_df.values,
            [sens_attr.title().replace('_', ' ')] * len(exp_rec_df),
            [model_name] * len(exp_rec_df),
            [dataset_name] * len(exp_rec_df),
            _rec_result.to_numpy(),
            [policy] * len(exp_rec_df)
        ].tolist()
    )


def create_table_best_explanations_per_group(_metric_df):
    nop_mask = _metric_df["Policy"] == no_pert_col
    metr_df_nop = _metric_df[nop_mask].copy()
    metr_df_p = _metric_df[~nop_mask].copy()

    metr_df_nop["Status"] = "Before"
    metr_df_p["Status"] = "After"

    metr_df = pd.concat(
        [metr_df_p] + [metr_df_nop.copy().replace(no_pert_col, p) for p in metr_df_p.Policy.unique()],
        ignore_index=True
    )

    metr_df_mean = metr_df.groupby(
        ["Dataset", "Model", "Policy", "Status", "Sens Attr", "Demo Group"]
    ).mean()

    metric_col = metr_df_mean.columns[0]

    metr_stest = {}
    metr_df_gb = metr_df.groupby(
        ["Dataset", "Model", "Policy", "Sens Attr"]
    )
    for mdf_stat_cols, mdf_stat in metr_df_gb:
        stat_dg_gby = mdf_stat.groupby("Demo Group")
        s_attr_stat = dict.fromkeys(stat_dg_gby.groups.keys())
        tot_val = dict.fromkeys(["Before", "After"])
        for stat_dg, stat_dg_df in stat_dg_gby:
            after_mask = stat_dg_df["Status"] == "After"
            after_stat_val = stat_dg_df.loc[after_mask, metric_col]
            before_stat_val = stat_dg_df.loc[~after_mask, metric_col]
            for stat_status, stat_status_val in zip(["Before", "After"], [before_stat_val, after_stat_val]):
                if tot_val[stat_status] is None:
                    tot_val[stat_status] = stat_status_val
                else:
                    tot_val[stat_status] = (tot_val[stat_status].to_numpy() + stat_status_val.to_numpy()) / 2
            s_attr_stat[stat_dg] = scipy.stats.ttest_rel(before_stat_val, after_stat_val)
        s_attr_stat["Total"] = scipy.stats.ttest_rel(*tot_val.values())
        metr_stest[mdf_stat_cols] = s_attr_stat

    metr_df_pivot = metr_df_mean.reset_index().pivot(
        index=["Dataset", "Model", "Policy"],
        columns=["Sens Attr", "Demo Group", "Status"],
        values=metric_col
    )

    table_df = metr_df_pivot.reindex(
        ['Gender', 'Age', 'User Wide Zone'], axis=1, level=0
    ).reindex(
        ["M", "F", "Y", "O", "America", "Other"], axis=1, level=1
    ).reindex(
        ['Before', 'After'], axis=1, level=2
    )
    level_attrs = ["Gender", "Age", "User Wide Zone"]
    attrs_demo_groups = [["M", "F"], ["Y", "O"], ["America", "Other"]]

    def relative_change(row, l_attr, row_dg):
        v1, v2 = row.loc["After"].item(), row.loc["Before"].item()
        ch = v1 - v2
        stat_pv = metr_stest[(*row.name, l_attr)][row_dg].pvalue
        stat_s = '*' if stat_pv < P_VALUE else '\hspace{3pt}'
        zeros = "04.1f" if ch >= 0 else "05.1f"
        return f'{v1:.2f}{stat_s} ({"+" if ch >= 0 else ""}{(ch / v2) * 100:{zeros}}%)'

    for level_attr, demo_groups in zip(level_attrs, attrs_demo_groups):
        final_table_df = []
        if level_attr in table_df:
            table_total_df = (table_df[(level_attr, demo_groups[0])] + table_df[(level_attr, demo_groups[1])]) / 2
            for tab_dg in demo_groups + ["Total"]:
                curr_tab_df = table_df[(level_attr, tab_dg)] if tab_dg in demo_groups else table_total_df

                tab_dg_df = curr_tab_df.apply(lambda row: relative_change(row, level_attr, tab_dg), axis=1).to_frame()
                tab_dg_df.columns = pd.MultiIndex.from_product([[level_attr], [tab_dg]])

                final_table_df.append(tab_dg_df)
            final_table_df = pd.concat(final_table_df, axis=1)
            final_table_df.columns = final_table_df.columns.map(lambda x: (x[0], group_name_map.get(x[1], x[1])))
            table_out_bar_df = pd.melt(final_table_df, ignore_index=False).reset_index()
            table_out_bar_df.to_csv(os.path.join(plots_path, f"total_table_{level_attr}_{exp_data_name}_{metric}_best_epoch.csv"))
            final_table_df.columns.names = [''] * len(final_table_df.columns.names)
            final_table_df.index = final_table_df.index.map(lambda x: (dataset_map[x[0]], *x[1:]))
            final_table_df.replace('%', '\%', regex=True).to_latex(
                os.path.join(plots_path, f"total_table_{level_attr}_{exp_data_name}_{metric}_best_epoch.tex"),
                multicolumn_format="c",
                escape=False
            )


def create_table_best_explanations(_metric_df):
    nop_mask = _metric_df["Policy"] == no_pert_col
    metr_df_nop = _metric_df[nop_mask].copy()
    metr_df_p = _metric_df[~nop_mask].copy()

    metr_df_nop["Status"] = "Before"
    metr_df_p["Status"] = "After"

    metr_df = pd.concat(
        [metr_df_p] + [metr_df_nop.copy().replace(no_pert_col, p) for p in metr_df_p.Policy.unique()],
        ignore_index=True
    )

    metr_df_mean = metr_df.groupby(
        ["Dataset", "Model", "Policy", "Status", "Sens Attr", "Demo Group"]
    ).mean().reset_index()
    metr_df_pivot = metr_df_mean.pivot(
        index=["Dataset", "Model", "Policy"],
        columns=["Sens Attr", "Demo Group", "Status"],
        values="Value"
    )
    table_df = metr_df_pivot.reindex(
        ['Gender', 'Age', 'User Wide Zone'], axis=1, level=0
    ).reindex(
        ["M", "F", "Y", "O", "America", "Other"], axis=1, level=1
    ).reindex(
        ['Before', 'After'], axis=1, level=2
    )
    level_attrs = ["Gender", "Age", "User Wide Zone"]
    attrs_demo_groups = [["M", "F"], ["Y", "O"], ["America", "Other"]]

    for level_attr, demo_groups in zip(level_attrs, attrs_demo_groups):
        if level_attr in table_df:
            table_dp_df = (table_df[(level_attr, demo_groups[0])] - table_df[(level_attr, demo_groups[1])]).abs()
            table_dp_df.columns = pd.MultiIndex.from_product([
                [level_attr], [f"{level_attr} $\Delta$ {metric.upper()}"], ["Before", "After"]
            ])
            table_df = pd.concat([table_df, table_dp_df], axis=1)
    table_df.columns = table_df.columns.map(lambda x: (x[0], group_name_map.get(x[1], x[1]), x[2]))
    table_out_bar_df = pd.melt(table_df, ignore_index=False).reset_index()
    table_out_bar_df.to_csv(os.path.join(plots_path, f"table_{exp_data_name}_{metric}_best_epoch.csv"))
    table_df.columns.names = [''] * len(table_df.columns.names)
    table_df.round(3).to_latex(
        os.path.join(plots_path, f"table_{exp_data_name}_{metric}_best_epoch.tex"),
        multicolumn_format="c",
        escape=False
    )

    return table_out_bar_df


def create_table_topk_list_change(data_df: pd.DataFrame, col_dist='Edit Dist'):
    info_cols = ["Dataset", "Model", "Policy", "Sens Attr", "Demo Group"]

    data_df = data_df[data_df["Policy"] != no_pert_col].reset_index(drop=True)
    data_df = data_df.drop(data_df.columns[~data_df.columns.isin(info_cols + [col_dist])], axis=1)

    mean_col, std_col = col_dist + ' Mean', col_dist + ' Std'

    data_df_gby = data_df.groupby(info_cols)
    data_df_mean = data_df_gby.mean()
    data_df_mean = data_df_mean.rename(columns={col_dist: mean_col})
    data_df_std = data_df_gby.std()
    data_df_std = data_df_std.rename(columns={col_dist: std_col})

    data_df_stats = data_df_mean.join(data_df_std)
    data_df_stats[col_dist] = data_df_stats.apply(
        lambda x: f"{x[mean_col]:.2f} ($\pm$ {x[std_col]:.2f})", axis=1
    )
    data_df_stats = data_df_stats.drop([mean_col, std_col], axis=1).reset_index()
    data_df_pivot = data_df_stats.pivot(
        index=["Dataset", "Model", "Policy"],
        columns=["Sens Attr", "Demo Group"],
        values=col_dist
    )
    table_df = data_df_pivot.reindex(
        ['Gender', 'Age', 'User Wide Zone'], axis=1, level=0
    ).reindex(
        ["M", "F", "Y", "O", "America", "Other"], axis=1, level=1
    )

    table_df.columns = table_df.columns.map(lambda x: (x[0], group_name_map.get(x[1], x[1])))
    table_out_bar_df = pd.melt(table_df, ignore_index=False).reset_index()
    table_out_bar_df.to_csv(os.path.join(plots_path, f"table_{exp_data_name}_topk_dist_best_epoch.csv"))
    table_df.columns.names = [''] * len(table_df.columns.names)
    table_df.to_latex(
        os.path.join(plots_path, f"table_{exp_data_name}_{col_dist.replace(' ', '_')}_best_epoch.tex"),
        multicolumn_format="c",
        escape=False
    )


def create_fig_bar2_legend(fig, _palette, _hatches, demo_groups, loc="upper left"):
    policies, pol_colors = zip(*_palette.items())
    fig.legend(
        [mpatches.Rectangle((0, 0), width=0, height=0)] +
        [mpatches.Rectangle((0, 0), width=2, height=0.7, edgecolor=c, facecolor=c) for c in pol_colors] +
        [mpatches.Rectangle((0, 0), width=0, height=0)] +
        [mpatches.Patch(facecolor='w', alpha=0, hatch=h) for h in _hatches],
        ['Policy'] + list(policies) + ['Demographic Groups'] + demo_groups,
        # bbox_to_anchor=(1, 1)
        loc=loc
    )


def clean_quantile_ax(_fig, _ax, _model, iteration, max_iteration, bbox_to_anchor=None, zorder=10):
    bbox_to_anchor = (1.15, 0.5) if bbox_to_anchor is None else bbox_to_anchor

    _ax.set_xticks([])
    _ax.set_xlabel(_model)
    _ax.set_ylabel("")
    _ax.spines.bottom.set_color('none')
    _ax.spines.right.set_color('none')
    _ax.spines.top.set_position(('data', 0))
    _ax.spines.top.set_linestyle((0, (4, 4)))

    if iteration == max_iteration - 1:
        _handles, _labels = _ax.get_legend_handles_labels()
        _ax.get_legend().remove()
        _legend = _fig.legend(
            _handles, _labels, loc="center right", bbox_to_anchor=bbox_to_anchor,
            bbox_transform=_fig.transFigure
        )
        _legend.set_zorder(zorder)
    else:
        _ax.get_legend().remove()


# %%
parser = argparse.ArgumentParser()
parser.add_argument('--model_files', nargs='+', required=True)
parser.add_argument('--explainer_config_files', required=True, nargs='+', type=str)
parser.add_argument('--iterations', default=100)
parser.add_argument('--overwrite_plot_data', '--opd', action="store_true")
parser.add_argument('--overwrite_extracted_data', '--oed', action="store_true")
parser.add_argument('--overwrite_graph_metrics', '--ogm', action="store_true")

args = parser.parse_args()

assert len(args.model_files) == len(args.explainer_config_files), \
    "Pass the same number of perturbed model files and configuration files to be loaded"

script_path = os.path.abspath(os.path.dirname(inspect.getsourcefile(lambda: 0)))
script_path = os.path.join(script_path, 'src') if 'src' not in script_path else script_path

print(args)

policy_map = {
    'force_removed_edges': 'MonDel',   # Monotonic Deletions
    'group_deletion_constraint': 'DelCons'  # Deletion Constraint
}

dataset_map = {
    "ml-100k": "ML 100K",
    "ml-1m": "ML 1M",
    "lastfm-1k": "Last.FM 1K",
    "coco_8_America": "COCO 8 (America)",
    "insurance": "Insurance",
    "yelp40": "Yelp 40",
    "rent_the_runway": "Rent The Runway",
    "tafeng": "Ta Feng"
}


real_group_map = {
    'gender': {'M': 'M', 'F': 'F'},
    'age': {'M': 'Y', 'F': 'O'},
    'user_wide_zone': {'M': 'America', 'F': 'Other'}
}

group_name_map = {
    "M": "Males",
    "F": "Females",
    "Y": "Younger",
    "O": "Older",
    "America": "America",
    "Other": "Other"
}

colors = {
    "Gender": {"M": "#0173b2", "F": "#de8f05"},
    "Age": {"Y": "#0173b2", "O": "#de8f05"}
}

P_VALUE = 0.05

no_pert_col = "NoPerturbation"

exp_epochs, config_ids, datasets_list, models_list, sens_attrs = [], [], [], [], []
for exp_config_file in args.explainer_config_files:
    _, dset, model, _, s_attr, eps, cid, _ = exp_config_file.split('dp_ndcg_explanations')[1].split(os.sep)
    datasets_list.append(dset)
    models_list.append(model)
    sens_attrs.append(s_attr)
    exp_epochs.append(eps.replace('epochs_', ''))
    config_ids.append(cid)

unique_datasets, unique_models, unique_sens_attrs = \
    np.unique(datasets_list).tolist(), np.unique(models_list).tolist(), np.unique(sens_attrs).tolist()
plots_path = os.path.join(
    get_plots_path('_'.join(unique_datasets), '_'.join(unique_models)),
    '_'.join(exp_epochs),
    '_'.join(config_ids),
    '_'.join(sens_attrs)
)
if not os.path.exists(plots_path):
    os.makedirs(plots_path)

if os.path.exists(os.path.join(plots_path, 'rec_df.csv')) and \
        os.path.exists(os.path.join(plots_path, 'incdisp.pkl')) and not args.overwrite_plot_data:
    test_rows, rec_rows = 2, 3

    rec_df = pd.read_csv(os.path.join(plots_path, 'rec_df.csv'), skiprows=rec_rows)
    rec_del_df = pd.read_csv(os.path.join(plots_path, 'rec_del_df.csv'), skiprows=rec_rows)

    with open(os.path.join(plots_path, 'rec_df.csv'), 'r') as f:
        metadata = [next(f) for _ in range(rec_rows)]
        exp_rec_data = metadata[2].split(': ')[1].strip()

    if exp_rec_data != "test":
        test_df = pd.read_csv(os.path.join(plots_path, 'test_df.csv'), skiprows=test_rows)
        test_del_df = pd.read_csv(os.path.join(plots_path, 'test_del_df.csv'), skiprows=test_rows)
    else:
        test_df, test_del_df = None, None

    with open(os.path.join(plots_path, 'incdisp.pkl'), 'rb') as f:
        incdisp = pickle.load(f)

    with open(os.path.join(plots_path, 'del_edges.pkl'), 'rb') as f:
        del_edges = pickle.load(f)

    with open(os.path.join(plots_path, 'all_batch_exps.pkl'), 'rb') as f:
        all_batch_exps = pickle.load(f)

    with open(os.path.join(plots_path, 'train_datasets.pkl'), 'rb') as f:
        train_datasets = pickle.load(f)

    with open(os.path.join(plots_path, 'datasets_train_inter_sizes.pkl'), 'rb') as f:
        datasets_train_inter_sizes = pickle.load(f)

else:
    # %%
    incdisp = {}
    del_edges = {}
    no_policies = set()
    all_batch_exps = {}
    train_datasets = {}
    datasets_train_inter_sizes = {}
    test_df_data, rec_df_data = [], []
    test_del_df_data, rec_del_df_data = [], []
    for model_file, exp_config_file in zip(args.model_files, args.explainer_config_files):
        config, model, dataset, train_data, valid_data, test_data = utils.load_data_and_model(model_file, exp_config_file)

        dataset_name = dataset.dataset_name
        model_name = model.__class__.__name__
        sens_attr, epochs, batch_exp = config['sensitive_attribute'], config['cf_epochs'], config['user_batch_exp']
        cf_topk = config['cf_topk']
        policy = '+'.join([pm for p, pm in policy_map.items() if config['explainer_policies'][p]])
        incdisp[(dataset_name, model_name)] = 'IncDisp' if config['explainer_policies']['increase_disparity'] else ''  # Increase Disparity
        all_batch_exps[dataset_name] = batch_exp

        train_datasets[dataset_name] = train_data.dataset
        datasets_train_inter_sizes[dataset_name] = train_data.dataset.inter_num

        exp_epochs.append(epochs)

        edge_additions = config['edge_additions']
        exp_rec_data = config['exp_rec_data']
        delete_adv_group = config['delete_adv_group']
        rec_data = locals()[f"{exp_rec_data}_data"]

        uid_field = train_data.dataset.uid_field

        user_df = pd.DataFrame({
            uid_field: train_data.dataset.user_feat[uid_field].numpy(),
            sens_attr: train_data.dataset.user_feat[sens_attr].numpy()
        })

        attr_map = dataset.field2id_token[sens_attr]
        f_idx, m_idx = (attr_map == 'F').nonzero()[0][0], (attr_map == 'M').nonzero()[0][0]
        user_num, item_num = dataset.user_num, dataset.item_num

        config['metrics'] = ['Recall', 'NDCG', 'MRR', 'Hit']  # force all these metrics if does not have them in its config
        evaluator = Evaluator(config)

        metrics = evaluator.metrics
        model_dp_s = f'{model_name}+FairDP'

        exp_paths = {model_dp_s: os.path.dirname(exp_config_file)}

        del model
        del dataset

        additional_best_cols = ['test_cf_dist', 'rec_cf_dist']
        if exp_rec_data != "test":
            best_test_exp_df, best_test_exp_result = plot_utils.extract_best_metrics(
                exp_paths,
                'auto',
                evaluator,
                test_data.dataset,
                config=config,
                additional_cols=additional_best_cols[:1]
            )
        else:
            best_test_exp_df, best_test_exp_result = None, None
        best_rec_exp_df, best_rec_exp_result = plot_utils.extract_best_metrics(
            exp_paths,
            'auto',
            evaluator,
            rec_data.dataset,
            config=config,
            additional_cols=additional_best_cols[1:]
        )

        for exp_df in [best_test_exp_df, best_rec_exp_df]:
            if exp_df is not None:
                set_dist = np.array([
                    cf_topk - len(set(orig) & set(pred))
                    for orig, pred in zip(exp_df[model_dp_s]['topk_pred'], exp_df[model_dp_s]['topk_pred'])
                ])
                exp_df[model_dp_s]['set_dist'] = set_dist

        # the deleted edges are repeated for each row, so take the first is the same
        test_del_edges = best_test_exp_df[model_dp_s]['del_edges'].iloc[0].tolist() if best_test_exp_df is not None else None
        rec_del_edges = best_rec_exp_df[model_dp_s]['del_edges'].iloc[0].tolist()
        for exp_data_name, exp_del_edges in zip(["test", exp_rec_data], [test_del_edges, rec_del_edges]):
            if exp_del_edges is not None:
                for policy_type in [no_pert_col, policy]:
                    del_edges[(exp_data_name, dataset_name, model_name, policy_type, sens_attr.title().replace('_', ' '))] = exp_del_edges

        test_uid = best_test_exp_df[model_dp_s]['user_id'].to_numpy() if best_test_exp_df is not None else None
        rec_uid = best_rec_exp_df[model_dp_s]['user_id'].to_numpy()

        if exp_rec_data != "test":
            all_exp_test_dfs, test_result_all_data, _, _ = plot_utils.extract_all_exp_metrics_data(
                exp_paths,
                train_data,
                test_data.dataset,
                evaluator,
                sens_attr,
                rec=False,
                overwrite=args.overwrite_extracted_data
            )
        else:
            all_exp_test_dfs, test_result_all_data = None, None

        all_exp_rec_dfs, rec_result_all_data, _, _ = plot_utils.extract_all_exp_metrics_data(
            exp_paths,
            train_data,
            rec_data.dataset,
            evaluator,
            sens_attr,
            rec=True,
            overwrite=args.overwrite_extracted_data
        )

        for metric in metrics:
            group_edge_del = update_plot_data(
                test_df_data,
                rec_df_data,
                additional_best_cols=additional_best_cols + ["set_dist"]
            )
            update_plot_del_data(test_del_df_data, rec_del_df_data)

        no_policies.add((dataset_name, model_name, sens_attr))

    cols = ['user_id', 'Sens Attr', 'Demo Group', 'Model', 'Dataset', 'Metric', 'Value', 'Policy',
            'Edit Dist', 'Set Dist']
    duplicated_cols_subset = [c for c in cols if c not in ['Value']]
    rec_df = pd.DataFrame(rec_df_data, columns=cols).drop_duplicates(subset=duplicated_cols_subset, ignore_index=True)

    del_cols = ['user_id', 'Epoch', '# Del Edges', 'Fair Loss',
                'Metric', 'Demo Group', 'Sens Attr', 'Model', 'Dataset', 'Value', 'Policy']
    duplicated_del_cols_subset = [c for c in del_cols if c not in ['Value', 'Fair Loss', 'Epoch']]
    rec_del_df = pd.DataFrame(rec_del_df_data, columns=del_cols).drop_duplicates(
        subset=duplicated_del_cols_subset, ignore_index=True
    )

    if exp_rec_data != "test":
        test_df = pd.DataFrame(test_df_data, columns=cols).drop_duplicates(
            subset=duplicated_cols_subset, ignore_index=True
        )
        test_del_df = pd.DataFrame(test_del_df_data, columns=del_cols).drop_duplicates(
            subset=duplicated_del_cols_subset, ignore_index=True
        )

        with open(os.path.join(plots_path, 'test_df.csv'), 'w') as f:
            f.write(f'# model_files {" ".join(args.model_files)}\n')
            f.write(f'# explainer_config_files {" ".join(args.explainer_config_files)}\n')
            test_df.to_csv(f, index=None)

        with open(os.path.join(plots_path, 'test_del_df.csv'), 'w') as f:
            f.write(f'# model_files {" ".join(args.model_files)}\n')
            f.write(f'# explainer_config_files {" ".join(args.explainer_config_files)}\n')
            test_del_df.to_csv(f, index=None)
    else:
        test_df, test_del_df = None, None

    with open(os.path.join(plots_path, 'rec_df.csv'), 'w') as f:
        f.write(f'# model_files {" ".join(args.model_files)}\n')
        f.write(f'# explainer_config_files {" ".join(args.explainer_config_files)}\n')
        f.write(f'exp_rec_data: {exp_rec_data}\n')
        rec_df.to_csv(f, index=None)

    with open(os.path.join(plots_path, 'rec_del_df.csv'), 'w') as f:
        f.write(f'# model_files {" ".join(args.model_files)}\n')
        f.write(f'# explainer_config_files {" ".join(args.explainer_config_files)}\n')
        f.write(f'exp_rec_data: {exp_rec_data}\n')
        rec_del_df.to_csv(f, index=None)

    with open(os.path.join(plots_path, 'incdisp.pkl'), 'wb') as f:
        pickle.dump(incdisp, f)

    with open(os.path.join(plots_path, 'del_edges.pkl'), 'wb') as f:
        pickle.dump(del_edges, f)

    with open(os.path.join(plots_path, 'all_batch_exps.pkl'), 'wb') as f:
        pickle.dump(all_batch_exps, f)

    with open(os.path.join(plots_path, 'train_datasets.pkl'), 'wb') as f:
        pickle.dump(train_datasets, f)

    with open(os.path.join(plots_path, 'datasets_train_inter_sizes.pkl'), 'wb') as f:
        pickle.dump(datasets_train_inter_sizes, f)

base_all_plots_path = os.path.join(script_path, os.pardir, f'dp_plots')
if os.path.exists(os.path.join(base_all_plots_path, 'graph_metrics_dfs.pkl')):
    with open(os.path.join(base_all_plots_path, 'graph_metrics_dfs.pkl'), 'rb') as f:
        graph_metrics_dfs = pickle.load(f)
else:
    graph_metrics_dfs = {}

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

for _dataset in unique_datasets:
    if _dataset not in graph_metrics_dfs and not args.overwrite_graph_metrics:
        graph_metrics_dfs[_dataset] = plot_utils.extract_graph_metrics_per_node(
            train_datasets[_dataset],
            remove_first_row_col=True,
            metrics="all"
        )

        last_user_id = train_datasets[_dataset].user_num - 2
        graph_mdf = graph_metrics_dfs[_dataset].set_index('Node')
        graph_mdf.loc[:last_user_id, 'Node Type'] = 'User'
        graph_mdf.loc[(last_user_id + 1):, 'Node Type'] = 'Item'
        graph_metrics_dfs[_dataset] = graph_mdf.reset_index()

    pg = sns.PairGrid(graph_metrics_dfs[_dataset], hue='Node Type')
    pg.map_diag(sns.histplot)
    pg.map_offdiag(sns.scatterplot)
    pg.add_legend()
    os.makedirs(os.path.join(base_all_plots_path, _dataset), exist_ok=True)
    pg.figure.savefig(os.path.join(base_all_plots_path, _dataset, f'{_dataset}_graph_metrics_pair_grid.png'))
    plt.close(pg.figure)

with open(os.path.join(base_all_plots_path, 'graph_metrics_dfs.pkl'), 'wb') as f:
    pickle.dump(graph_metrics_dfs, f)

rec_df["Policy"] = rec_df["Policy"].map(lambda p: {'NoPolicy': no_pert_col}.get(p, p))
rec_del_df["Policy"] = rec_del_df["Policy"].map(lambda p: {'NoPolicy': no_pert_col}.get(p, p))
if exp_rec_data != "test":
    test_df["Policy"] = test_df["Policy"].map(lambda p: {'NoPolicy': no_pert_col}.get(p, p))
    test_del_df["Policy"] = test_del_df["Policy"].map(lambda p: {'NoPolicy': no_pert_col}.get(p, p))

qnt_size = 100
ch_quantile = 95
hue_order = {'Gender': ['Males', 'Females'], 'Age': ['Younger', 'Older'], 'User Wide Zone': ['America', 'Other']}
unique_policies = sorted(rec_df['Policy'].unique(), key=lambda x: 0 if x == no_pert_col else len(x))
for df, del_df, exp_data_name in zip([test_df, rec_df], [test_del_df, rec_del_df], ["test", exp_rec_data]):
    if df is None or del_df is None:
        continue

    gm_metrics = next(graph_metrics_dfs.values().__iter__()).columns
    gm_metrics = gm_metrics[~gm_metrics.isin(['Node', '# Del Edges', 'Node Type'])]
    fig_gm = {}
    table_gm = []
    for gm in gm_metrics:
        fig_gm[gm] = {}
        for gm_s_attr in unique_sens_attrs:
            fig_gm[gm][gm_s_attr] = plt.figure(figsize=(40, 15), constrained_layout=True)
            fig_gm[gm][gm_s_attr].subfigures(len(unique_datasets), 1)
            for dset, subfig in zip(unique_datasets, fig_gm[gm][gm_s_attr].subfigs):
                subfig.suptitle(dataset_map[dset])
                subfig.subplots(1, len(unique_models), sharey=True)

    fig_heat = {}
    for heat_s_attr in unique_sens_attrs:
        fig_heat[heat_s_attr] = plt.figure(figsize=(15, 8), constrained_layout=True)
        fig_heat[heat_s_attr].subfigures(len(unique_datasets), 1)
        len_pol = len(unique_policies[1:])

        mosaic = list(map(str, range(len_pol * len(unique_models))))
        for dset, subfig in zip(unique_datasets, fig_heat[heat_s_attr].subfigs):
            subfig.suptitle(dataset_map[dset])
            subfig.subplot_mosaic(
                f"""
                {'.'.join([''.join(mosaic[i * len_pol:(i + 1) * len_pol]) for i in range(len(unique_models))])}
                """,
                sharey=True
            )

    fig_pca = {}
    for pca_s_attr in unique_sens_attrs:
        fig_pca[pca_s_attr] = plt.figure(figsize=(15, 22), constrained_layout=True)
        fig_pca[pca_s_attr].subfigures(len(unique_datasets), 1)
        for dset, subfig in zip(unique_datasets, fig_pca[pca_s_attr].subfigs):
            subfig.suptitle(dataset_map[dset])
            subfig.subplots(len(unique_models), len(unique_policies), sharex=True, sharey=True)

    topk_dist_data = {}
    for s_attr in unique_sens_attrs:
        topk_dist_data[s_attr] = []

    mds_gby = df.groupby(["Model", "Dataset", "Sens Attr"])
    del_mds_gby = del_df.groupby(["Model", "Dataset", "Sens Attr"])
    _metr_df_gby = df.groupby("Metric")
    _metr_del_df_gby = del_df.groupby("Metric")

    gm_mi_data = []
    gm_wasser_data = []
    gm_kl_data = []
    gm_dep_order = np.array(['Degree', 'Sparsity', 'Reachability'])
    m_dset_attr_groups = list(mds_gby.groups.keys())
    for groups_it, (_model, _dataset, _s_attr) in enumerate(m_dset_attr_groups):
        gm_data = {}
        len_pol = len(unique_policies[1:])
        for _policy in unique_policies:
            if _policy != no_pert_col:  # without perturbation that are no perturbed edges
                de = np.array(del_edges[(exp_data_name, _dataset, _model, _policy, _s_attr)])

                pert_train_dataset = utils.get_dataset_with_perturbed_edges(de.copy(), train_datasets[_dataset])
                train_adj = train_datasets[_dataset].inter_matrix(form='csr').astype(np.float32)[1:, 1:].todense()
                pert_train_adj = pert_train_dataset.inter_matrix(form='csr').astype(np.float32)[1:, 1:].todense()

                heat_ax = fig_heat[_s_attr.lower().replace(' ', '_')].subfigs[
                    unique_datasets.index(_dataset)].axes[
                    unique_models.index(_model) * len(unique_policies[1:]) + unique_policies[1:].index(_policy)
                ]

                kernel = tuple(map(lambda x: max(min(x, 40), round(x / 40)), train_adj.shape))
                del_heatmap = utils.rolling_window(train_adj - pert_train_adj, kernel, kernel, op="sum")

                sns.heatmap(del_heatmap, ax=heat_ax, cbar=False, xticklabels=False, yticklabels=False)
                heat_ax.set_xlabel(_policy)

                if len_pol % 2 == 0 and unique_policies[1:].index(_policy) == len_pol // 2 - 1:
                    heat_ax.set_title(_model, loc='right')
                elif len_pol % 2 == 1 and unique_policies[1:].index(_policy) == (len_pol - 1) // 2:
                    heat_ax.set_title(_model)

                # remove user and item id 0 padding
                de -= 1
                de[1] -= 1

                graph_mdf = graph_metrics_dfs[_dataset].copy(deep=True)
                # each edge is counted once for one node and once for the other (it is equal to a bincount)
                graph_mdf['# Del Edges'] = np.bincount(de.flatten(), minlength=len(graph_mdf))

                s_attr_df = df.set_index(['Dataset', 'Sens Attr', 'Model', 'Policy', 'Metric']).loc[
                    tuple([_dataset, _s_attr, _model, _policy] + ['NDCG'])  # metric not relevant here
                ].copy(deep=True)
                s_attr_df['user_id'] -= 1  # reindexing to zero
                sens_gmdf = graph_mdf.join(
                    s_attr_df.reset_index()[['user_id', 'Sens Attr', 'Demo Group']].set_index('user_id'),
                    on='Node'
                ).fillna('Item')

                dset_stats = os.path.join(script_path, os.pardir, f'dp_plots', 'datasets_stats.csv')
                if os.path.exists(dset_stats):
                    dsets_df = pd.read_csv(dset_stats, index_col=0)
                    if _dataset in dsets_df.columns:
                        graph_stats_dg = sens_gmdf.groupby(['Sens Attr', 'Demo Group']).mean().loc[_s_attr, gm_dep_order]
                        for gm in gm_dep_order:
                            gm_str = f"Avg. {gm} per user"
                            if gm_str not in dsets_df.index:
                                gm_stat_dg = ""
                            else:
                                gm_stat_dg = dsets_df.loc[gm_str, _dataset]
                                gm_stat_dg = "" if isinstance(gm_stat_dg, float) else gm_stat_dg  # avoids NaN
                                if _s_attr in gm_stat_dg:
                                    continue

                            gm_stat_dg += f" {_s_attr} " + '; '.join(graph_stats_dg[gm].to_frame().reset_index().apply(
                                lambda x: f"{x['Demo Group']} : {x[gm] * (100 if gm != 'Degree' else 1):.1f}\%", axis=1
                            ).values)
                            dsets_df.loc[gm_str, _dataset] = gm_stat_dg

                            dsets_df.to_csv(dset_stats)

                # https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0087357&type=printable
                # Paper that states how select number of neighbors, repetitions and usage of median
                for (gm_sens_attr, gm_dg), gm_dgdf in sens_gmdf.groupby(["Sens Attr", "Demo Group"]):
                    mi_res = np.zeros((args.iterations, len(gm_dep_order)), dtype=float)
                    wd_res = [np.inf] * len(gm_dep_order)
                    kl_res = [0] * len(gm_dep_order)
                    kl_eps = 1e-8  # avoids NaN

                    degree_scaled = sklearn.preprocessing.MinMaxScaler().fit_transform(
                        gm_dgdf.loc[:, ['Degree']].to_numpy()
                    ).squeeze()
                    n_del_edges_scaled = sklearn.preprocessing.MinMaxScaler().fit_transform(
                        gm_dgdf.loc[:, ['# Del Edges']].to_numpy()
                    ).squeeze()
                    for gm_i, gm in enumerate(gm_dep_order):
                        wd_data = gm_dgdf.loc[:, gm] if gm != 'Degree' else degree_scaled
                        wd_res[gm_i] = scipy.stats.wasserstein_distance(wd_data, n_del_edges_scaled)
                        kl_res[gm_i] = scipy.stats.entropy(
                            gm_dgdf.loc[:, '# Del Edges'] + kl_eps, gm_dgdf.loc[:, gm] + kl_eps, base=2
                        )

                    for mi_i in range(args.iterations):
                        mi_res[mi_i] = sk_feats.mutual_info_regression(
                            np.c_[degree_scaled, gm_dgdf.loc[:, gm_dep_order[gm_dep_order != 'Degree']].values],
                            n_del_edges_scaled,
                            n_neighbors=3
                        )
                    mi_res = np.median(mi_res, axis=0)
                    gm_mi_data.append([_dataset, _model, gm_sens_attr, _policy, gm_dg, *mi_res])
                    gm_wasser_data.append([_dataset, _model, gm_sens_attr, _policy, gm_dg, *wd_res])
                    gm_kl_data.append([_dataset, _model, gm_sens_attr, _policy, gm_dg, *kl_res])

                for graph_metric in graph_mdf.columns[~graph_mdf.columns.isin(['Node', '# Del Edges', 'Node Type'])]:
                    gm_policy_data = gm_data.setdefault(graph_metric, [])
                    nt_gmdfby = graph_mdf.groupby('Node Type')
                    for nt, node_type in enumerate(['User', 'Item']):
                        nt_gmdf = nt_gmdfby.get_group(node_type)
                        gm_values = nt_gmdf.sort_values(graph_metric)['# Del Edges']
                        gm_policy_data.extend([
                            (i, pct.sum() * ((-1) ** nt), node_type, _policy)
                            for i, pct in enumerate(np.array_split(gm_values, qnt_size))
                        ])

                del_mds_df = del_mds_gby.get_group((_model, _dataset, _s_attr))
                topk_dist_data[_s_attr.lower().replace(' ', '_')].append(
                    [_model, _dataset, _s_attr, del_mds_df]
                )

            pca_ax = fig_pca[_s_attr.lower().replace(' ', '_')].subfigs[unique_datasets.index(_dataset)].axes[
                unique_models.index(_model) * len(unique_policies) + unique_policies.index(_policy)
            ]
            try:
                pca_del_edges = del_edges[(exp_data_name, _dataset, _model, _policy, _s_attr)]
            except KeyError:
                pca_del_edges = del_edges[(exp_data_name, _dataset, _model, "NoPolicy", _s_attr)]
            train_pca, pert_train_pca = utils.get_decomposed_adj_matrix(
                pca_del_edges,
                train_datasets[_dataset]
            )

            f_idx = (train_datasets[_dataset].field2id_token[_s_attr.lower().replace(' ', '_')] == 'F').nonzero()[0][0]
            sens_data = train_datasets[_dataset].user_feat[_s_attr.lower().replace(' ', '_')].numpy()[1:]
            sens_data = np.array(
                [group_name_map[real_group_map[_s_attr.lower().replace(' ', '_')]['F' if idx == f_idx else 'M']]
                 for idx in sens_data]
            )

            if _policy != no_pert_col:
                changes = np.abs(train_pca - pert_train_pca)
                mask = changes > np.percentile(changes, ch_quantile)
                rel_chs, = np.bitwise_or.reduce(mask, axis=1).nonzero()
                irrel_chs, = np.bitwise_or.reduce(~mask, axis=1).nonzero()

                sns.scatterplot(x=pert_train_pca[irrel_chs, 0], y=pert_train_pca[irrel_chs, 1],
                                hue=sens_data[irrel_chs],
                                marker='o', ax=pca_ax, zorder=1, alpha=0.2, hue_order=hue_order[_s_attr], legend=False)
                sns.scatterplot(x=pert_train_pca[rel_chs, 0], y=pert_train_pca[rel_chs, 1], hue=sens_data[rel_chs],
                                marker='o', zorder=3, ax=pca_ax, hue_order=hue_order[_s_attr])

                for rel_tr_pca, rel_pert_tr_pca in zip(train_pca[rel_chs], pert_train_pca[rel_chs]):
                    pca_ax.annotate(
                        "",
                        xy=rel_pert_tr_pca, xytext=rel_tr_pca,
                        arrowprops=dict(arrowstyle='->', lw=0.5, connectionstyle='arc3', zorder=2)
                    )

                utils.legend_without_duplicate_labels(pca_ax)
            elif not pca_ax.collections:
                sns.scatterplot(x=train_pca[:, 0], y=train_pca[:, 1], hue=sens_data, marker='o', ax=pca_ax,
                                hue_order=hue_order[_s_attr])
                pca_ax.set_ylabel(_model)

            pca_ax.set_title(_policy)
            if groups_it == len(m_dset_attr_groups) - 1 and unique_policies.index(_policy) == len(unique_policies) - 1:
                pca_handles, pca_labels = pca_ax.get_legend_handles_labels()
                pca_ax.get_legend().remove()
                pca_legend = fig_pca[_s_attr.lower().replace(' ', '_')].legend(
                    pca_handles, pca_labels, loc="center right",
                    bbox_to_anchor=(1.1, 0.5),
                    bbox_transform=fig_pca[_s_attr.lower().replace(' ', '_')].transFigure
                )
                pca_legend.set_zorder(10)
            else:
                pca_ax.get_legend().remove()

        for graph_metric, gm_plot_data in gm_data.items():
            gm_ax = fig_gm[graph_metric][_s_attr.lower().replace(' ', '_')].subfigs[
                unique_datasets.index(_dataset)].axes[unique_models.index(_model)]
            gm_df = pd.DataFrame(gm_plot_data, columns=["x", graph_metric.title(), "Node Type", "Policy"])

            for (gm_nt, gm_pol), gm_nt_pol_df in gm_df.groupby(["Node Type", "Policy"]):
                # to mimic the sorting of the number of deleted edges over the graph metric, the "x" of the quantile
                # is repeated based on the number of deleted edges
                gm_distrib = gm_nt_pol_df["x"].repeat(gm_nt_pol_df[graph_metric.title()].abs())

                for stat, stat_f in zip(
                        ['Skewness', 'Kurtosis'],
                        [lambda x: scipy.stats.skew(x), lambda x: scipy.stats.kurtosis(x, fisher=False)]
                ):
                    table_gm.append((
                        _model,
                        _dataset,
                        _s_attr,
                        gm_pol,
                        graph_metric.title(),
                        gm_nt,
                        stat,
                        stat_f(gm_distrib)
                    ))

            sns.lineplot(x="x", y=graph_metric.title(), data=gm_df, ax=gm_ax,
                         hue="Node Type", hue_order=["User", "Item"],
                         style="Policy", style_order=unique_policies[1:])

            clean_quantile_ax(
                fig_gm[graph_metric][_s_attr.lower().replace(' ', '_')], gm_ax, _model, groups_it,
                len(m_dset_attr_groups)
            )

            if unique_models.index(_model) == 0:
                gm_ax.text(-0.4, 0.01, f"$-$ # Del Edges", transform=gm_ax.transAxes)
                gm_ax.text(-0.4, 0.99, f"# Del Edges", transform=gm_ax.transAxes)

    for gm in fig_gm:
        for gm_s_attr in fig_gm[gm]:
            fig_gm[gm][gm_s_attr].savefig(
                os.path.join(plots_path, f'{gm_s_attr}_graph_metric_({gm.title()})_plot_{exp_data_name}.png'),
                bbox_inches="tight"
            )

    for heat_s_attr in fig_heat:
        fig_heat[heat_s_attr].savefig(
            os.path.join(plots_path, f'{heat_s_attr}_heatmap_del_edges_{exp_data_name}.png'),
            bbox_inches="tight"
        )

    for pca_s_attr in fig_pca:
        fig_pca[pca_s_attr].savefig(
            os.path.join(plots_path, f'{pca_s_attr}_adj_matrix_decomposition_{exp_data_name}.png'),
            bbox_inches="tight"
        )
    plt.close("all")

    for gm_dep_data, dep_type in zip([gm_mi_data, gm_wasser_data, gm_kl_data], ["mi", "wd", "kl"]):
        gm_dep_df = pd.DataFrame(
            gm_dep_data,
            columns=["Dataset", "Model", "Sens Attr", "Policy", "Demo Group", *gm_dep_order]
        ).drop_duplicates(subset=["Dataset", "Model", "Sens Attr", "Policy", "Demo Group"])  # removes duplicated gms of item nodes
        gm_dep_df = gm_dep_df.melt(
            ['Dataset', 'Model', 'Sens Attr', 'Policy', 'Demo Group'],
            var_name="Graph Metric", value_name="Value"
        )
        # vmin, vmax = gm_heat_df["Value"].min(), gm_heat_df["Value"].max()
        # gm_heat_df["Policy"] = gm_heat_df["Policy"].map({'MonDel': 'MD', 'MonDel+DelCons': 'MD+DC'})
        # gm_heat_df = gm_heat_df.set_index('Sens Attr')
        # for gm_sens_attr in unique_sens_attrs:
        #     fig_heat_gm, axs_heat_gm = plt.subplots(2, len(unique_datasets), sharex=True, sharey=True, figsize=(40, 15), squeeze=False)
        #     for nt_i, nt in enumerate(["User", "Item"]):
        #         for _dset_j, _dset in enumerate(unique_datasets):
        #             gm_heat_pivot = gm_heat_df.loc[gm_sens_attr.title().replace('_', ' ')].pivot(
        #                 index=['Node Type', 'Dataset', 'Model', "Policy"],
        #                 columns="Graph Metric"
        #             ).loc[(nt, _dset)].droplevel(0, axis=1)
        #             sns.heatmap(
        #                 gm_heat_pivot, vmin=vmin, vmax=vmax, ax=axs_heat_gm[nt_i, _dset_j],
        #                 cbar=_dset == 'insurance', yticklabels=_dset == 'ml-100k'# 'ml-1m'
        #             )
        #             axs_heat_gm[nt_i, _dset_j].set_ylabel("")
        #             axs_heat_gm[nt_i, _dset_j].set_xlabel("")
        #             if _dset == 'ml-100k': # 'ml-1m':
        #                 plot_utils.hierarchical_labels(axs_heat_gm[nt_i, _dset_j], axis="y", offset=0.1)
        #     plt.tight_layout()
        #     fig_heat_gm.savefig(
        #         os.path.join(plots_path, f'mi_graph_metrics_heatmap_{gm_sens_attr}_{exp_data_name}.png'),
        #         bbox_inches="tight",
        #         pad_inches=0
        #     )
        #     plt.close()
        for gm_sa, gmd_sa_df in gm_dep_df.groupby('Sens Attr'):
            # for RQ3
            gmd_df = gmd_sa_df[(gmd_sa_df["Model"] != 'NGCF') & (gmd_sa_df["Policy"] == 'MonDel')]
            #####
            gm_dep_pivot = gmd_df[["Dataset", "Model", "Demo Group", "Graph Metric", "Value"]].pivot(
                index=['Model'],
                columns=['Dataset', 'Graph Metric', 'Demo Group']
            ).droplevel(0, axis=1).reindex(
                gm_dep_order, axis=1, level=1
            ).reindex(
                ["Item", "M", "F", "Y", "O", "America", "Other"], axis=1, level=2
            )
            gm_dep_pivot.columns = gm_dep_pivot.columns.map(lambda x: (*x[:2], group_name_map.get(x[2], x[2])))
            gm_dep_pivot.round(2).to_latex(
                os.path.join(plots_path, f"{dep_type}_table_graph_metrics_{gm_sa}_{exp_data_name}.tex"),
                multicolumn_format="c",
                escape=False
            )

    table_gm_df = pd.DataFrame(
        table_gm,
        columns=["Model", "Dataset", "Sens Attr", "Policy", "Graph Metric", "Node Type", "Statistic", "Value"]
    )
    table_gm_pivot = table_gm_df.pivot(
        index=['Sens Attr', 'Dataset', 'Model', 'Policy'],
        columns=['Graph Metric', 'Statistic', 'Node Type'],
        values='Value'
    )
    table_gm_final = table_gm_pivot.reindex(
        ['Degree', 'Reachability', 'Sparsity', 'Sharing Potentiality'], axis=1, level=0
    ).reindex(
        ["Skewness", "Kurtosis"], axis=1, level=1
    ).reindex(
        ['User', 'Item'], axis=1, level=2
    )
    table_gm_final.to_csv(os.path.join(plots_path, 'graph_metric_statistics.csv'))
    table_gm_final.round(2).to_latex(
        os.path.join(plots_path, 'graph_metric_statistics.tex'),
        multicolumn_format="c",
        escape=False
    )

    _metrics = list(_metr_df_gby.groups.keys())
    for metric in _metrics:
        metric_df = _metr_df_gby.get_group(metric)
        metric_del_df = _metr_del_df_gby.get_group(metric)

        table_bar_df = create_table_best_explanations(metric_df)

        if _metrics.index(metric) == 0:
            if 'Edit Dist' in df.columns:
                create_table_topk_list_change(metric_df, 'Edit Dist')
            if 'Set Dist' in df.columns:
                create_table_topk_list_change(metric_df, 'Set Dist')

        plot_df_data = []
        dp_df_data_per_group = []
        plot_del_df_data = []
        y_col = f"$\Delta$ {metric.upper()}"
        plot_columns = ["Model", "Dataset", "Policy", "Sens Attr", y_col]
        data_columns_per_group = ["Model", "Dataset", "Policy", "Sens Attr", "Demo Group", metric.upper()]
        plot_del_columns = ["Model", "Dataset", "Policy", "% Del Edges", "Sens Attr", y_col]
        palette = dict(zip(unique_policies, sns.color_palette("colorblind")))
        _m_dset_df_gby = metric_df.groupby(["Model", "Dataset", "Sens Attr"])
        _m_dset_pol_del_df = metric_del_df.groupby(["Model", "Dataset", "Policy", "Sens Attr"])

        fig_qnt = {}
        for qnt_s_attr in unique_sens_attrs:
            fig_qnt[qnt_s_attr] = plt.figure(figsize=(15, 15), constrained_layout=True)
            fig_qnt[qnt_s_attr].subfigures(len(unique_datasets), 1)
            for dset, subfig in zip(unique_datasets, fig_qnt[qnt_s_attr].subfigs):
                subfig.suptitle(dataset_map[dset])
                subfig.subplots(1, len(unique_models), sharey=True)

        m_dset_attr_list = list(_m_dset_df_gby.groups.keys())
        for it, (_model, _dataset, _s_attr) in enumerate(tqdm.tqdm(m_dset_attr_list, desc="Extracting DP across random samples")):
            sub_df = _m_dset_df_gby.get_group((_model, _dataset, _s_attr))

            qnt_data = []
            for _policy, sub_policy_df in sub_df.groupby("Policy"):
                for dg_i, (dg, dg_df) in enumerate(sub_policy_df.groupby("Demo Group")):
                    qnt_values = dg_df.sort_values("Value", ascending=False)["Value"]
                    qnt_data.extend([
                        (i, pct.mean() * (1 if dg_i == 0 else -1), group_name_map[dg], _policy)
                        for i, pct in enumerate(np.array_split(qnt_values, qnt_size))
                    ])

                dp_samples, dgs_order = utils.compute_DP_across_random_samples(
                    sub_policy_df, _s_attr, "Demo Group", _dataset, 'Value', batch_size=all_batch_exps[_dataset], iterations=args.iterations
                )

                plot_df_data.extend(list(zip(
                    [_model] * args.iterations,
                    [_dataset] * args.iterations,
                    [_policy] * args.iterations,
                    [_s_attr] * args.iterations,
                    dp_samples[:, -1]
                )))

                dp_df_data_per_group.extend(list(zip(
                    [_model] * args.iterations * len(dgs_order),
                    [_dataset] * args.iterations * len(dgs_order),
                    [_policy] * args.iterations * len(dgs_order),
                    [_s_attr] * args.iterations * len(dgs_order),
                    np.repeat(dgs_order, args.iterations),
                    np.concatenate(dp_samples[:, :-1].T)
                )))

                sub_del_df = _m_dset_pol_del_df.get_group((_model, _dataset, _policy, _s_attr))
                n_del_df_gby = sub_del_df.groupby("# Del Edges")
                sorted_del_edges = sub_del_df.sort_values("# Del Edges")["# Del Edges"].unique()
                for n_del in sorted_del_edges:
                    n_del_df = n_del_df_gby.get_group(n_del)
                    del_dp_samples, _ = utils.compute_DP_across_random_samples(
                        n_del_df, _s_attr, "Demo Group", _dataset, 'Value', batch_size=all_batch_exps[_dataset], iterations=args.iterations
                    )

                    plot_del_df_data.extend(list(zip(
                        [_model] * args.iterations,
                        [_dataset] * args.iterations,
                        [_policy] * args.iterations,
                        [n_del] * args.iterations,
                        [_s_attr] * args.iterations,
                        del_dp_samples[:, -1]
                    )))

            qnt_ax = fig_qnt[_s_attr.lower().replace(' ', '_')].subfigs[
                unique_datasets.index(_dataset)].axes[unique_models.index(_model)]
            qnt_df = pd.DataFrame(qnt_data, columns=["x", metric.upper(), "Demo Group", "Policy"])

            sns.lineplot(x="x", y=metric.upper(), data=qnt_df, ax=qnt_ax,
                         hue="Demo Group", hue_order=hue_order[_s_attr], style="Policy", style_order=unique_policies)

            if unique_models.index(_model) == 0:
                qnt_ax.text(-0.29, 0.01, f"$-${metric.upper()}", transform=qnt_ax.transAxes)
                qnt_ax.text(-0.29, 0.99, f"{metric.upper()}", transform=qnt_ax.transAxes)

            clean_quantile_ax(fig_qnt[_s_attr.lower().replace(' ', '_')], qnt_ax, _model, it, len(m_dset_attr_list))

        for qnt_s_attr in fig_qnt:
            fig_qnt[qnt_s_attr].savefig(
                os.path.join(plots_path, f'{qnt_s_attr}_percentile_plot_{exp_data_name}_{metric}.png'),
                bbox_inches="tight"
            )
        plt.close("all")

        hatches = ['//', 'o']
        fig_bar2, axs_bar2 = plt.subplots(len(unique_sens_attrs), len(unique_datasets), squeeze=False, figsize=(10, 6))
        axs_bar2 = [axs_bar2] if not isinstance(axs_bar2, np.ndarray) else axs_bar2
        table_bar_df.loc[table_bar_df["Status"] == "Before", "Policy"] = no_pert_col
        table_bar_df = table_bar_df.drop("Status", axis=1).rename(columns={'value': y_col})

        plot_del_df_line = pd.DataFrame(plot_del_df_data, columns=plot_del_columns)
        plot_df_box_bar = pd.DataFrame(plot_df_data, columns=plot_columns)
        dp_df_per_group = pd.DataFrame(dp_df_data_per_group, columns=data_columns_per_group)

        create_table_best_explanations_per_group(dp_df_per_group)

        plot_table_df_bar_gby = table_bar_df.groupby(["Sens Attr", "Dataset"])
        plot_del_df_line_gby = plot_del_df_line.groupby(["Sens Attr", "Dataset", "Model"])
        plot_df_box_bar_gby = plot_df_box_bar.groupby(["Sens Attr", "Dataset"])
        for s_attr_i, orig_sens_attr in enumerate(unique_sens_attrs):
            sens_attr = orig_sens_attr.title().replace('_', ' ')

            fig_pol_box, axs_pol_box = plt.subplots(1, len(unique_datasets), figsize=(10, 6))
            axs_pol_box = [axs_pol_box] if not isinstance(axs_pol_box, np.ndarray) else axs_pol_box

            fig_box, axs_box = plt.subplots(1, len(unique_datasets), figsize=(10, 6))
            axs_box = [axs_box] if not isinstance(axs_box, np.ndarray) else axs_box

            s_attr_dgs = []
            for i, (dset, ax_pol_box, ax_box) in enumerate(zip(unique_datasets, axs_pol_box, axs_box)):
                if (sens_attr, dset) in plot_df_box_bar_gby.groups:
                    dset_box_bar_sattr_df = plot_df_box_bar_gby.get_group((sens_attr, dset))
                    box = sns.boxplot(x="Model", y=y_col, data=dset_box_bar_sattr_df, hue="Policy", ax=ax_pol_box, palette=palette, showfliers=False)
                    # ax_bar.set_title(dataset_map[dset], pad=30)
                    ax_pol_box.set_xlabel("")

                    patches = [artist for artist in box.get_children() if isinstance(artist, mpatches.PathPatch)]
                    # sorted by x0 of the Bbox to ensure the order from left to right
                    patches = sorted(patches, key=lambda x: x.get_extents().x0)
                    stest_df = dset_box_bar_sattr_df.set_index(["Model", "Policy"])
                    plt.rcParams['hatch.linewidth'] = 0.4
                    for mod, mod_ptcs in zip(unique_models, np.array_split(patches, len(unique_models))):
                        mod_ptcs_text = [''] * len(mod_ptcs)
                        data_ptcs = [stest_df.loc[(mod, pol), y_col] for pol in sorted(unique_policies)]
                        bxp_stats = [mpl_cbook.boxplot_stats(d_ptc)[0] for d_ptc in data_ptcs]
                        height = max([_bxps['whishi'] for _bxps in bxp_stats])
                        yerr = 0
                        for (data1, ptc1), (data2, ptc2) in itertools.combinations(zip(data_ptcs, mod_ptcs), 2):
                            tt_stat, tt_pv = scipy.stats.ttest_rel(data1, data2)
                            if tt_pv < P_VALUE:
                                x, h = [], [height, height]
                                for _ptc in [ptc1, ptc2]:
                                    x.append(np.mean([_ptc.get_path().vertices[0][0], _ptc.get_path().vertices[1][0]]))
                                dh, barh = .05, .02
                                ax_pol_t = plot_utils.annotate_brackets(
                                    ax_pol_box, 0, 1, tt_pv, x, h, [yerr, yerr], dh, barh, fs=MEDIUM_SIZE
                                )
                                # yerr is updated such that next brackets are drawn over others
                                offset_norm = abs(height)
                                yerr += dh * offset_norm + barh * offset_norm + offset_norm * SMALL_SIZE * 5e-4

                        for _ptc, _bxp, bxp_hatch in zip(mod_ptcs, bxp_stats, ['X', '/', '.']):
                            x = np.mean([_ptc.get_path().vertices[0][0], _ptc.get_path().vertices[1][0]])
                            med = _bxp["med"]
                            med *= 1 + (0.05 * np.sign(med))
                            med_text = ax_pol_box.text(
                                x, med, f"{med:.3f}", fontsize=MEDIUM_SIZE, ha='center', color='w'
                            )
                            med_text.set_path_effects([
                                mpl_path_eff.Stroke(linewidth=1.5, foreground='k'),
                                mpl_path_eff.Normal(),
                            ])

                            _ptc.set_hatch(bxp_hatch)

                    left_xpatch, right_xpatch = zip(
                        *[(_ptc.get_path().vertices[0][0], _ptc.get_path().vertices[1][0]) for _ptc in patches]
                    )
                    left_min, right_max = min(left_xpatch), max(right_xpatch)
                    offset_xlim = 0.005 * (right_max - left_min)
                    ax_pol_box.set_xlim(left_min - offset_xlim, right_max + offset_xlim)

                    dset_box_bar_sattr_mondel_df = dset_box_bar_sattr_df[dset_box_bar_sattr_df["Policy"] != 'MonDel+DelCons']
                    sns.boxplot(x="Model", y=y_col, data=dset_box_bar_sattr_mondel_df, hue="Policy", ax=ax_box, palette=palette)
                    # ax_box.set_title(dataset_map[dset], pad=30)
                    ax_box.set_xlabel("")

                    if i == len(axs_pol_box) - 1 and dset == "ml-1m":
                        pol_box_handles, pol_box_labels = ax_pol_box.get_legend_handles_labels()
                        fig_pol_box.legend(pol_box_handles, pol_box_labels, loc='lower center', ncol=len(pol_box_labels), bbox_to_anchor=[0.5, 1.01])
                        box_handles, box_labels = ax_box.get_legend_handles_labels()
                        fig_box.legend(box_handles, box_labels, loc='upper center', ncol=len(box_labels))
                    ax_pol_box.get_legend().remove()
                    ax_box.get_legend().remove()

                    if dset != 'insurance':
                        ax_pol_box.tick_params('x', bottom=False, labelbottom=False)

                if (sens_attr, dset) in plot_table_df_bar_gby.groups:
                    plot_tdf_bar_sattr_df = plot_table_df_bar_gby.get_group((sens_attr, dset))
                    _ax = axs_bar2[s_attr_i, i]
                    dg_df_gby = plot_tdf_bar_sattr_df.groupby("Demo Group")
                    s_attr_dgs = [x for x in sorted(dg_df_gby.groups) if 'Delta' not in x]
                    for dg, hatch in zip(s_attr_dgs, hatches):
                        dg_df = dg_df_gby.get_group(dg)
                        sns.barplot(x="Model", y=y_col, data=dg_df, hue="Policy", ax=_ax, palette=palette, edgecolor='black', alpha=0.6, hatch=hatch)
                        _ax.get_legend().remove()
                        _ax.set_title(dataset_map[dset])

            create_fig_bar2_legend(fig_bar2, palette, hatches, s_attr_dgs, loc="upper left")

            fig_pol_box.tight_layout(pad=0)
            fig_pol_box.savefig(
                os.path.join(plots_path, f"{sens_attr}_all_policies_boxplot_{exp_data_name}_{metric}_DP_random_samples.png"),
                bbox_inches='tight', pad_inches=0, dpi=250
            )

            fig_box.tight_layout()
            fig_box.savefig(os.path.join(plots_path, f"{sens_attr}_boxplot_MonDel_{exp_data_name}_{metric}_DP_random_samples.png"))

        fig_bar2.tight_layout()
        fig_bar2.savefig(os.path.join(plots_path, f"overlapping_barplot_{exp_data_name}_{metric}_DP_random_samples.png"))
        plt.close("all")

        # need to be done here due to maximum number of axes opened simultaneously
        for s_attr_i, orig_sens_attr in enumerate(unique_sens_attrs):
            sens_attr = orig_sens_attr.title().replace('_', ' ')
            fig_line = plt.figure(figsize=(15, 15), constrained_layout=True)
            subfigs = fig_line.subfigures(len(unique_datasets), 1)
            subfigs = [subfigs] if not isinstance(subfigs, np.ndarray) else subfigs

            for i, dset in enumerate(unique_datasets):
                subfigs[i].suptitle(dset.upper())
                axs_line = subfigs[i].subplots(1, len(unique_models))
                axs_line = [axs_line] if not isinstance(axs_line, np.ndarray) else axs_line
                for m, ax_line in zip(unique_models, axs_line):
                    if (sens_attr, dset, m) in plot_del_df_line_gby.groups:
                        dset_model_line_df = plot_del_df_line_gby.get_group((sens_attr, dset, m))
                        sns.lineplot(x="% Del Edges", y=y_col, data=dset_model_line_df, hue="Policy", ax=ax_line, palette=palette, ci=None)
                        ax_line.set_title(m.upper() + (f'+{incdisp[(dset, m)]}' if incdisp[(dset, m)] else ''))
                        ax_line.xaxis.set_major_formatter(mpl_tick.FuncFormatter(lambda x, pos: f"{x / datasets_train_inter_sizes[dset] * 100:.2f}%"))

            fig_line.suptitle(sens_attr.title().replace('_', ' '))
            fig_line.savefig(os.path.join(plots_path, f"{sens_attr}_lineplot_{exp_data_name}_{metric}_DP_random_samples.png"))

        plt.close("all")
