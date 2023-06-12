# %%
import os
import gc
import pickle
import inspect
import argparse
import tempfile
import itertools

import tqdm
import dcor
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

import biga.utils as utils
import biga.evaluation as eval_utils


# %%
def get_plots_path(datasets_names, model_names, old=False):
    plots_path = os.path.join(
        script_path,
        os.pardir,
        f'dp_plots' if old else 'new_dp_plots',
        datasets_names,
        model_names
    )

    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    return plots_path


def gini(data):
    data = np.sort(data)
    n = len(data)
    idxs = np.arange(1, n + 1)
    return (n + 1 - 2 * ((n + 1 - idxs) * data).sum() / data.sum()) / n


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
            user_df.set_index(uid_field).loc[test_uid, sens_attr].map(attr_map.__getitem__).map(
                real_group_map[sens_attr]),
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
    uid_list_test = next(exp_test_df.groupby('epoch').__iter__())[1].user_id if exp_test_df is not None else None
    uid_list_rec = next(exp_rec_df.groupby('epoch').__iter__())[1].user_id

    result_test_df_data, result_rec_df_data = [], []
    if test_result_all_data is not None:
        for epoch_n_del, res in test_result_all_data[model_dp_s].items():
            res_epoch, n_del = epoch_n_del.split('_')
            result_test_df_data.extend(list(zip(
                [int(res_epoch)] * len(uid_list_test),
                [int(n_del)] * len(uid_list_test),
                uid_list_test,
                res[metric][:, -1],
                [metric.upper()] * len(uid_list_test)
            )))
    for epoch_n_del, res in rec_result_all_data[model_dp_s].items():
        res_epoch, n_del = epoch_n_del.split('_')
        result_rec_df_data.extend(list(zip(
            [int(res_epoch)] * len(uid_list_rec),
            [int(n_del)] * len(uid_list_rec),
            uid_list_rec,
            res[metric][:, -1],
            [metric.upper()] * len(uid_list_rec)
        )))

    if exp_test_df is not None:
        exp_test_df = exp_test_df.join(
            pd.DataFrame(
                result_test_df_data, columns=['epoch', 'n_del_edges', 'user_id', 'Value', 'Metric']
            ).set_index(['epoch', 'user_id']),
            on=['epoch', 'user_id']
        ).join(user_df.set_index(uid_field), on='user_id')
        exp_test_df[sens_attr] = exp_test_df[sens_attr].map(attr_map.__getitem__).map(real_group_map[sens_attr])
    exp_rec_df = exp_rec_df.join(
        pd.DataFrame(
            result_rec_df_data, columns=['epoch', 'n_del_edges', 'user_id', 'Value', 'Metric']
        ).set_index(['epoch', 'n_del_edges', 'user_id']),
        on=['epoch', 'n_del_edges', 'user_id']
    ).join(user_df.set_index(uid_field), on='user_id')
    exp_rec_df[sens_attr] = exp_rec_df[sens_attr].map(attr_map.__getitem__).map(real_group_map[sens_attr])

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


def compute_graph_metrics_analysis_tables_data(_sens_gmdf,
                                               _gm_analysis_tables_data: dict,
                                               gm_rel_info,
                                               _iterations=100,
                                               kl_eps=1e-8):  # avoids NaN
    # https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0087357&type=printable
    # Paper that states how select number of neighbors, repetitions and usage of median
    # we regroup again by sensitive attribute because `Item` was added as label
    for gm_sens_attr, gm_sattr_df in _sens_gmdf.groupby("Sens Attr"):
        gm_sattr_df_norm = gm_sattr_df.copy(deep=True)
        gm_sattr_df_norm['Degree'] = sklearn.preprocessing.MinMaxScaler().fit_transform(
            gm_sattr_df_norm.loc[:, ['Degree']].to_numpy()
        ).squeeze()
        gm_sattr_df_norm['Del Edges Count Scaled'] = sklearn.preprocessing.MinMaxScaler().fit_transform(
            gm_sattr_df_norm.loc[:, ['Del Edges Count']].to_numpy()
        ).squeeze()
        for gm_dg, gm_dgdf in gm_sattr_df_norm.groupby("Demo Group"):
            mi_res = np.zeros((_iterations, len(gm_dep_order)), dtype=float)
            wd_res = [np.inf] * len(gm_dep_order)
            kl_res = [0] * len(gm_dep_order)
            dcor_res = [None] * len(gm_dep_order)
            dcor_pval = [None] * len(gm_dep_order)
            gm_del_dist = [None] * len(gm_dep_order)

            n_del_edges_scaled = gm_dgdf.loc[:, 'Del Edges Count Scaled'].to_numpy()
            for gm_i, gm in enumerate(gm_dep_order):
                gm_dg_data = gm_dgdf.loc[:, gm].to_numpy()
                gm_dg_data = gm_dg_data.astype(n_del_edges_scaled.dtype)

                if "wd" in _gm_analysis_tables_data:
                    wd_res[gm_i] = scipy.stats.wasserstein_distance(gm_dg_data, n_del_edges_scaled)
                if "kl" in _gm_analysis_tables_data:
                    kl_res[gm_i] = scipy.stats.entropy(
                        n_del_edges_scaled + kl_eps, gm_dg_data + kl_eps, base=2
                    )
                # if "dcor" in _gm_analysis_tables_data:
                #     dcor_res[gm_i] = dcor.distance_correlation(gm_dg_data, n_del_edges_scaled)
                #     dcor_pval[gm_i] = dcor.independence.distance_covariance_test(
                #         gm_dg_data, n_del_edges_scaled, num_resamples=10
                #     ).pvalue

                if "del_dist" in _gm_analysis_tables_data:
                    quartiles = np.array_split(gm_dgdf.sort_values(gm), 20)
                    gm_del_dist[gm_i] = [q['Del Edges Count'].sum() / de_count for q in quartiles]

            if "mi" in _gm_analysis_tables_data:
                for mi_i in range(_iterations):
                    mi_res[mi_i] = sk_feats.mutual_info_regression(
                        gm_dgdf.loc[:, gm_dep_order].values,
                        n_del_edges_scaled,
                        n_neighbors=3
                    )
                mi_res = np.median(mi_res, axis=0)

            gm_info = [*gm_rel_info, gm_sens_attr, gm_dg]

            if "mi" in _gm_analysis_tables_data:
                _gm_analysis_tables_data["mi"].append([*gm_info, *mi_res])
            if "wd" in _gm_analysis_tables_data:
                _gm_analysis_tables_data["wd"].append([*gm_info, *wd_res])
            if "kl" in _gm_analysis_tables_data:
                _gm_analysis_tables_data["kl"].append([*gm_info, *kl_res])
            # if "dcor" in _gm_analysis_tables_data:
            #     _gm_analysis_tables_data["dcor"].append([*gm_info, *dcor_res])
            #     _gm_analysis_tables_data["dcor_pval"].append([*gm_info, *dcor_pval])
            if "del_dist" in _gm_analysis_tables_data:
                _gm_analysis_tables_data["del_dist"].append([*gm_info, *gm_del_dist])


def create_del_table_best_explanations_per_group(_metric_df):
    nop_mask = _metric_df["Policy"] == no_pert_col
    metr_df_nop = _metric_df[nop_mask].copy()
    metr_df_p = _metric_df[~nop_mask].copy()

    metr_df_nop = metr_df_nop[["Dataset", "Model", "Policy", "Sens Attr", "Demo Group", metric]]
    metr_df_p = metr_df_p[["Dataset", "Model", "Policy", "Sens Attr", "Demo Group", metric]]

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
            table_out_bar_df.to_csv(
                os.path.join(plots_path, f"total_del_table_{level_attr}_{exp_data_name}_{metric}_best_epoch.csv"))
            final_table_df.columns.names = [''] * len(final_table_df.columns.names)
            final_table_df.index = final_table_df.index.map(lambda x: (dataset_map[x[0]], *x[1:]))
            final_table_df.replace('%', '\%', regex=True).to_latex(
                os.path.join(plots_path, f"total_del_table_{level_attr}_{exp_data_name}_{metric}_best_epoch.tex"),
                multicolumn_format="c",
                escape=False
            )


def create_table_best_explanations_per_group(_metric_df):
    nop_mask = _metric_df["Policy"] == no_pert_col
    metr_df_nop = _metric_df[nop_mask].copy()
    metr_df_p = _metric_df[~nop_mask].copy()

    metr_df_nop = metr_df_nop[["Dataset", "Model", "Policy", "Sens Attr", "Demo Group", metric]]
    metr_df_p = metr_df_p[["Dataset", "Model", "Policy", "Sens Attr", "Demo Group", metric]]

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
            table_out_bar_df.to_csv(
                os.path.join(plots_path, f"total_table_{level_attr}_{exp_data_name}_{metric}_best_epoch.csv"))
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
parser.add_argument('--iterations', default=100, type=int)
parser.add_argument('--overwrite_plot_data', '--opd', action="store_true")
parser.add_argument('--overwrite_extracted_data', '--oed', action="store_true")
parser.add_argument('--overwrite_graph_metrics', '--ogm', action="store_true")
parser.add_argument('--overwrite_del_dp_plot_data', '--odppd', action="store_true")
parser.add_argument('--overwrite_casper_explanations', '--oce', action="store_true")
parser.add_argument('--overwrite_overlay_explanations', '--ooe', action="store_true")
parser.add_argument('--utility_metrics', '--um', nargs='+', default=None)
parser.add_argument('--add_plot_table', '--apt', action="store_true")
parser.add_argument('--plot_only_graph_metrics', '--pogm', action="store_true")
parser.add_argument('--extract_only_graph_metrics_mm', '--eogmmm', action="store_true")  # mm => memory management
parser.add_argument('--graph_metrics_analysis_tables', '--gmat', nargs='+', default=None)
parser.add_argument('--casper_explanations', '--ce', default=None, nargs='+', type=str)
parser.add_argument('--overlay_explanations', '--laye', default=None, action="store_true")
parser.add_argument('--overlay_threshold', '--layth', default=0.8)
parser.add_argument('--overlay_min_length', '--layml', default=None)

args = parser.parse_args()

assert len(args.model_files) == len(args.explainer_config_files), \
    "Pass the same number of perturbed model files and configuration files to be loaded"

args.graph_metrics_analysis_tables = args.graph_metrics_analysis_tables or ["mi", "wd", "kl", "dcor", "del_dist"]

script_path = os.path.abspath(os.path.dirname(inspect.getsourcefile(lambda: 0)))
script_path = os.path.join(script_path, 'src') if 'src' not in script_path else script_path

print(args)

mondel_pol = 'GNNUERS'
delcons_pol = 'GNNUERS+CN'
random_pol = 'RND-P'
casper_pol = 'CASPER'
no_pert_col = "NP"  # NoPerturbation

policy_order_base = [mondel_pol, delcons_pol, random_pol, casper_pol, no_pert_col]

palette = dict(zip(policy_order_base, sns.color_palette("colorblind")))
pol_hatches = dict(zip(policy_order_base, ['X', '.', '/', 'O', '*']))

policy_map = {
    'force_removed_edges': mondel_pol,  # Monotonic Deletions
    'group_deletion_constraint': delcons_pol,  # Deletion Constraint
    'random_perturbation': random_pol  # Random Perturbation
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

plot_dataset_map = {
    "ml-1m": "ML-1M",
    "lastfm-1k": "LFM-1K",
    "insurance": "INS",
    "tafeng": "FENG"
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

del_dist_map = ["LOW", "MEDIUM-LOW", "MEDIUM-HIGH", "HIGH"]
del_dist_map2 = ["Q1", "Q2", "Q3", "Q4"]

P_VALUE = 0.05

old_explanations = False
exp_epochs, config_ids, datasets_list, models_list, sens_attrs = [], [], [], [], []
for exp_config_file in args.explainer_config_files:
    if 'experiments' in exp_config_file:
        if old_explanations:
            raise ValueError("old and new explanations cannot be mixed")
        _, dset, model, _, _, s_attr, eps, cid, _ = exp_config_file.split('dp_explanations')[1].split(os.sep)
    else:
        old_explanations = True
        _, dset, model, _, s_attr, eps, cid, _ = exp_config_file.split('dp_ndcg_explanations')[1].split(os.sep)
    datasets_list.append(dset)
    models_list.append(model)
    sens_attrs.append(s_attr)
    exp_epochs.append(eps.replace('epochs_', ''))
    config_ids.append(cid)

unique_datasets, unique_models, unique_sens_attrs = \
    np.unique(datasets_list).tolist(), np.unique(models_list).tolist(), np.unique(sens_attrs).tolist()

plots_path = os.path.join(
    get_plots_path('_'.join(unique_datasets), '_'.join(unique_models), old=old_explanations),
    '_'.join(exp_epochs),
    '_'.join(config_ids),
    '_'.join(sens_attrs)
)

if not os.path.exists(plots_path):
    os.makedirs(plots_path)

if os.path.exists(os.path.join(plots_path, 'rec_df.csv')) and \
        os.path.exists(os.path.join(plots_path, 'incdisp.pkl')) and not args.overwrite_plot_data:
    test_rows, rec_rows = 2, 3

    if not args.extract_only_graph_metrics_mm:
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

        with open(os.path.join(plots_path, 'all_del_edges.pkl'), 'rb') as f:
            all_del_edges = pickle.load(f)

        with open(os.path.join(plots_path, 'all_batch_exps.pkl'), 'rb') as f:
            all_batch_exps = pickle.load(f)

        with open(os.path.join(plots_path, 'datasets_train_inter_sizes.pkl'), 'rb') as f:
            datasets_train_inter_sizes = pickle.load(f)

    with open(os.path.join(plots_path, 'train_datasets.pkl'), 'rb') as f:
        train_datasets = pickle.load(f)

else:
    # %%
    incdisp = {}
    del_edges = {}
    all_del_edges = {}
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
        for p, pm in policy_map.items():
            if config['explainer_policies'].get(p, False):  # if next policies are true `policy` is overwritten
                policy = pm
        policy = random_pol if random_pol in policy else policy
        incdisp[(dataset_name, model_name)] = 'IncDisp' if config['explainer_policies'][
            'increase_disparity'] else ''  # Increase Disparity
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

        config['metrics'] = args.utility_metrics or ['Recall', 'NDCG', 'MRR', 'Hit']
        evaluator = Evaluator(config)

        metrics = evaluator.metrics
        model_dp_s = f'{model_name}+FairDP'

        exp_paths = {model_dp_s: os.path.dirname(exp_config_file)}

        del model
        del dataset

        additional_best_cols = ['test_cf_dist', 'rec_cf_dist']
        if exp_rec_data != "test":
            if old_explanations:
                best_test_exp_df, best_test_exp_result = eval_utils.old_extract_best_metrics(
                    exp_paths,
                    'auto',
                    evaluator,
                    test_data.dataset,
                    config=config,
                    additional_cols=additional_best_cols[:1]
                )
            else:
                best_test_exp_df, best_test_exp_result = eval_utils.extract_best_metrics(
                    exp_paths,
                    'auto',
                    evaluator,
                    test_data.dataset,
                    config=config,
                    additional_cols=additional_best_cols[:1]
                )
        else:
            best_test_exp_df, best_test_exp_result = None, None
        if old_explanations:
            best_rec_exp_df, best_rec_exp_result = eval_utils.old_extract_best_metrics(
                exp_paths,
                'auto',
                evaluator,
                rec_data.dataset,
                config=config,
                additional_cols=additional_best_cols[1:]
            )
        else:
            best_rec_exp_df, best_rec_exp_result = eval_utils.extract_best_metrics(
                exp_paths,
                'auto',
                evaluator,
                rec_data.dataset,
                config=config,
                additional_cols=additional_best_cols[1:]
            )

        test_uid = best_test_exp_df[model_dp_s]['user_id'].to_numpy() if best_test_exp_df is not None else None
        rec_uid = best_rec_exp_df[model_dp_s]['user_id'].to_numpy()

        if exp_rec_data != "test":
            if old_explanations:
                all_exp_test_dfs, test_result_all_data, _, _ = eval_utils.old_extract_all_exp_metrics_data(
                    exp_paths,
                    train_data,
                    test_data.dataset,
                    evaluator,
                    sens_attr,
                    rec=False,
                    overwrite=args.overwrite_extracted_data
                )
            else:
                all_exp_test_dfs, test_result_all_data, _, _ = eval_utils.extract_all_exp_metrics_data(
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

        if old_explanations:
            all_exp_rec_dfs, rec_result_all_data, _, _ = eval_utils.old_extract_all_exp_metrics_data(
                exp_paths,
                train_data,
                rec_data.dataset,
                evaluator,
                sens_attr,
                rec=True,
                overwrite=args.overwrite_extracted_data
            )
        else:
            all_exp_rec_dfs, rec_result_all_data, _, _ = eval_utils.extract_all_exp_metrics_data(
                exp_paths,
                train_data,
                rec_data.dataset,
                evaluator,
                sens_attr,
                rec=True,
                overwrite=args.overwrite_extracted_data
            )

        for all_exp_data_name, all_exp_del_edges in zip(["test", exp_rec_data], [all_exp_test_dfs, all_exp_rec_dfs]):
            if all_exp_del_edges is not None:
                for all_epoch, all_n_del_edges_df in all_exp_del_edges[model_dp_s].groupby('epoch'):
                    all_del_edges[
                        (all_exp_data_name, dataset_name, model_name, policy,
                         sens_attr.title().replace('_', ' '), all_epoch)
                    ] = all_n_del_edges_df['del_edges'].iloc[0].tolist()

        for metric in metrics:
            # group_edge_del = update_plot_data(
            #     test_df_data,
            #     rec_df_data,
            #     additional_best_cols=additional_best_cols + ["set_dist"]
            # )
            update_plot_del_data(test_del_df_data, rec_del_df_data)

        no_policies.add((dataset_name, model_name, sens_attr))

        gc.collect()

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

    with open(os.path.join(plots_path, 'all_del_edges.pkl'), 'wb') as f:
        pickle.dump(all_del_edges, f)

    with open(os.path.join(plots_path, 'all_batch_exps.pkl'), 'wb') as f:
        pickle.dump(all_batch_exps, f)

    with open(os.path.join(plots_path, 'train_datasets.pkl'), 'wb') as f:
        pickle.dump(train_datasets, f)

    with open(os.path.join(plots_path, 'datasets_train_inter_sizes.pkl'), 'wb') as f:
        pickle.dump(datasets_train_inter_sizes, f)

base_all_plots_path = os.path.join(script_path, os.pardir, 'dp_plots' if old_explanations else 'new_dp_plots')
if os.path.exists(os.path.join(base_all_plots_path, 'graph_metrics_dfs.pkl')):
    with open(os.path.join(base_all_plots_path, 'graph_metrics_dfs.pkl'), 'rb') as f:
        graph_metrics_dfs = pickle.load(f)
else:
    graph_metrics_dfs = {}

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

gm_metrics_base = ['Degree', 'Density', 'Reachability']
gm_dep_order = np.array(gm_metrics_base)
for _dataset in unique_datasets:
    if _dataset not in graph_metrics_dfs or args.overwrite_graph_metrics:
        graph_metrics_dfs[_dataset] = eval_utils.extract_graph_metrics_per_node(
            train_datasets[_dataset],
            remove_first_row_col=True,
            metrics=gm_metrics_base  # "all"
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
    del pg

with open(os.path.join(base_all_plots_path, 'graph_metrics_dfs.pkl'), 'wb') as f:
    pickle.dump(graph_metrics_dfs, f)

if args.extract_only_graph_metrics_mm:
    print("Graph metrics extracted and saved")
    exit()


if args.overlay_explanations:
    args.overlay_explanations = []
    for dset in unique_datasets:
        for s_attr in unique_sens_attrs:
            ol_path = os.path.join(plots_path, f"{dset}_{sens_attr}.npy")
            if not os.path.exists(os.path.join(plots_path, f"{dset}_{s_attr}.npy")):
                overlay_edges = eval_utils.overlay_del_edges(
                    train_datasets[dset],
                    s_attr,
                    th=args.overlay_threshold,
                    min_length=args.overlay_min_length
                )
                np.save(ol_path, overlay_edges)

            args.overlay_explanations.extend([(dset, sens_attr), ol_path])

for extra_arg, extra_ow_arg, extra_df_path, extra_pol in zip(
    [args.casper_explanations, args.overlay_explanations],
    [args.overwrite_casper_explanations, args.overwrite_overlay_explanations],
    ['casper_exp_df.csv', 'overlay_exp_df.csv'],
    [casper_pol, overlay_pol]
):
    extra_pert_df = None
    extra_df_path = os.path.join(plots_path, extra_df_path)
    if os.path.exists(extra_df_path) and not extra_ow_arg:
        extra_pert_df = pd.read_csv(extra_df_path)
    elif extra_arg:
        extra_exp_paths = dict(zip(extra_arg[::2], extra_arg[1::2]))
        extra_pert_df = eval_utils.extract_metrics_from_perturbed_edges(
            extra_exp_paths,
            models=["NGCF", "GCMC", "LightGCN"],
            metrics=rec_del_df['Metric'].unique(),
            models_path='saved',
            policy_name=extra_pol
        )
        mapped_dg_dfs = []
        for extra_s_attr, extra_s_attr_df in extra_pert_df.groupby('Sens Attr'):
            if extra_s_attr.lower() in real_group_map:
                _extra_sadf = extra_s_attr_df.copy(deep=True)
                _extra_sadf["Demo Group"] = _extra_sadf["Demo Group"].map(real_group_map[extra_s_attr.lower()]).to_numpy()
                mapped_dg_dfs.append(_extra_sadf)

        extra_pert_df = pd.concat(mapped_dg_dfs, ignore_index=True)
        extra_pert_df.to_csv(extra_df_path, index=False)

    if extra_pert_df is not None:
        rec_del_df = pd.concat([rec_del_df, extra_pert_df], ignore_index=True)


rec_df["Policy"] = rec_df["Policy"].map(lambda p: {'NoPolicy': no_pert_col}.get(p, p))
rec_del_df["Policy"] = rec_del_df["Policy"].map(lambda p: {'NoPolicy': no_pert_col}.get(p, p))
if exp_rec_data != "test":
    test_df["Policy"] = test_df["Policy"].map(lambda p: {'NoPolicy': no_pert_col}.get(p, p))
    test_del_df["Policy"] = test_del_df["Policy"].map(lambda p: {'NoPolicy': no_pert_col}.get(p, p))

old_new_policy_map = {
    'MonDel': mondel_pol,
    'MonDel+DelCons': delcons_pol,
    'MonDel+Random': random_pol,
    'NoPerturbation': no_pert_col
}

rec_df["Policy"] = rec_df["Policy"].map(lambda p: old_new_policy_map.get(p, p))
rec_del_df["Policy"] = rec_del_df["Policy"].map(lambda p: old_new_policy_map.get(p, p))
if exp_rec_data != "test":
    test_df["Policy"] = test_df["Policy"].map(lambda p: old_new_policy_map.get(p, p))
    test_del_df["Policy"] = test_del_df["Policy"].map(lambda p: old_new_policy_map.get(p, p))

qnt_size = 100
ch_quantile = 95
hue_order = {'Gender': ['Males', 'Females'], 'Age': ['Younger', 'Older'], 'User Wide Zone': ['America', 'Other']}

if not rec_df.empty:
    unique_policies = sorted(rec_df['Policy'].unique(), key=lambda x: 0 if x == no_pert_col else len(x))
else:
    unique_policies = sorted(rec_del_df['Policy'].unique(), key=lambda x: 0 if x == no_pert_col else len(x))

policy_order = [p for p in policy_order_base if p in unique_policies]

# from pympler import asizeof

# def blabla(_locals_items):
#     for kk, vv in _locals_items:
#         try:
#             size = asizeof.asizeof(vv)/1024/1024
#             if size > 100:
#                 print(kk, f"{size:.2f} MB")
#         except KeyError:
#             print(kk, "ERROR")

for df, del_df, exp_data_name in zip([test_df, rec_df], [test_del_df, rec_del_df], ["test", exp_rec_data]):
    if df is None or del_df is None:
        continue

    fairest_del_edges_conf = {}
    if os.path.exists(os.path.join(plots_path, 'fairest_del_edges_conf.pkl')) and not args.overwrite_del_dp_plot_data:
        with open(os.path.join(plots_path, 'fairest_del_edges_conf.pkl'), 'rb') as f:
            fairest_del_edges_conf = pickle.load(f)

    _metr_del_df_gby = del_df.groupby("Metric")

    index_cols = ['Dataset', 'Model', 'Policy', 'Sens Attr', 'Epoch']
    _metrics = args.utility_metrics or list(_metr_del_df_gby.groups.keys())
    for metric in _metrics:
        if args.plot_only_graph_metrics:
            break

        metric_del_df = _metr_del_df_gby.get_group(metric)
        plot_del_df_data = []
        plot_del_df_data_per_group = []
        y_col = f"$\Delta$ {metric.upper()}"
        plot_columns = ["Model", "Dataset", "Policy", "Sens Attr", y_col]
        data_columns_per_group = ["Model", "Dataset", "Policy", "Sens Attr", "Demo Group", metric.upper()]
        plot_del_columns = ["Model", "Dataset", "Policy", "Epoch", "# Del Edges", "Sens Attr", y_col]
        del_data_columns_per_group = ["Model", "Dataset", "Policy", "Epoch", "# Del Edges", "Sens Attr", "Demo Group", metric.upper()]

        palette = {k: v for k, v in palette.items() if k in policy_order}
        m_dset_del_gby = metric_del_df.groupby(["Model", "Dataset", "Sens Attr"])

        m_dset_attr_list = list(m_dset_del_gby.groups.keys())
        if not (
                os.path.exists(os.path.join(plots_path, 'fairest_del_edges_conf.pkl')) and
                os.path.exists(os.path.join(plots_path, f'{metric.lower()}_fairest_dp_df.csv')) and
                os.path.exists(os.path.join(plots_path, f'{metric.lower()}_fairest_dp_df_per_group.csv'))
           ) or args.overwrite_del_dp_plot_data:
            if not (
                    os.path.exists(os.path.join(plots_path, f'{metric.lower()}_plot_del_df_data.pkl')) and
                    os.path.exists(os.path.join(plots_path, f'{metric.lower()}_plot_del_df_data_per_group.pkl'))
               ) or args.overwrite_del_dp_plot_data:
                for it, (_model, _dataset, _s_attr) in enumerate(tqdm.tqdm(
                    m_dset_attr_list, desc="Extracting DP across random samples"
                )):
                    sub_del_df = m_dset_del_gby.get_group((_model, _dataset, _s_attr))

                    qnt_data = []
                    for _policy, sub_del_policy_df in sub_del_df.groupby("Policy"):

                        # sub_del_df = _m_dset_pol_del_df.get_group((_model, _dataset, _policy, _s_attr))
                        del_epoch_df_gby = sub_del_policy_df.groupby("Epoch")
                        sorted_epochs = sub_del_policy_df.sort_values("Epoch")["Epoch"].unique()
                        for del_epoch in sorted_epochs:
                            del_epoch_df = del_epoch_df_gby.get_group(del_epoch)
                            n_del = del_epoch_df.loc[del_epoch_df.index[0], '# Del Edges']
                            del_dp_samples, dgs_order = eval_utils.compute_DP_across_random_samples(
                                del_epoch_df, _s_attr, "Demo Group", _dataset, 'Value', batch_size=all_batch_exps[_dataset],
                                iterations=args.iterations
                            )

                            plot_del_df_data.extend(list(zip(
                                [_model] * args.iterations,
                                [_dataset] * args.iterations,
                                [_policy] * args.iterations,
                                [del_epoch] * args.iterations,
                                [n_del] * args.iterations,
                                [_s_attr] * args.iterations,
                                del_dp_samples[:, -1]
                            )))

                            plot_del_df_data_per_group.extend(list(zip(
                                [_model] * args.iterations * len(dgs_order),
                                [_dataset] * args.iterations * len(dgs_order),
                                [_policy] * args.iterations * len(dgs_order),
                                [del_epoch] * args.iterations * len(dgs_order),
                                [n_del] * args.iterations * len(dgs_order),
                                [_s_attr] * args.iterations * len(dgs_order),
                                np.repeat(dgs_order, args.iterations),
                                np.concatenate(del_dp_samples[:, :-1].T)
                            )))
                with open(os.path.join(plots_path, f'{metric.lower()}_plot_del_df_data.pkl'), 'wb') as pddd_file:
                    pickle.dump(plot_del_df_data, pddd_file)
                with open(os.path.join(plots_path, f'{metric.lower()}_plot_del_df_data_per_group.pkl'), 'wb') as pdddpg_file:
                    pickle.dump(plot_del_df_data_per_group, pdddpg_file)
            else:
                with open(os.path.join(plots_path, f'{metric.lower()}_plot_del_df_data.pkl'), 'rb') as pddd_file:
                    plot_del_df_data = pickle.load(pddd_file)
                with open(os.path.join(plots_path, f'{metric.lower()}_plot_del_df_data_per_group.pkl'), 'rb') as pdddpg_file:
                    plot_del_df_data_per_group = pickle.load(pdddpg_file)

            plot_del_df_line = pd.DataFrame(plot_del_df_data, columns=plot_del_columns)

            if metric.lower() not in fairest_del_edges_conf:
                fairest_del_edge_df = plot_del_df_line.groupby(index_cols).mean().sort_values(y_col)
                fairest_del_edge_df = fairest_del_edge_df.reset_index().groupby(index_cols[:-1]).first()
                fair_conf = fairest_del_edge_df.reset_index().set_index(index_cols).index
                fairest_del_edges_conf[metric.lower()] = fair_conf
            else:
                fair_conf = fairest_del_edges_conf[metric.lower()]

            plot_df_box_bar = plot_del_df_line.set_index(index_cols).loc[fair_conf].reset_index()
            plot_df_box_bar.to_csv(
                os.path.join(plots_path, f'{metric.lower()}_fairest_dp_df.csv'), index=False
            )
            del plot_del_df_line

            dp_df_per_group = pd.DataFrame(plot_del_df_data_per_group, columns=del_data_columns_per_group)
            fairest_dp_df_per_group = dp_df_per_group.set_index(index_cols).loc[fair_conf].reset_index()
            fairest_dp_df_per_group.to_csv(
                os.path.join(plots_path, f'{metric.lower()}_fairest_dp_df_per_group.csv'), index=False
            )
            del dp_df_per_group
        else:
            plot_df_box_bar = pd.read_csv(os.path.join(plots_path, f'{metric.lower()}_fairest_dp_df.csv'))
            fairest_dp_df_per_group = pd.read_csv(os.path.join(plots_path, f'{metric.lower()}_fairest_dp_df_per_group.csv'))

        create_table_best_explanations_per_group(fairest_dp_df_per_group)

        #         plot_table_df_bar_gby = table_bar_df.groupby(["Sens Attr", "Dataset"])
        # plot_del_df_line_gby = plot_del_df_line.groupby(["Sens Attr", "Dataset", "Model"])
        plot_df_box_bar_gby = plot_df_box_bar.groupby(["Sens Attr", "Dataset"])

        metr_df_plot_table_gby = metric_del_df.groupby(index_cols)

        hatches = ['//', 'o']
        fig_bar2, axs_bar2 = plt.subplots(len(unique_sens_attrs), len(unique_datasets), squeeze=False, figsize=(10, 6))
        axs_bar2 = [axs_bar2] if not isinstance(axs_bar2, np.ndarray) else axs_bar2
        # table_bar_df.loc[table_bar_df["Status"] == "Before", "Policy"] = no_pert_col
        # table_bar_df = table_bar_df.drop("Status", axis=1).rename(columns={'value': y_col})

        for s_attr_i, orig_sens_attr in enumerate(unique_sens_attrs):
            sens_attr = orig_sens_attr.title().replace('_', ' ')

            fig_pol_box, axs_pol_box = plt.subplots(1, len(unique_datasets), figsize=(10, 6))
            axs_pol_box = [axs_pol_box] if not isinstance(axs_pol_box, np.ndarray) else axs_pol_box

            fig_box, axs_box = plt.subplots(1, len(unique_datasets), figsize=(10, 6))
            axs_box = [axs_box] if not isinstance(axs_box, np.ndarray) else axs_box

            # col_widhts = [0.07, 0.07, 0.07, 0.09, 0.09, 0.07, 0.07, 0.09, 0.09, 0.07, 0.07, 0.07]
            under_plot_table_data = [[], []]

            s_attr_dgs = []
            for i, (dset, ax_pol_box, ax_box) in enumerate(zip(unique_datasets, axs_pol_box, axs_box)):
                if (sens_attr, dset) in plot_df_box_bar_gby.groups:
                    dset_box_bar_sattr_df = plot_df_box_bar_gby.get_group((sens_attr, dset))
                    box = sns.boxplot(x="Model", y=y_col, data=dset_box_bar_sattr_df, hue="Policy",
                                      hue_order=policy_order, ax=ax_pol_box, palette=palette, showfliers=False)
                    # ax_bar.set_title(dataset_map[dset], pad=30)
                    ax_pol_box.set_xlabel("")

                    patches = [artist for artist in box.get_children() if isinstance(artist, mpatches.PathPatch)]
                    # sorted by x0 of the Bbox to ensure the order from left to right
                    patches = sorted(patches, key=lambda x: x.get_extents().x0)

                    stest_df = dset_box_bar_sattr_df.set_index(index_cols[:-1])
                    plt.rcParams['hatch.linewidth'] = 0.4
                    for mod, mod_ptcs in zip(unique_models, np.array_split(patches, len(unique_models))):
                        mod_ptcs_text = [''] * len(mod_ptcs)
                        data_ptcs = [stest_df.loc[(dset, mod, pol, sens_attr), y_col] for pol in policy_order if
                                     (dset, mod, pol, sens_attr) in stest_df.index]

                        if len(under_plot_table_data[0]) > 0:
                            under_plot_table_data[0] += ['']
                            under_plot_table_data[1] += ['']

                        for stest_pol in policy_order:
                            if (dset, mod, stest_pol, sens_attr) in stest_df.index:
                                stest_epoch = stest_df.loc[(dset, mod, stest_pol, sens_attr), 'Epoch'].iloc[0]
                                stest_metr_val = metr_df_plot_table_gby.get_group(
                                    (dset, mod, stest_pol, sens_attr, stest_epoch)
                                )['Value'].mean()
                                under_plot_table_data[0] += [f"{stest_metr_val:.2f}"]

                                stest_del_edges = stest_df.loc[(dset, mod, stest_pol, sens_attr), '# Del Edges'].iloc[0]
                                stest_del_edges = stest_del_edges / train_datasets[dset].inter_num * 100
                                under_plot_table_data[1] += [f"{stest_del_edges:.1f}" if stest_pol != no_pert_col else '']

                        bxp_stats = [mpl_cbook.boxplot_stats(d_ptc)[0] for d_ptc in data_ptcs]
                        height = max([_bxps['whishi'] for _bxps in bxp_stats])
                        yerr = 0
                        for (data1, ptc1), (data2, ptc2) in itertools.combinations(zip(data_ptcs, mod_ptcs), 2):
                            tt_stat, tt_pv = scipy.stats.ttest_rel(data1, data2)
                            if tt_pv < P_VALUE:
                                x, h = [], [height, height]
                                for _ptc in [ptc1, ptc2]:
                                    x.append(np.mean([_ptc.get_path().vertices[0][0], _ptc.get_path().vertices[1][0]]))
                                dh, barh = .06, .015
                                ax_pol_t = utils.annotate_brackets(
                                    ax_pol_box, 0, 1, tt_pv, x, h, [yerr, yerr], dh, barh, fs=SMALL_SIZE
                                )
                                # yerr is updated such that next brackets are drawn over others
                                offset_norm = abs(height)
                                yerr += dh * offset_norm + barh * offset_norm + offset_norm * SMALL_SIZE * 5e-4

                        bxp_hatches = [v for k, v in pol_hatches.items() if k in policy_order]
                        for _ptc, _bxp, bxp_hatch in zip(mod_ptcs, bxp_stats, bxp_hatches):
                            x = np.mean([_ptc.get_path().vertices[0][0], _ptc.get_path().vertices[1][0]])
                            med = _bxp["med"]
                            med *= 1 + (0.05 * np.sign(med))
                            med_text = ax_pol_box.text(
                                x, med, f"{med:.3f}", fontsize=SMALL_SIZE, ha='center', color='w'
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

                    if args.add_plot_table:
                        ax_pol_table = ax_pol_box.table(
                            under_plot_table_data,
                            rowLabels=[f"{metric.upper()}@10", 'Del Edges (%)'],
                            # colWidths=col_widhts,
                            loc='upper left',
                            bbox=[0, -0.1, 1, 0.1]
                        )
                        ax_pol_table.auto_set_font_size(False)
                        for (tab_row, tab_col), tab_cell in ax_pol_table.get_celld().items():
                            fontsize = SMALL_SIZE + (0 if tab_col != -1 else -3)
                            tab_cell.set_fontsize(fontsize)

                        # if dset == "insurance":
                        #     for tick in ax_pol_box.xaxis.get_major_ticks():
                        #         tick.set_pad(42)
                    ax_pol_box.tick_params(axis='x', length=0)

                    dset_box_bar_sattr_mondel_df = dset_box_bar_sattr_df[dset_box_bar_sattr_df["Policy"] != delcons_pol]
                    sns.boxplot(
                        x="Model", y=y_col, data=dset_box_bar_sattr_mondel_df, hue="Policy", ax=ax_box, palette=palette
                    )
                    # ax_box.set_title(dataset_map[dset], pad=30)
                    ax_box.set_xlabel("")

                    if False:  # i == len(axs_pol_box) - 1 and dset == "ml-1m":
                        pol_box_handles, pol_box_labels = ax_pol_box.get_legend_handles_labels()
                        fig_pol_box.legend(
                            pol_box_handles, pol_box_labels, loc='lower center',
                            ncol=len(pol_box_labels), bbox_to_anchor=[0.5, 1.01]
                        )
                        box_handles, box_labels = ax_box.get_legend_handles_labels()
                        fig_box.legend(box_handles, box_labels, loc='upper center', ncol=len(box_labels))
                    ax_pol_box.get_legend().remove()
                    ax_box.get_legend().remove()

                    # if dset != 'insurance':
                    ax_pol_box.tick_params('x', bottom=False, labelbottom=False)

            # create_fig_bar2_legend(fig_bar2, palette, hatches, s_attr_dgs, loc="upper left")

            fig_pol_box.tight_layout(pad=0)
            # fig_pol_box.subplots_adjust(left=0.2, bottom=0.2)
            fig_pol_box.savefig(
                os.path.join(
                    plots_path,
                    f"{'_'.join(unique_datasets)}_{sens_attr}_all_policies_boxplot_{exp_data_name}_{metric}_DP_random_samples.png"),
                bbox_inches='tight', pad_inches=0, dpi=250
            )

            fig_box.tight_layout()
            fig_box.savefig(
                os.path.join(plots_path, f"{sens_attr}_boxplot_{mondel_pol}_{exp_data_name}_{metric}_DP_random_samples.png")
            )

        # fig_bar2.tight_layout()
        # fig_bar2.savefig(os.path.join(plots_path, f"overlapping_barplot_{exp_data_name}_{metric}_DP_random_samples.png"))
        plt.close("all")

        # need to be done here due to maximum number of axes opened simultaneously
#         for s_attr_i, orig_sens_attr in enumerate(unique_sens_attrs):
#             sens_attr = orig_sens_attr.title().replace('_', ' ')
#             fig_line = plt.figure(figsize=(15, 15), constrained_layout=True)
#             subfigs = fig_line.subfigures(len(unique_datasets), 1)
#             subfigs = [subfigs] if not isinstance(subfigs, np.ndarray) else subfigs

#             for i, dset in enumerate(unique_datasets):
#                 subfigs[i].suptitle(dset.upper())
#                 axs_line = subfigs[i].subplots(1, len(unique_models), sharey=True)
#                 axs_line = [axs_line] if not isinstance(axs_line, np.ndarray) else axs_line
#                 for m, ax_line in zip(unique_models, axs_line):
#                     if (sens_attr, dset, m) in plot_del_df_line_gby.groups:
#                         dset_model_line_df = plot_del_df_line_gby.get_group((sens_attr, dset, m))
#                         sns.lineplot(x="# Del Edges", y=y_col, data=dset_model_line_df, hue="Policy", ax=ax_line,
#                                      palette=palette, ci=None)
#                         # ax_line.set_title(m.upper() + (f'+{incdisp[(dset, m)]}' if incdisp[(dset, m)] else ''))
#                         ax_line.xaxis.set_major_formatter(
#                             mpl_tick.FuncFormatter(lambda x, pos: f"{x / datasets_train_inter_sizes[dset] * 100:.2f}%"))
#                     if m == unique_models[-1] and dset == "ml-1m":
#                         line_handles, line_labels = ax_line.get_legend_handles_labels()
#                         fig_line.legend(line_handles, line_labels, loc='lower center', ncol=len(line_labels),
#                                         bbox_to_anchor=[0.5, 1.01])
#                     ax_line.get_legend().remove()

#             # fig_line.suptitle(sens_attr.title().replace('_', ' '))
#             fig_line.tight_layout()
#             fig_line.savefig(
#                 os.path.join(plots_path, f"{sens_attr}_lineplot_{exp_data_name}_{metric}_DP_random_samples.png"),
#                 bbox_inches="tight", pad_inches=0, dpi=250
#             )

        plt.close("all")

    with open(os.path.join(plots_path, 'fairest_del_edges_conf.pkl'), 'wb') as f:
        pickle.dump(fairest_del_edges_conf, f)

    gc.collect()

    gm_analysis_tables_data = {}
    # if "mi" in args.graph_metrics_analysis_tables:
    #     gm_analysis_tables_data["mi"] = []
    # if "wd" in args.graph_metrics_analysis_tables:
    #     gm_analysis_tables_data["wd"] = []
    # if "kl" in args.graph_metrics_analysis_tables:
    #     gm_analysis_tables_data["kl"] = []
    # if "dcor" in args.graph_metrics_analysis_tables:
    #     gm_analysis_tables_data["dcor"] = []
    #     gm_analysis_tables_data["dcor_pval"] = []
    if "del_dist" in args.graph_metrics_analysis_tables:
        gm_analysis_tables_data["del_dist"] = []

    def get_adv_group_for_graph_metric(adv_gr_df_gby, adv_gr_meta):
        adv_gr_df = adv_gr_df_gby.get_group(tuple(list(adv_gr_meta)[:2] + ["NP"] + [adv_gr_meta[-1]]))
        adv_gr_df = adv_gr_df[adv_gr_df.Metric == 'NDCG']
        adv_gr_df = next(adv_gr_df.groupby('Epoch').__iter__())[1]
        adv_gr = adv_gr_df.groupby("Demo Group").mean()["Value"].sort_values().index[-1]

        return adv_gr

    adv_groups_map = {}
    fairest_conf = fairest_del_edges_conf["ndcg"]  # metric not relevant here
    fairest_del_df = del_df.set_index(index_cols).loc[fairest_conf]
    fairest_del_df = fairest_del_df[fairest_del_df["Metric"] == "NDCG"]  # metric not relevant here
    m_dset_attr_pol_gby = fairest_del_df.reset_index().groupby(index_cols[:-1])
    for (_dataset, _model, _policy, _s_attr), m_dset_attr_pol_df in m_dset_attr_pol_gby:
        df_metadata = (_dataset, _model, _s_attr)
        if df_metadata not in adv_groups_map:
            gm_adv_group = get_adv_group_for_graph_metric(m_dset_attr_pol_gby, df_metadata)
            adv_groups_map[df_metadata] = group_name_map[gm_adv_group]

        # len_pol = len(unique_policies[1:])
        if _policy not in [no_pert_col, casper_pol]:  # without perturbation there are no perturbed edges
            s_attr_df = m_dset_attr_pol_df.copy(deep=True)

            best_epoch = s_attr_df.loc[s_attr_df.index[0], 'Epoch']
            de = np.array(all_del_edges[(exp_data_name, _dataset, _model, _policy, _s_attr, best_epoch)])
            de_count = de.shape[1]

            # remove user and item id 0 padding
            de -= 1
            de[1] -= 1

            graph_mdf = graph_metrics_dfs[_dataset].copy(deep=True)
            # each edge is counted once for one node and once for the other (it is equal to a bincount)
            graph_mdf['Del Edges Count'] = np.bincount(de.flatten(), minlength=len(graph_mdf))

            s_attr_df['user_id'] -= 1  # reindexing to zero
            sens_gmdf = graph_mdf.join(
                s_attr_df.reset_index()[['user_id', 'Sens Attr', 'Demo Group']].set_index('user_id'),
                on='Node'
            ).fillna('Item')

            dset_stats = os.path.join(base_all_plots_path, 'datasets_stats.csv')
            if os.path.exists(dset_stats):
                dsets_df = pd.read_csv(dset_stats, index_col=0)
                if _dataset in dsets_df.columns:
                    from pandas.api.types import is_numeric_dtype
                    graph_mean_dg = sens_gmdf.groupby(['Sens Attr', 'Demo Group']).mean().loc[_s_attr, gm_dep_order]
                    graph_gini_dg = sens_gmdf.groupby(['Sens Attr', 'Demo Group']).agg(
                        {col: gini for col in sens_gmdf.columns if is_numeric_dtype(sens_gmdf[col])}
                    ).loc[_s_attr, gm_dep_order]
                    for gm in gm_dep_order:
                        for gm_str, graph_stats_dg in zip(
                            ["bar" + "{" + f"{gm}" + "}", f"Gini {gm}"], [graph_mean_dg, graph_gini_dg]
                        ):
                            if gm_str not in dsets_df.index:
                                gm_stat_dg = ""
                            else:
                                gm_stat_dg = dsets_df.loc[gm_str, _dataset]
                                gm_stat_dg = "" if isinstance(gm_stat_dg, float) else gm_stat_dg  # avoids NaN
                                if _s_attr in gm_stat_dg:
                                    continue

                            gm_stat_dg += f" {_s_attr} " + '; '.join(graph_stats_dg[gm].to_frame().reset_index().apply(
                                lambda x: f"{x['Demo Group']} : {x[gm]:{'.1f' if gm == 'Degree' and 'bar' in gm_str else '.2f'}}", axis=1
                            ).values)
                            dsets_df.loc[gm_str, _dataset] = gm_stat_dg

                            dsets_df.to_csv(dset_stats)

            compute_graph_metrics_analysis_tables_data(
                sens_gmdf,
                gm_analysis_tables_data,
                [_dataset, _model, _policy],
                _iterations=args.iterations
            )

    def del_dist_applymap(twentiles):
        offset = 5
        quartiles = np.array([sum(twentiles[offset * i: offset * (i + 1)]) for i in range(20 // offset)])
        highest_idx = np.argmax(quartiles)
        return f"{del_dist_map[highest_idx]} ({quartiles[highest_idx] * 100:.1f}\%)"

    def del_dist_hl(del_dist_row, row_sattr):
        new_row = del_dist_row.copy(deep=True)
        row_model = del_dist_row.name
        for row_cols, row_col_val in del_dist_row.items():
            if adv_groups_map[(row_cols[0], row_model, row_sattr)] == row_cols[2]:
                new_row.loc[row_cols] = ("\hl{" + str(row_col_val) + "}")
            else:
                new_row.loc[row_cols] = (str(row_col_val))
        return new_row

    del_dist_giant_cols = [
        "Dataset", "Model", "Policy", "Sens Attr", "Graph Metric",
        "Demo Group", "Del Edges Distribution", "Quartile"
    ]
    del_dist_giant_path = os.path.join(base_all_plots_path, 'del_dist_giant.csv')
    if os.path.exists(del_dist_giant_path):
        del_dist_giant = pd.read_csv(del_dist_giant_path)
    else:
        del_dist_giant = pd.DataFrame(columns=del_dist_giant_cols)
    for dep_type, gm_dep_data in gm_analysis_tables_data.items():
        if not gm_dep_data:
            continue

        gm_dep_df = pd.DataFrame(
            gm_dep_data,
            columns=["Dataset", "Model", "Policy", "Sens Attr", "Demo Group", *gm_dep_order]
        ).drop_duplicates(
            subset=["Dataset", "Model", "Policy", "Sens Attr", "Demo Group"]  # removes duplicated gms of item nodes
        )
        gm_dep_df = gm_dep_df.melt(
            ['Dataset', 'Model', 'Policy', 'Sens Attr', 'Demo Group'],
            var_name="Graph Metric", value_name="Value"
        )

        for _pol, gmd_pol_df in gm_dep_df.groupby("Policy"):
            if _pol == no_pert_col:
                continue

            if dep_type == 'del_dist' and _pol == mondel_pol:
                pol_del_dist_giant = []

            gmd_models = gmd_pol_df['Model'].unique()

            del_fig, del_axs = {}, {}
            for gm_mod in gmd_models:
                del_fig[gm_mod], del_axs[gm_mod] = plt.subplots(
                    len(gmd_pol_df['Sens Attr'].unique()),
                    len(gmd_pol_df['Graph Metric'].unique()),
                    sharex=True, sharey='row', figsize=(25, 10)
                )

            for gm_sa_i, (gm_sa, gmd_df) in enumerate(gmd_pol_df.groupby('Sens Attr')):
                # for RQ3
                # gmd_df = gmd_sa_df[(gmd_sa_df["Model"] != 'NGCF') & (gmd_sa_df["Policy"] == mondel_pol)]
                # gmd_df = gmd_sa_df[(gmd_sa_df["Policy"] == mondel_pol)]
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

                if dep_type == "del_dist":
                    dset = gm_dep_pivot.columns.get_level_values(0).unique()[0]
                    # if gm_sa.lower() == 'gender' and \
                       # 'ml-1m' in gm_dep_pivot.columns.get_level_values(0).unique() and \
                    for gm_dep_model in gm_dep_pivot.index:
                        del_dist_plot_df = gm_dep_pivot.loc[gm_dep_model, dset].copy(deep=True)
                        len_dist_plot_df = len(del_dist_plot_df)
                        parts = 4
                        del_dist_plot_df = del_dist_plot_df.apply(
                            # lambda x: np.array([0] + [sum(x[2 * i: 2 * (i + 1)]) for i in range(parts)])
                            lambda x: np.array([sum(x[(20 // parts) * i: (20 // parts) * (i + 1)]) for i in range(parts)])
                        )
                        del_dist_plot_df = del_dist_plot_df.to_frame().explode(gm_dep_model)
                        del_dist_plot_df.rename(columns={gm_dep_model: 'Del Edges Distribution'}, inplace=True)
                        col_name = 'Del Edges Distribution'
                        del_dist_plot_df[col_name] = del_dist_plot_df[col_name].astype(float)
                        # del_dist_plot_df[col_name] = del_dist_plot_df.reset_index().groupby(
                        #     ['Graph Metric', 'Demo Group']
                        # )[col_name].cumsum().to_numpy()  # transformation to CDF

                        # percentiles = np.tile(np.arange(0, parts + 1), len_dist_plot_df)
                        percentiles = np.tile(np.arange(0, parts), len_dist_plot_df)
                        bar_width = percentiles[1] / parts / 2
                        del_dist_plot_df['Percentile'] = percentiles / parts + bar_width

                        del_dist_plot_df_vals = del_dist_plot_df.reset_index().values
                        del_dist_giant_data = np.c_[
                            [dset] * del_dist_plot_df_vals.shape[0],
                            [gm_dep_model] * del_dist_plot_df_vals.shape[0],
                            [_pol]  * del_dist_plot_df_vals.shape[0],
                            [gm_sa]  * del_dist_plot_df_vals.shape[0],
                            del_dist_plot_df_vals
                        ]
                        pol_del_dist_giant.extend(del_dist_giant_data.tolist())

                        plot_cols = del_dist_plot_df.index.get_level_values(0).unique()
                        plot_cols = [x for x in gm_metrics_base if x in plot_cols]  # reordering
                        for _del_ax, plot_col in zip(del_axs[gm_mod][gm_sa_i], plot_cols):
                            sns.barplot(
                                data=del_dist_plot_df.loc[plot_col].reset_index(),
                                x='Percentile', y=col_name, hue='Demo Group', palette='colorblind', ax=_del_ax
                            )
                            # sns.scatterplot(
                            #     data=del_dist_plot_df.loc[plot_col].reset_index(), legend=False,
                            #     x='Percentile', y=col_name, hue='Demo Group', palette='colorblind', ax=_del_ax
                            # )
                            _del_ax.grid(axis='both', ls=':')
                            if gm_sa_i == 0:
                                _del_ax.set_title(plot_col)
                            if gm_sa_i != len(gmd_pol_df['Sens Attr'].unique()) - 1:
                                _del_ax.set_xlabel('')
                            if plot_col != plot_cols[0]:
                                _del_ax.set_ylabel('')
                            # # _del_ax.set_xlim((-0.01, 1.01))
                            # _del_ax.tick_params(
                            #     axis='both',
                            #     which='both',
                            #     bottom=gm_sa_i == len(gmd_pol_df['Sens Attr'].unique()) - 1,
                            #     left=gm_sa_i == 0,
                            #     labelbottom=gm_sa_i == len(gmd_pol_df['Sens Attr'].unique()) - 1,
                            #     labelleft=gm_sa_i == 0,
                            # )

#                         bottom, top = del_axs[0].get_ylim()
#                         whisk_h = abs(top - bottom) / 30

#                         for _del_ax in del_axs:
#                             _del_ax.plot(
#                                 np.repeat([0, 0.25, 0.5, 0.75, 1.00], 4),
#                                 [top - whisk_h / 2, top - whisk_h, top, top - whisk_h / 2] * 5,
#                                 'k'
#                             )
#                             for x_label, dd_map_label in zip([0.125, 0.375, 0.625, 0.875], del_dist_map):  # del_dist_map2:
#                                 _del_ax.annotate(
#                                     dd_map_label, (x_label, top - whisk_h / 2),
#                                     xytext=(0, (top - whisk_h / 2) / 2),
#                                     textcoords="offset points",
#                                     ha='center',
#                                     va='bottom',
#                                 )

                    gm_dep_pivot = gm_dep_pivot.applymap(del_dist_applymap)
                    if gm_sa != "Item":
                        gm_dep_pivot = gm_dep_pivot.apply(lambda r: del_dist_hl(r, gm_sa), axis=1)
                else:
                    gm_dep_pivot = gm_dep_pivot.round(2)

                gm_dep_pivot.to_latex(
                    os.path.join(plots_path, f"{dep_type}_{_pol}_table_graph_metrics_{gm_sa}_{exp_data_name}.tex"),
                    column_format="c" * (gm_dep_pivot.shape[1] + 1),
                    multicolumn_format="c",
                    escape=False
                )

            if dep_type == 'del_dist' and _pol == mondel_pol:
                pol_del_dist_giant_df = pd.DataFrame(pol_del_dist_giant, columns=del_dist_giant_cols)
                if not del_dist_giant['Dataset'].isin(unique_datasets).any():
                    del_dist_giant = pd.concat([del_dist_giant, pol_del_dist_giant_df], ignore_index=True)
                del_dist_giant.to_csv(del_dist_giant_path, index=False)

            for gm_mod in gmd_models:
                del_fig[gm_mod].tight_layout()
                del_fig[gm_mod].savefig(
                    os.path.join(plots_path, f"{gm_mod}_del_dist_{_pol}_plot_{dset}.png"),
                    bbox_inches="tight", pad_inches=0, dpi=250
                )

    if "Sens Attr" not in del_dist_giant:
        dg_to_sa = {
            "Younger": "Age", "Older": "Age", "Males": "Gender", "Females": "Gender", "Item": "Item"
        }
        del_dist_giant["Sens Attr"] = del_dist_giant["Demo Group"].map(dg_to_sa)

    short_gm = {"Degree": "DEG", "Density": "DY", "Sparsity": "SP", "Reachability": "SP"}

    rq3_confs = {
        "Gender": ["ml-1m", "lastfm-1k"],
        "Age": ["ml-1m", "tafeng", "lastfm-1k"],
        "Item": ["ml-1m", "tafeng", "lastfm-1k"]
    }
    rq3_dsets = list(np.unique([d for conf in rq3_confs.values() for d in conf]))
    rq3_groups = [(0, group_name_map[gr]) for gr in ["Y", "O", "M", "F"]] + [(1, "Item")]
    if (set(del_dist_giant["Dataset"].unique()) & set(rq3_dsets)) == set(rq3_dsets):
        spine_color = "red"

        del_dist_giant["Dataset"] = del_dist_giant["Dataset"].map(plot_dataset_map)
        unique_quantiles = del_dist_giant["Quartile"].unique()
        parts = len(unique_quantiles) * 2
        del_dist_giant["Quartile"] = del_dist_giant["Quartile"].map(
            lambda x: [f"Q{i + 1}" for i in range(parts)][int((x * parts - 1) // 2)]
        )
        giant_hue_col = "Demo Group"
        giant_y_col = "Del Edges Distribution"
        giant_x_col = "Quartile"
        giant_hue_col_order = {
            "Age": ["Younger", "Older"],
            "Gender": ["Males", "Females"],
            "Item": ["Item"],
        }

        fs_titles_labels = 32
        fs_ticks = 22

        for (giant_sa, giant_mod), giant_samod_df in del_dist_giant.groupby(["Sens Attr", "Model"]):
            if giant_sa in rq3_confs:
                giant_dsets = rq3_confs[giant_sa]
                mapped_giant_dsets = [plot_dataset_map[d] for d in giant_dsets]
                giant_samod_df = giant_samod_df[giant_samod_df["Dataset"].isin(mapped_giant_dsets)]
                giant_gms = giant_samod_df["Graph Metric"].unique()

                markers_map = dict(zip(giant_hue_col_order[giant_sa], ["d", "X", "*"]))

                adv_dsets_color_map = {
                    "ML-1M": {"Males": spine_color, "Younger": spine_color},
                    "FENG": {"Older": spine_color},
                    "LFM-1K": {"Females": spine_color, "Older": spine_color},
                    "INS": {"Males": spine_color, "Younger": spine_color},
                }
                if giant_mod == "NGCF":
                    del adv_dsets_color_map["INS"]["Younger"]
                    adv_dsets_color_map["INS"]["Older"] = spine_color

                giant_fig, giant_axs = plt.subplots(
                    len(giant_dsets), len(giant_gms), sharey='row', sharex=True,
                    figsize=(30, 3 * len(giant_dsets)), layout="constrained"
                )
                giant_fig.supylabel('Del Edges Distribution', fontsize=fs_titles_labels)
                giant_fig.supxlabel('Quartiles', ha='left', fontsize=fs_titles_labels)

                giant_df_gby = giant_samod_df.groupby(["Dataset", "Graph Metric"])
                for d_i, giant_dset in enumerate(mapped_giant_dsets):
                    for gm_i, giant_gm in enumerate(giant_gms):
                        giant_ax = giant_axs[d_i, gm_i]

                        if (giant_dset, giant_gm) in giant_df_gby.groups:
                            dset_g_plot_df = giant_df_gby.get_group((giant_dset, giant_gm))
                            dset_g_plot_df = dset_g_plot_df[[giant_x_col, giant_y_col, giant_hue_col]]
                            sns.lineplot(
                                x=giant_x_col,
                                y=giant_y_col,
                                data=dset_g_plot_df,
                                hue=giant_hue_col,
                                hue_order=giant_hue_col_order[giant_sa],
                                style=giant_hue_col,
                                style_order=giant_hue_col_order[giant_sa],
                                markers=markers_map,
                                markersize=25,
                                lw=2,
                                dashes=False,
                                legend="full",
                                alpha=0.8,
                                ax=giant_ax
                            )

                            if d_i == 0 and gm_i == 0:
                                _handles, _labels = giant_ax.get_legend_handles_labels()
                                giant_ax.get_legend().remove()
                                giant_legend = giant_fig.legend(
                                    _handles, _labels, loc="lower center", ncol=3,
                                    bbox_to_anchor=(0.5, 1.01, 0.05, 0.05),
                                    bbox_transform=giant_fig.transFigure,
                                    markerscale=3, prop={'size': fs_titles_labels}
                                )
                                giant_legend.set_zorder(10)
                            else:
                                giant_ax.get_legend().remove()

                            if gm_i == 0:
                                giant_ax.yaxis.set_major_formatter(mpl_tick.StrMethodFormatter("{x:.2f}"))
                                giant_ax.set_ylabel(giant_dset, fontsize=fs_titles_labels)
                            else:
                                giant_ax.set_ylabel('')

                            if d_i == len(giant_dsets) - 1:
                                giant_ax.set_xlabel(giant_ax.get_xlabel(), fontsize=fs_titles_labels)

                            giant_ax.grid(True, axis='both', ls=':')
                            giant_ax.set_xlabel('')
                        else:
                            giant_ax.set_axis_off()
                            giant_ax.text(
                                0.5, 0.5, 'Node Group NA', ha='center', va='center',
                                fontdict=dict(fontsize=20), transform=giant_ax.transAxes
                            )

                        if d_i == 0:
                            giant_ax.set_title(
                                f"{giant_gm} ({short_gm[giant_gm]})" if giant_gm != 'Reachability' \
                                                                     else 'Intra-Group Distance (IDG)',
                                fontsize=fs_titles_labels
                            )

                        giant_ax.tick_params(which='major', labelsize=fs_ticks)
                        if gm_i > 0:
                            giant_ax.tick_params(length=0, labelleft=False)
                        if d_i < len(giant_dsets) - 1:
                            giant_ax.tick_params(length=0, labelbottom=False)

                        # for spine in giant_ax.spines.values():
                        #     spine.set_edgecolor(adv_dsets_color_map[giant_dset].get(giant_g, 'black'))

                # giant_fig.subfigs[0].subplots_adjust(
                #     left=0.08, bottom=0.08, right=0.98, top=0.98, wspace=0.1, hspace=0.08
                # )
                # giant_fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.4)
                giant_fig.savefig(
                    os.path.join(base_all_plots_path, f"{giant_sa}_{giant_mod}_{'_'.join(rq3_dsets)}_del_dist_plot_per_gm.png"),
                    bbox_inches="tight", pad_inches=0, dpi=250
                )
#     if (set(del_dist_giant["Dataset"].unique()) & set(rq3_dsets)) == set(rq3_dsets):
#         spine_color = "red"
#         markers_map = dict(zip(gm_metrics_base, ["d", "X", "*"]))

#         del_dist_giant["Dataset"] = del_dist_giant["Dataset"].map(plot_dataset_map)
#         unique_quantiles = del_dist_giant["Quartile"].unique()
#         parts = len(unique_quantiles) * 2
#         del_dist_giant["Quartile"] = del_dist_giant["Quartile"].map(
#             lambda x: [f"Q{i + 1}" for i in range(parts)][int((x * parts - 1) // 2)]
#         )
#         for giant_mod, giant_mod_df in del_dist_giant.groupby("Model"):
#             adv_dsets_color_map = {
#                 "ML-1M": {"Males": spine_color, "Younger": spine_color},
#                 "FENG": {"Older": spine_color},
#                 "LFM-1K": {"Females": spine_color, "Older": spine_color},
#                 "INS": {"Males": spine_color, "Younger": spine_color},
#             }
#             if giant_mod == "NGCF":
#                 del adv_dsets_color_map["INS"]["Younger"]
#                 adv_dsets_color_map["INS"]["Older"] = spine_color

#             giant_fig = plt.figure(figsize=(30, 12), layout="constrained")
#             giant_fig.supylabel('Del Edges Distribution')
#             giant_fig.supxlabel('Quartiles')
#             giant_subfigs = giant_fig.subfigures(
#                 1, 2, wspace=0.011, width_ratios=[0.8, 0.2]
#             )
#             giant_left_axs = giant_subfigs[0].subplots(len(rq3_dsets), len(rq3_groups) - 1, sharex=True, sharey=True)
#             giant_right_axs = giant_subfigs[1].subplots(len(rq3_dsets), 1, sharey=True)
#             giant_subaxs = {0: giant_left_axs, 1: giant_right_axs}

#             idx_giant_df = giant_mod_df.set_index(["Dataset", "Demo Group"])
#             for d_i, giant_dset in enumerate([plot_dataset_map[rq3_d] for rq3_d in rq3_dsets]):
#                 for g_i, (g_subf, giant_g) in enumerate(rq3_groups):
#                     giant_axs = giant_subaxs[g_subf]
#                     giant_ax = giant_axs[d_i, g_i] if g_subf == 0 else giant_axs[d_i]

#                     if g_subf == 0:
#                         for item_ax_idx in range(len(rq3_dsets)):
#                             giant_ax.get_shared_x_axes().remove(giant_axs[item_ax_idx, -1])

#                     if d_i == 0:
#                         giant_ax.set_title(giant_g)
#                     if g_i > 0 and g_subf == 0:
#                         giant_ax.tick_params(length=0, labelleft=False)
#                     if d_i < len(rq3_dsets) - 1:
#                         giant_ax.tick_params(length=0, labelbottom=False)

#                     if (giant_dset, giant_g) in idx_giant_df.index:
#                         dset_g_plot_df = idx_giant_df.loc[(giant_dset, giant_g)].reset_index(drop=True)
#                         dset_g_plot_df = dset_g_plot_df[["Quartile", "Del Edges Distribution", "Graph Metric"]]
#                         sns.lineplot(
#                             x="Quartile",
#                             y="Del Edges Distribution",
#                             data=dset_g_plot_df,
#                             hue="Graph Metric",
#                             hue_order=gm_metrics_base,
#                             style="Graph Metric",
#                             style_order=gm_metrics_base,
#                             markers=markers_map,
#                             markersize=25,
#                             lw=2,
#                             dashes=False,
#                             legend="full",
#                             alpha=0.8,
#                             ax=giant_ax
#                         )

#                         if d_i == 0 and g_i == 0:
#                             _handles, _labels = giant_ax.get_legend_handles_labels()
#                             giant_ax.get_legend().remove()
#                             giant_legend = giant_fig.legend(
#                                 _handles, _labels, loc="lower center", ncol=3,
#                                 bbox_to_anchor=(0.5, 1.01, 0.05, 0.05),
#                                 bbox_transform=giant_fig.transFigure,
#                                 markerscale=3
#                             )
#                             giant_legend.set_zorder(10)
#                         else:
#                             giant_ax.get_legend().remove()

#                         if g_subf == 1:
#                             giant_ax.set_ylim((-0.05, 1.05))

#                         if g_i == 0:
#                             giant_ax.yaxis.set_major_formatter(mpl_tick.StrMethodFormatter("{x:.2f}"))
#                             giant_ax.set_ylabel(giant_dset)
#                         else:
#                             giant_ax.set_ylabel('')

#                         giant_ax.grid(True, axis='both', ls=':')
#                         giant_ax.set_xlabel('')
#                     else:
#                         giant_ax.set_axis_off()
#                         giant_ax.text(
#                             0.5, 0.5, 'Node Group NA', ha='center', va='center',
#                             fontdict=dict(fontsize=BIGGER_SIZE), transform=giant_ax.transAxes
#                         )

#                     for spine in giant_ax.spines.values():
#                         spine.set_edgecolor(adv_dsets_color_map[giant_dset].get(giant_g, 'black'))

#             # giant_fig.subfigs[0].subplots_adjust(
#             #     left=0.08, bottom=0.08, right=0.98, top=0.98, wspace=0.1, hspace=0.08
#             # )
#             # giant_fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.4)
#             giant_fig.savefig(
#                 os.path.join(base_all_plots_path, f"{giant_mod}_{'_'.join(rq3_dsets)}_del_dist_plot.png"),
#                 bbox_inches="tight", pad_inches=0, dpi=250
#             )
