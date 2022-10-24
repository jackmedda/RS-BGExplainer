import itertools

import tqdm
import scipy
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import src.utils as utils


def extract_best_metrics(_exp_paths, best_exp_col, evaluator, data):
    result_all = {}
    pref_data_all = {}

    for e_type, e_path in _exp_paths.items():
        if e_path is None:
            continue

        model_name = e_type.replace('+FairDP', '')
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


def extract_all_exp_metrics_data(_exp_paths, train_data, rec_data, evaluator, sens_attr, rec=False):
    sensitive_map = train_data.dataset.field2id_token[sens_attr]

    user_df = pd.DataFrame({
        'user_id': train_data.dataset.user_feat['user_id'].numpy(),
        sens_attr: train_data.dataset.user_feat[sens_attr].numpy()
    })

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
                result_data[e_type][n_del][metric] = utils.compute_metric(evaluator, rec_data, gr_df, 'cf_topk_pred', metric)

            t_dist = gr_df['topk_dist'].to_numpy()
            topk_dist[e_type].extend(list(
                zip([n_del] * len(t_dist), t_dist / len(t_dist), gr_df['topk_dist'].to_numpy() / len(t_dist))
            ))

            gr_df_attr = gr_df['user_id'].drop_duplicates().to_frame().join(user_df.set_index('user_id'), on='user_id')
            n_users_data[e_type][n_del] = {sens_attr: gr_df_attr[sens_attr].value_counts().to_dict()}
            n_users_del = n_users_data[e_type][n_del][sens_attr]
            n_users_data[e_type][n_del][sens_attr] = {sensitive_map[dg]: n_users_del[dg] for dg in n_users_del}

    return exp_dfs, result_data, n_users_data, topk_dist


def compute_exp_stats_data(_result_all_data, _pref_dfs, orig_result, order, attr, user_df, d_grs, del_edges_map, metric, test_f="f_oneway"):
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


def result_data_per_epoch_per_group(exp_dfs, evaluator, group_idxs: tuple, user_df, rec_data, sens_attr):
    m_idx, f_idx = group_idxs

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
                result_per_epoch[e_type][epoch][m_idx][metric] = utils.compute_metric(evaluator, rec_data, m_df, 'cf_topk_pred', metric)[:, -1].mean()
                result_per_epoch[e_type][epoch][f_idx][metric] = utils.compute_metric(evaluator, rec_data, f_df, 'cf_topk_pred', metric)[:, -1].mean()

            del_edges = epoch_df.iloc[0]['del_edges']
            del_edges_per_epoch[e_type][epoch][m_idx] = del_edges[:, (epoch_df.loc[m_mask].user_id.values[:, None] == del_edges[0]).nonzero()[1]]
            del_edges_per_epoch[e_type][epoch][f_idx] = del_edges[:, (epoch_df.loc[f_mask].user_id.values[:, None] == del_edges[0]).nonzero()[1]]

            fair_loss_per_epoch[e_type][epoch] = epoch_df.iloc[0]['fair_loss']

    return result_per_epoch, del_edges_per_epoch, fair_loss_per_epoch


def get_adv_group_idx_to_delete(exp_path,
                                orig_model_name,
                                evaluator,
                                rec_data,
                                user_feat,
                                delete_adv_group,
                                sens_attr="gender",
                                m_idx=1,
                                f_idx=2):
    m_group, f_group = (user_feat[sens_attr] == m_idx).nonzero().T[0].numpy() - 1, \
                       (user_feat[sens_attr] == f_idx).nonzero().T[0].numpy() - 1

    # Does not matter which explanation we take if we evaluate just the recommendations of the original model
    exp_rec_df, rec_result_data = extract_best_metrics(
        {f'{orig_model_name}+FairDP': exp_path},
        "first",
        evaluator,
        rec_data.dataset
    )

    orig_m_ndcg = rec_result_data[orig_model_name]["ndcg"][
                      (m_group[:, None] == (exp_rec_df[f'{orig_model_name}+FairDP'].user_id.values - 1)).nonzero()[1]
                  ][:, -1].mean()

    orig_f_ndcg = rec_result_data[orig_model_name]["ndcg"][
                      (f_group[:, None] == (exp_rec_df[f'{orig_model_name}+FairDP'].user_id.values - 1)).nonzero()[1]
                  ][:, -1].mean()

    if orig_m_ndcg >= orig_f_ndcg:
        if delete_adv_group is not None:
            group_edge_del = m_idx if delete_adv_group else f_idx
        else:
            group_edge_del = m_idx
    else:
        if delete_adv_group is not None:
            group_edge_del = f_idx if delete_adv_group else m_idx
        else:
            group_edge_del = f_idx

    return group_edge_del


def get_centrality_graph_df(graph_nx, top, original=True, sens_attr_map=None):
    label = "Original" if original else "Perturbed"

    bottom = set(graph_nx) - top

    node_type_map = dict(zip(top, ["users"] * len(top)))
    node_type_map.update(dict(zip(bottom, ["items"] * len(bottom))))

    centr = nx.bipartite.degree_centrality(graph_nx, top)

    top, bottom = list(top), list(bottom)
    df_data = zip(top + bottom, [centr[n] for n in (top + bottom)])

    df = pd.DataFrame(df_data, columns=["node_id_minus_1", "Centrality"])

    df["Node Type"] = df["node_id_minus_1"].map(node_type_map)

    user_df, item_df = df[df["Node Type"] == "users"], df[df["Node Type"] == "items"]

    if sens_attr_map is not None:
        user_df["Group"] = user_df["node_id_minus_1"].map(sens_attr_map)

    user_df["Graph Type"], item_df["Graph Type"] = label, label

    return user_df, item_df


def get_data_sh_lt(dataloader, short_head=0.05):
    """
    Get items id mapping to short head and long tails labels
    :param dataloader:
    :param short_head:
    :return:
    """
    _, _, item_pop = dataloader.dataset.history_user_matrix()

    item_pop = item_pop[1:].numpy()

    item_pop = np.argsort(item_pop)[::-1]

    sh_n = round(len(item_pop) * short_head)
    short_head, long_tail = np.split(item_pop, [sh_n])

    return dict(zip(
        np.concatenate([short_head, long_tail]),
        ["Short Head"] * len(short_head) + ["Long Tail"] * len(long_tail)
    ))


def get_data_active_inactive(dataloader, inactive_perc=0.3, group_mask=None):
    """
    Get users id mapping to active and inactive labels
    :param dataloader:
    :param inactive_perc:
    :param group_mask:
    :return:
    """
    _, _, user_inters = dataloader.dataset.history_item_matrix()

    user_inters = user_inters[1:].numpy()

    user_inters = np.argsort(user_inters)

    inactive_n = round(len(user_inters) * inactive_perc)
    inactive, active = np.split(user_inters, [inactive_n])

    return dict(zip(
        np.concatenate([inactive, active]),
        ["Inactive"] * len(inactive) + ["Active"] * len(active)
    ))


def off_margin_ticks(*axs):
    # Turn off tick visibility for the measure axis on the marginal plots
    for ax in axs:
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_xticklabels(minor=True), visible=False)
