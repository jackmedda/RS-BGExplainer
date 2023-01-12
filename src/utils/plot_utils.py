import os
import pickle
import itertools

import tqdm
import scipy
import scipy.sparse as sp
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import src.utils as utils


def extract_best_metrics(_exp_paths, best_exp_col, evaluator, data, config=None, additional_cols=None):
    result_all = {}
    pref_data_all = {}
    filter_cols = ['user_id', 'rec_topk', 'rec_cf_topk'] if additional_cols is None else additional_cols
    filter_cols = list(set(filter_cols) | {'user_id', 'rec_topk', 'rec_cf_topk'})

    for e_type, e_path in _exp_paths.items():
        if e_path is None:
            continue

        model_name = e_type.replace('+FairDP', '')
        exps_data = utils.load_dp_exps_file(e_path)

        bec = best_exp_col[e_type] if isinstance(best_exp_col, dict) else best_exp_col

        if not isinstance(bec, list):
            bec = bec.lower() if isinstance(bec, str) else bec
        else:
            bec[0] = bec[0].lower()
        top_exp_func = None
        if isinstance(bec, int):
            def top_exp_func(exp): return exp[bec]
        elif bec == "first":
            def top_exp_func(exp): return exp[0]
        elif bec == "last":
            def top_exp_func(exp): return exp[-1]
        elif bec == "mid":
            def top_exp_func(exp): return exp[len(exp) // 2]
        elif isinstance(bec, list):
            top_exp_col = utils.EXPS_COLUMNS.index(bec) if bec is not None else None
            if top_exp_col is not None:
                def top_exp_func(exp): return sorted(exp, key=lambda x: x[top_exp_col])[0]
        elif bec == "auto":
            assert config is not None, "`auto` mode can be used only with config"
            best_epoch = utils.get_best_exp_early_stopping(exps_data[0], config)
            epoch_idx = utils.EXPS_COLUMNS.index('epoch')
            def top_exp_func(exp): return [e for e in sorted(exp, key=lambda x: abs(x[epoch_idx] - best_epoch)) if e[epoch_idx] <= best_epoch][0]
        elif isinstance(bec, list):
            top_exp_col = utils.EXPS_COLUMNS.index(bec[0])
            def top_exp_func(exp): return sorted(exp, key=lambda x: abs(x[top_exp_col] - bec[1]))[0]

        pref_data = []
        for exp_entry in exps_data:
            if top_exp_func is not None:
                _exp = top_exp_func(exp_entry)
            else:
                _exp = exp_entry[0]

            idxs = [utils.EXPS_COLUMNS.index(col) for col in filter_cols]
            del_edges_idx = utils.EXPS_COLUMNS.index('del_edges')
            del_edges_data = [_exp[del_edges_idx]] * len(_exp[idxs[0]])

            pref_data.extend(list(zip(*[*[_exp[idx] for idx in idxs], del_edges_data])))

        pref_data = pd.DataFrame(pref_data, columns=filter_cols + ['del_edges'])
        pref_data.rename(columns={'rec_topk': 'topk_pred', 'rec_cf_topk': 'cf_topk_pred'}, inplace=True)
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
        train_data.dataset.uid_field: train_data.dataset.user_feat[train_data.dataset.uid_field].numpy(),
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
        if len(_exp_paths) == 1:
            saved_path = os.path.join(e_path, 'extracted_exp_data.pkl')
            if os.path.exists(saved_path):
                with open(saved_path, 'rb') as saved_file:
                    saved_data = pickle.load(saved_file)
                for s_data, curr_data in zip(saved_data, [exp_dfs, result_data, n_users_data, topk_dist]):
                    curr_data.update(s_data)
                break
        exps_data = utils.load_dp_exps_file(e_path)

        exp_data = []
        for exp_entry in exps_data:
            for _exp in exp_entry[:-1]:  # the last epoch is a stub epoch to retrieve back the best epoch
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
        if data_df.empty:
            print(f"User explanations are empty for {e_type}")
            continue

        data_df['n_del_edges'] = data_df['del_edges'].map(lambda x: x.shape[1])
        exp_dfs[e_type] = data_df

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

            gr_df_attr = gr_df['user_id'].drop_duplicates().to_frame().join(user_df.set_index(train_data.dataset.uid_field), on='user_id')
            n_users_data[e_type][n_del] = {sens_attr: gr_df_attr[sens_attr].value_counts().to_dict()}
            n_users_del = n_users_data[e_type][n_del][sens_attr]
            n_users_data[e_type][n_del][sens_attr] = {sensitive_map[dg]: n_users_del[dg] for dg in n_users_del}

        if len(_exp_paths) == 1:
            saved_path = os.path.join(e_path, 'extracted_exp_data.pkl')
            if not os.path.exists(saved_path):
                with open(saved_path, 'wb') as saved_file:
                    pickle.dump((exp_dfs, result_data, n_users_data, topk_dist), saved_file)

    return exp_dfs, result_data, n_users_data, topk_dist


def compute_exp_stats_data(_result_all_data,
                           _pref_dfs,
                           orig_result,
                           order,
                           attr,
                           user_df,
                           d_grs,
                           del_edges_map,
                           metric,
                           test_f="f_oneway",
                           uid_field='user_id'):
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
                e_d_grs_df = e_df_grby.get_group(n_del).join(user_df.set_index(uid_field), on="user_id")
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

    u_df = user_df.set_index(rec_data.uid_field)

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
            uid = epoch_df[rec_data.uid_field]

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


def extract_graph_metrics_per_node(dataset, remove_first_row_col=False, metrics="all", **sp_kwargs):
    metrics = ["Degree", "Sparsity", "Reachability", "Sharing Potentiality"] if metrics == "all" else metrics
    sp_kwargs = sp_kwargs or dict(length=2, depth=2)

    G_df = None
    node_col = 'Node'

    igg = utils.get_bipartite_igraph(dataset, remove_first_row_col=remove_first_row_col)
    for metr in metrics:
        if metr == "Degree":
            df = pd.DataFrame(dict(zip(igg.vs.indices, igg.degree())).items(), columns=[node_col, metr])
        elif metr == "Reachability":
            last_user_id = dataset.user_num - (1 + remove_first_row_col)
            user_reach = utils.get_user_reachability(igg, last_user_id=last_user_id)

            first_item_id = last_user_id + 1
            item_reach = utils.get_item_reachability(igg, first_item_id=first_item_id)

            df = pd.DataFrame({**user_reach, **item_reach}.items(), columns=[node_col, metr])
        elif metr == "Sparsity":
            item_hist, _, item_pop = dataset.history_user_matrix()
            user_hist, _, user_hist_len = dataset.history_item_matrix()
            if remove_first_row_col:
                item_hist = item_hist[1:].where(item_hist[1:] == 0, item_hist[1:] - 1)
                item_pop = item_pop
                user_hist = user_hist[1:].where(user_hist[1:] == 0, user_hist[1:] - 1)
                user_hist_len = user_hist_len

            user_density = np.nan_to_num(
                ((item_pop[user_hist] / user_hist.shape[0]).sum(dim=1) / user_hist_len[1:]).numpy(),
                nan=0
            )
            user_sparsity = 1 - user_density

            # item density represents the activity of the users that interact with an item.
            # If only a user interact with item X and this user interacted with all the items in the catalog, then
            # the density of X is maximum. A low density then means a high sparsity, which means the users that interact
            # with that item interact with a few others
            item_density = np.nan_to_num(
                ((user_hist_len[item_hist] / item_hist.shape[0]).sum(dim=1) / item_pop[1:]).numpy(),
                nan=0
            )
            item_sparsity = 1 - item_density

            df = pd.DataFrame(
                zip(igg.vs.indices, np.concatenate([user_sparsity, item_sparsity])),
                columns=[node_col, metr]
            )
        elif metr == "Sharing Potentiality":
            item_hist, _, item_hist_len = dataset.history_user_matrix()
            user_hist, _, user_hist_len = dataset.history_item_matrix()

            n_users = user_hist[1:].shape[0]
            user_user = utils.get_node_node_graph_data(user_hist[1:].numpy())
            user_user = sp.coo_matrix(
                (user_user[:, -1], (user_user[:, 0], user_user[:, 1])), shape=(n_users, n_users)
            ).todense()
            user_sp = utils.compute_sharing_potentiality(
                user_user, user_hist[1:].numpy(), user_hist_len[1:].numpy(), **sp_kwargs
            )

            n_items = item_hist[1:].shape[0]
            item_item = utils.get_node_node_graph_data(item_hist[1:].numpy())
            item_item = sp.coo_matrix(
                (item_item[:, -1], (item_item[:, 0], item_item[:, 1])), shape=(n_items, n_items)
            ).todense()
            item_sp = utils.compute_sharing_potentiality(
                item_item, item_hist[1:].numpy(), item_hist_len[1:].numpy(), **sp_kwargs
            )

            sp_data = np.concatenate([user_sp, item_sp])
            df = pd.DataFrame(zip(range(len(sp_data)), sp_data), columns=[node_col, metr])
            import pdb; pdb.set_trace()

        if G_df is None:
            G_df = df
        else:
            G_df = G_df.join(df.set_index(node_col), on=node_col)
    return G_df


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


def get_data_active_inactive(dataloader, inactive_perc=0.3):
    """
    Get users id mapping to active and inactive labels
    :param dataloader:
    :param inactive_perc:
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


def get_user_user_data_sens_df(dataset, user_df, sens_attr, attr_map=None):
    user_history, _, _ = dataset.history_item_matrix()
    user_graph_df = get_node_node_data_feature_df(user_history.numpy(), user_df, dataset.uid_field, sens_attr, attr_map=attr_map)

    return user_graph_df


def get_item_item_data_pop_df(dataset, item_df, pop_attr):
    item_history, _, _ = dataset.history_user_matrix()
    item_graph_df = get_node_node_data_feature_df(item_history.numpy(), item_df, dataset.iid_field, pop_attr)

    return item_graph_df


def get_node_node_data_feature_df(history, node_df, id_label, feat_attr, attr_map=None):
    graph = utils.get_node_node_graph_data(history)
    graph = graph[graph[:, -1] > 0]

    graph_df = pd.DataFrame(graph, columns=[f'{id_label}_1', f'{id_label}_2', 'n_common_edges'])
    graph_df = graph_df.join(
        node_df.set_index(id_label), on=f"{id_label}_1"
    ).reset_index(drop=True).rename(columns={feat_attr: f"{feat_attr}_1"})

    if attr_map is not None:
        graph_df[f"{feat_attr}_1"] = graph_df[f"{feat_attr}_1"].map(attr_map)

    graph_df = graph_df.join(
        node_df.set_index(id_label), on=f'{id_label}_2'
    ).reset_index(drop=True).rename(columns={feat_attr: f"{feat_attr}_2"})

    if attr_map is not None:
        graph_df[f"{feat_attr}_2"] = graph_df[f"{feat_attr}_2"].map(attr_map)

    return graph_df


def compute_homophily(graph_df, group_sizes, feat_attr):
    homophily = {}
    for (gr1, gr2), gr_df in graph_df.groupby([f"{feat_attr}_1", f"{feat_attr}_2"]):
        if gr1 == gr2:
            all_gr_edges = graph_df.loc[
                (graph_df[f"{feat_attr}_1"] == gr1) |  # interactions of gr1 with any group (also gr1)
                ((graph_df[f"{feat_attr}_2"] == gr1) & (graph_df[f"{feat_attr}_1"] != gr1)),  # interactions from gr2 to gr1
                'n_common_edges'
            ].sum()
            homophily[gr1] = gr_df['n_common_edges'].sum() / all_gr_edges - group_sizes[gr1]

    return homophily


def off_margin_ticks(*axs, axis='x'):
    # Turn off tick visibility for the measure axis on the marginal plots
    f = f"get_{axis}ticklabels"

    for ax in axs:
        plt.setp(getattr(ax, f)(), visible=False)
        plt.setp(getattr(ax, f)(minor=True), visible=False)


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def add_bar_value_labels(ax, spacing=5, format='.2f', **kwargs):
    """Add labels to the end of each bar in a bar chart.
    https://stackoverflow.com/a/48372659
    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
        format (str): format of the value of the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = ("{:" + format + "}").format(y_value)

        # Create annotation
        ax.annotate(
            label,                       # Use `label` as label
            (x_value, y_value),          # Place label at end of the bar
            xytext=(0, space),           # Vertically shift label by `space`
            textcoords="offset points",  # Interpret `xytext` as offset in points
            ha='center',                 # Horizontally center label
            va=va,                       # Vertically align label differently for
            **kwargs)                    # positive and negative values.
