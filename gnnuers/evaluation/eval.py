import os
import pickle
import itertools

import tqdm
import torch
import scipy
import numba
import numpy as np
import pandas as pd
import recbole.evaluator.collector as recb_collector
from recbole.evaluator import Evaluator

import gnnuers.utils as utils


def extract_best_metrics(_exp_paths, best_exp_col, evaluator, data, config=None, additional_cols=None):
    result_all = {}
    pref_data_all = {}
    filter_cols = ['user_id', 'rec_topk', 'rec_cf_topk'] if additional_cols is None else additional_cols
    filter_cols = ['user_id'] + list((set(filter_cols) | {'user_id', 'rec_topk', 'rec_cf_topk'}) - {'user_id'})

    for e_type, e_path in _exp_paths.items():
        if e_path is None:
            continue

        model_name = e_type.replace('+FairDP', '')
        exps_data, rec_model_preds, test_model_preds = utils.load_dp_exps_file(e_path)
        model_pred_map = {'rec_topk': rec_model_preds, 'test_topk': test_model_preds}

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
            top_exp_col = utils.exp_col_index(bec) if bec is not None else None
            if top_exp_col is not None:
                def top_exp_func(exp): return sorted(exp, key=lambda x: x[top_exp_col])[0]
        elif bec == "auto":
            assert config is not None, "`auto` mode can be used only with config"
            best_epoch = utils.get_best_exp_early_stopping(exps_data[0], config)
            epoch_idx = utils.exp_col_index('epoch')
            def top_exp_func(exp): return [e for e in sorted(exp, key=lambda x: abs(x[epoch_idx] - best_epoch)) if e[epoch_idx] <= best_epoch][0]
        elif isinstance(bec, list):
            top_exp_col = utils.exp_col_index(bec[0])
            def top_exp_func(exp): return sorted(exp, key=lambda x: abs(x[top_exp_col] - bec[1]))[0]

        pref_data = []
        for exp_entry in exps_data:
            if top_exp_func is not None:
                _exp = top_exp_func(exp_entry)
            else:
                _exp = exp_entry[0]

            idxs = [utils.exp_col_index(col) for col in filter_cols]
            del_edges_idx = utils.exp_col_index('del_edges')
            del_edges_data = [_exp[del_edges_idx]] * len(_exp[idxs[0]])

            pref_data.extend(list(zip(
                  *[_exp[idx] if isinstance(idx, int) else model_pred_map[idx] for idx in idxs],
                  del_edges_data
            )))

        pref_data = pd.DataFrame(pref_data, columns=filter_cols + ['del_edges'])
        pref_data.rename(columns={'rec_topk': 'topk_pred', 'rec_cf_topk': 'cf_topk_pred'}, inplace=True)
        pref_data_all[e_type] = pref_data

        if not pref_data.empty:
            result_all[e_type] = {}
            for metric in evaluator.metrics:
                result_all[e_type][metric] = compute_metric(evaluator, data, pref_data, 'cf_topk_pred', metric)

                if model_name not in result_all:
                    result_all[model_name] = {}

                if metric not in result_all[model_name]:
                    result_all[model_name][metric] = compute_metric(evaluator, data, pref_data, 'topk_pred', metric)
        else:
            print("Pref Data is empty!")

    return pref_data_all, result_all


def extract_all_exp_metrics_data(_exp_paths,
                                 train_data,
                                 rec_data,
                                 evaluator,
                                 sens_attr,
                                 rec=False,
                                 overwrite=False,
                                 other_cols=None):
    sensitive_map = train_data.dataset.field2id_token[sens_attr]

    user_df = pd.DataFrame({
        train_data.dataset.uid_field: train_data.dataset.user_feat[train_data.dataset.uid_field].numpy(),
        sens_attr: train_data.dataset.user_feat[sens_attr].numpy()
    })

    first_type_cols = ['rec_cf_topk', 'test_cf_topk', 'rec_cf_dist', 'test_cf_dist']

    other_cols = [] if other_cols is None else other_cols
    cols = ['loss_graph_dist', 'fair_loss', 'del_edges', 'epoch'] + other_cols
    if not rec:
        cols = ['test_topk', 'test_cf_topk', 'test_cf_dist'] + cols
    else:
        cols = ['rec_topk', 'rec_cf_topk', 'rec_cf_dist'] + cols

    col_names = [
        'user_id',
        'topk_pred',
        'cf_topk_pred',
        'topk_dist',
        'dist_loss',
        'fair_loss',
        'del_edges',
        'epoch'
    ] + other_cols

    exp_dfs = {}
    result_data = {}
    n_users_data = {}
    topk_dist = {}
    for e_type, e_path in _exp_paths.items():
        if e_path is None:
            continue
        if len(_exp_paths) == 1:
            saved_path = os.path.join(e_path, 'extracted_exp_data.pkl')
            if os.path.exists(saved_path) and not overwrite:
                with open(saved_path, 'rb') as saved_file:
                    saved_data = pickle.load(saved_file)
                for s_data, curr_data in zip(saved_data, [exp_dfs, result_data, n_users_data, topk_dist]):
                    curr_data.update(s_data)
                break
        exps_data, rec_model_preds, test_model_preds = utils.load_dp_exps_file(e_path)
        model_pred_map = {'rec_topk': rec_model_preds, 'test_topk': test_model_preds}

        exp_data = []
        for exp_entry in exps_data:
            for _exp in exp_entry[:-1]:  # the last epoch is a stub epoch to retrieve back the best epoch
                exp_row_data = [_exp[0]]
                for col in cols:
                    if col in first_type_cols:
                        exp_row_data.append(_exp[utils.exp_col_index(col)])
                    elif col == "set":
                        orig_pred = model_pred_map[cols[0]]
                        cf_pred = _exp[utils.exp_col_index(cols[1])]
                        comm_items = np.array([len(set(orig) & set(pred)) for orig, pred in zip(orig_pred, cf_pred)])
                        exp_row_data.append(len(orig_pred[0]) - comm_items)
                    else:
                        idx = utils.exp_col_index(col)
                        if isinstance(idx, int):
                            exp_row_data.append([_exp[col]] * len(exp_row_data[0]))
                        else:
                            exp_row_data.append(np.tile(model_pred_map[col], len(exp_row_data[0])))

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
                result_data[e_type][n_del][metric] = compute_metric(evaluator, rec_data, gr_df, 'cf_topk_pred', metric)

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
            if not os.path.exists(saved_path) or overwrite:
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
                except ValueError:
                    orig_stats_data = [1] * len(final_bins)

                comb_orig_data = list(itertools.combinations(temp_orig_data, 2))
                for (g1, g2) in comb_orig_data:
                    orig_data.append(abs(g1 - g2))
                orig_data = [sum(orig_data) / len(comb_orig_data)] * len(final_bins)

    exp_data.insert(0, orig_data)
    stats_data.insert(0, orig_stats_data)

    return exp_data, stats_data, final_bins


def compute_metric(evaluator, dataset, pref_data, pred_col, metric):
    hist_matrix, _, _ = dataset.history_item_matrix()
    dataobject = recb_collector.DataStruct()
    uid_list = pref_data['user_id'].to_numpy()

    pos_matrix = np.zeros((dataset.user_num, dataset.item_num), dtype=int)
    pos_matrix[uid_list[:, None], hist_matrix[uid_list]] = 1
    pos_matrix[:, 0] = 0
    pos_len_list = torch.tensor(pos_matrix.sum(axis=1, keepdims=True))
    pos_idx = torch.tensor(pos_matrix[uid_list[:, None], np.stack(pref_data[pred_col].values)])
    pos_data = torch.cat((pos_idx, pos_len_list[uid_list]), dim=1)

    dataobject.set('rec.topk', pos_data)

    pos_index, pos_len = evaluator.metric_class[metric].used_info(dataobject)
    if metric in ['hit', 'mrr', 'precision']:
        result = evaluator.metric_class[metric].metric_info(pos_index)
    else:
        result = evaluator.metric_class[metric].metric_info(pos_index, pos_len)

    return result


def compute_metric_per_group(evaluator, data, user_df, pref_data, sens_attr, group_idxs, col='topk_pred', metric="ndcg", raw=False):
    m_idx, f_idx = group_idxs

    m_group_mask = pref_data.user_id.isin(user_df.loc[user_df[sens_attr] == m_idx, data.dataset.uid_field])
    f_group_mask = pref_data.user_id.isin(user_df.loc[user_df[sens_attr] == f_idx, data.dataset.uid_field])

    metric_result = compute_metric(
        evaluator,
        data.dataset,
        pref_data,
        col,
        metric
    )[:, -1]

    _m_result, _f_result = metric_result[m_group_mask], metric_result[f_group_mask]

    return (_m_result, _f_result) if raw else (_m_result.mean(), _f_result.mean())


def compute_DP_across_random_samples(df,
                                     sens_attr,
                                     demo_group_field,
                                     dataset_name,
                                     metric,
                                     iterations=100,
                                     batch_size=64,
                                     seed=124):
    np.random.seed(seed)

    if not hasattr(compute_DP_across_random_samples, "generated_groups"):
        compute_DP_across_random_samples.generated_groups = {}

    df = df.sort_values(demo_group_field)
    max_user = df['user_id'].max() + 1

    n_users = 0
    demo_groups_order = []
    size_perc = np.zeros((2,), dtype=float)
    groups = np.zeros((2, max_user), dtype=int)
    for i, (dg, gr_df) in enumerate(df.groupby(demo_group_field)):
        gr_users = gr_df['user_id'].unique()
        groups[i, gr_users] = 1
        n_users += gr_users.shape[0]
        size_perc[i] = gr_users.shape[0]
        demo_groups_order.append(dg)
    size_perc /= n_users

    gr_data = np.zeros(max_user)
    for gr_users in groups:
        pos = gr_users.nonzero()[0]
        gr_data[pos] = df.set_index('user_id').loc[pos, metric].to_numpy()

    if (dataset_name, sens_attr) not in compute_DP_across_random_samples.generated_groups:
        compute_DP_across_random_samples.generated_groups[(dataset_name, sens_attr)] = np.zeros(
            (iterations, 2, max_user), dtype=np.bool_
        )

    return _compute_DP_random_samples(
        gr_data,
        groups,
        size_perc,
        compute_DP_across_random_samples.generated_groups[(dataset_name, sens_attr)],
        batch_size=batch_size,
        iterations=iterations
    ), demo_groups_order


@numba.jit(nopython=True, parallel=True)
def _compute_DP_random_samples(group_data, groups, size_perc, out_samples, batch_size=64, iterations=100):
    out = np.empty((iterations, 3), dtype=np.float32)
    check = out_samples.nonzero()[0].shape[0] == 0
    for i in numba.prange(iterations):
        if check:
            samples = np.zeros_like(groups, dtype=np.bool_)
            for gr_i in range(len(groups)):
                sample_size = round(batch_size * size_perc[gr_i])
                samples[gr_i][np.random.choice(groups[gr_i].nonzero()[0], sample_size, replace=False)] = True
            out_samples[i] = samples

        gr1_mean = group_data[out_samples[i, 0]].mean()
        gr2_mean = group_data[out_samples[i, 1]].mean()
        out[i] = [gr1_mean, gr2_mean, np.abs(gr1_mean - gr2_mean)]

    return out


def best_epoch_DP_across_samples(exp_path,
                                 dataset,
                                 sens_attr,
                                 config,
                                 metric='ndcg',
                                 on_orig=False,
                                 iterations=100,
                                 seed=124,
                                 overwrite=False):
    """

    :param exp_path:
    :param dataset:
    :param sens_attr:
    :param config:
    :param metric:
    :param on_orig: if True computes DP on original recommendations
    :param iterations:
    :param seed:
    :param overwrite: if True it overwrites the metadata saved inside the function to compute DP
    :return:
    """
    exp_key = list(exp_path.keys())[0]
    evaluator = Evaluator(config)
    uid_field = dataset.uid_field

    pref_data_all, result_all = extract_best_metrics(
        exp_path, 'auto', evaluator, dataset, config=config
    )
    result = result_all[exp_key if not on_orig else exp_key.replace('+FairDP', '')][metric][:, -1]

    user_df = pd.DataFrame(
        {key: dataset.user_feat[key].numpy() for key in dataset.user_feat}
    )[[uid_field, sens_attr]]
    user_df[sens_attr] = user_df[sens_attr].map(dataset.field2id_token[sens_attr].__getitem__)
    user_df.rename(columns={sens_attr: 'Demo Group'}, inplace=True)

    metric_df = pd.DataFrame(
        zip(pref_data_all[exp_key][uid_field], result),
        columns=[uid_field, metric]
    )
    metric_df = metric_df.join(user_df.set_index(uid_field), on=uid_field)

    if overwrite and hasattr(compute_DP_across_random_samples, "generated_groups"):
        if (dataset.dataset_name, sens_attr) in compute_DP_across_random_samples.generated_groups:
            del compute_DP_across_random_samples.generated_groups[(dataset.dataset_name, sens_attr)]

    return compute_DP_across_random_samples(
        metric_df, sens_attr, 'Demo Group', dataset.dataset_name, metric,
        iterations=iterations, batch_size=config['user_batch_exp'], seed=seed
    )


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
                result_per_epoch[e_type][epoch][m_idx][metric] = compute_metric(evaluator, rec_data, m_df, 'cf_topk_pred', metric)[:, -1].mean()
                result_per_epoch[e_type][epoch][f_idx][metric] = compute_metric(evaluator, rec_data, f_df, 'cf_topk_pred', metric)[:, -1].mean()

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
