import os
import gc
import re
import argparse
import inspect
import pickle
import json

import torch
import tqdm
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from recbole.utils import set_color
from recbole.data.interaction import Interaction

import utils
from explainers.explainer import BGExplainer


def load_already_done_exps_user_id(base_exps_file):
    files = [f for f in os.listdir(base_exps_file) if re.match(r'user_\d+', f) is not None]

    return [int(_id) for f in files for _id in f.split('_')[1].split('.')[0].split('#')]


def get_base_exps_filepath(config):
    epochs = config['cf_epochs']
    base_exps_file = os.path.join(script_path, 'explanations', config['dataset'])
    if config['explain_fairness']:
        fair_metadata = "_".join(config["sensitive_attributes"])
        fair_loss = 'FairNDCGApprox' if config["explain_fairness_NDCGApprox"] else 'FairBD'
        base_exps_file = os.path.join(base_exps_file, fair_loss, fair_metadata, f"epochs_{epochs}")
    else:
        base_exps_file = os.path.join(base_exps_file, 'pred_explain', f"epochs_{epochs}")

    explain_scope = config['explain_scope']
    exp_scopes = ['group', 'individual']
    if explain_scope not in exp_scopes:
        raise ValueError(f"`{explain_scope}` is not in {exp_scopes}")
    base_exps_file = os.path.join(base_exps_file, explain_scope)

    return base_exps_file


def save_exps_df(base_exps_file, exps):
    data = [exp[:-3] + exp[-2:] for exp_list in exps.values() for exp in exp_list]
    data = list(map(lambda x: [x[0], *x[1:]], data))
    df = pd.DataFrame(
        data,
        columns=utils.EXPS_COLUMNS[:-3] + utils.EXPS_COLUMNS[-2:]
    )

    out_path = base_exps_file.split(os.sep)
    df.to_csv(
        f'{"_".join(out_path[out_path.index("explanations"):])}.csv',
        index=False
    )


def explain(config, model, test_data, base_exps_file, topk=10, **kwargs):
    epochs = config['cf_epochs']

    # iter_data = (
    #     tqdm.tqdm(
    #         test_data,
    #         total=len(test_data),
    #         ncols=100,
    #         desc=set_color(f"Explaining   ", 'pink'),
    #     )
    # )
    if not os.path.exists(base_exps_file):
        os.makedirs(base_exps_file)

    user_data = test_data.user_df[config['USER_ID_FIELD']][torch.randperm(test_data.user_df[config['USER_ID_FIELD']].shape[0])]

    loaded_user_ids = load_already_done_exps_user_id(base_exps_file)
    user_data = user_data[~torch.isin(user_data, torch.tensor(loaded_user_ids))]

    user_data = user_data.split(config['user_batch_exp'])
    iter_data = (
        tqdm.tqdm(
            user_data,
            total=len(user_data),
            ncols=100,
            desc=set_color(f"Explaining   ", 'pink'),
        )
    )

    if not config["explain_fairness"]:
        kwargs['train_bias_ratio'] = None

    with open(os.path.join(base_exps_file, "config.pkl"), 'wb') as config_file:
        pickle.dump(config, config_file)

    for batch_idx, batched_user in enumerate(iter_data):
        # user_id = batched_data[0].interaction[model.USER_ID]
        user_id = batched_user
        user_df_mask = torch.isin(test_data.user_df[test_data.uid_field], user_id)
        user_df = Interaction({k: v[user_df_mask] for k, v in test_data.user_df.interaction.items()})
        history_item = test_data.uid2history_item[user_id]

        if len(user_id) > 1:
            history_u = torch.cat([torch.full_like(hist_iid, i) for i, hist_iid in enumerate(history_item)])
            history_i = torch.cat(list(history_item))
        else:
            history_u = torch.full_like(history_item, 0)
            history_i = history_item
        batched_data = (user_df, (history_u, history_i), None, None)

        gc.collect()
        bge = BGExplainer(config, train_data.dataset, model, user_id, dist=config['cf_dist'], **kwargs)
        exp = bge.explain(batched_data, epochs, topk=topk)
        del bge

        if len(user_id) == 1:
            exps_file_user = os.path.join(base_exps_file, f"user_{user_id[0].item()}.pkl")
        else:
            exps_file_user = os.path.join(base_exps_file, f"user_{'#'.join(user_id.numpy())}.pkl")

        with open(exps_file_user, 'wb') as f:
            pickle.dump(exp, f)


def hops_analysis(config, model, test_data, topk=10, dist_type="damerau_levenshtein", diam=None):
    if diam is None:
        graph = utils.get_nx_adj_matrix(config, dataset)
        max_cc_graph = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()
        # cc_graphs = [graph.subgraph(x).copy() for x in nx.connected_components(graph)]
        # for x in cc_graphs:
        #     print(x.number_of_nodes(), x.number_of_edges(), nx.diameter(x))
        diam = nx.diameter(max_cc_graph)

    iter_data = (
        tqdm.tqdm(
            test_data,
            total=len(test_data),
            ncols=100,
            desc=set_color(f"Explaining   ", 'pink'),
        )
    )

    config['explainer_force_return'] = True
    config['only_subgraph'] = True

    hops_info = 'neighbors_' if config['neighbors_hops'] else ''
    stop_hops = (diam + 1) // 2 + 1 if config['neighbors_hops'] else diam + 1

    fair_analysis = config['explain_fairness']

    exps = []
    pref_data = []
    for batch_idx, batched_data in enumerate(iter_data):
        user_id = batched_data[0].interaction[model.USER_ID][0]
        for n_hops in range(1, stop_hops):
            config['n_hops'] = n_hops

            # if the diam is odd and the current n_hops is the last, then use normal hops
            if config['neighbors_hops'] and diam % 2 == 1 and n_hops == (stop_hops - 1):
                config['n_hops'] = diam
                config['neighbors_hops'] = False

            try:
                bge = BGExplainer(config, train_data.dataset, model, user_id, dist=dist_type)
                exp = bge.explain(batched_data, 1, topk=topk)
                exps.append([n_hops, user_id.item(), len(set(exp[0][1]) & set(exp[0][2])), exp[0][3], exp[0][-2] // 2])
                if fair_analysis and n_hops == (stop_hops - 1):
                    pref_data.append([user_id.item(), exp[0][1]])
            except Exception as e:
                print(e)
                pass

        if diam % 2 == 1:
            config['neighbors_hops'] = True if hops_info == 'neighbors_' else False  # reset if diam is odd

    df = pd.DataFrame(exps, columns=['n_hops', 'uid', 'intersection', 'edit_dist', 'n_edges'])

    df.to_csv(f'{config["dataset"]}_exps_test_1_epoch_{hops_info}hops_all_users.csv', index=False)

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    inters_box = sns.boxplot(x='n_hops', y='intersection', data=df, ax=axs[0])
    inters_box.set_title('Intersection')

    edit_dist_box = sns.boxplot(x='n_hops', y='edit_dist', data=df, ax=axs[1])
    edit_dist_box.set_title('Edit Distance')

    df['n_edges'] = df['n_edges'] / df[df['n_hops'] == stop_hops - 1].iloc[0]['n_edges']
    edges_lineplot = sns.lineplot(x='n_hops', y='n_edges', data=df, ax=axs[2])
    edges_lineplot.set_title('Edges Percentage')
    axs[2].set_xticks(range(1, stop_hops))

    fig.savefig(f'{config["dataset"]}_{hops_info}hops_analysis_boxplot_inters_edit.png')
    plt.close()

    return diam, pd.DataFrame(pref_data, columns=['user_id', 'topk'])


def load_diameter_info():
    _diam_info = {}
    diam_info_path = os.path.join(script_path, 'dataset', 'diam_info.json')
    if os.path.exists(diam_info_path):
        if os.stat(diam_info_path).st_size == 0:
            os.remove(diam_info_path)

        with open(diam_info_path, 'r') as f:
            _diam_info = json.load(f)

    return _diam_info.get(config['dataset'], None), _diam_info


def update_diameter_info(_loaded_diam, _diam_info, _diam):
    diam_info_path = os.path.join(script_path, 'dataset', 'diam_info.json')

    if _loaded_diam is None:
        _diam_info[config['dataset']] = _diam
        with open(diam_info_path, 'w') as f:
            json.dump(_diam_info, f)


def compute_user_preference(_hist_matrix, _item_df, n_categories=None):
    _item_df = _item_df.set_index('item_id')

    if n_categories is None:
        n_categories = _item_df['class'].to_numpy().flatten().max() + 1

    _pref_matrix = torch.zeros((_hist_matrix.shape[0], n_categories))

    for user_id, user_hist in enumerate(_hist_matrix):
        user_hist = user_hist[user_hist != 0]
        if len(user_hist) > 0:
            for item in user_hist:
                _pref_matrix[user_id, _item_df.loc[item, 'class']] += 1

    return _pref_matrix


def compute_uniform_categories_prob(_item_df, _item_categories_map):
    uni_cat_prob = np.zeros(_item_categories_map.shape)
    for cat_list in _item_df['class']:
        if cat_list:
            uni_cat_prob[cat_list] += 1

    return uni_cat_prob / (_item_df.shape[0] - 1)


def generate_bias_ratio(_train_data,
                        sensitive_attrs=None,
                        history_matrix: pd.DataFrame = None,
                        pred_col='cf_topk_pred',
                        mapped_keys=False):
    sensitive_attrs = ['gender', 'age'] if sensitive_attrs is None else sensitive_attrs
    user_df = pd.DataFrame({
        'user_id': _train_data.dataset.user_feat[config['USER_ID_FIELD']].numpy(),
        **{sens_attr: _train_data.dataset.user_feat[sens_attr].numpy() for sens_attr in sensitive_attrs}
    })

    item_df = pd.DataFrame({
        'item_id': _train_data.dataset.item_feat[config['ITEM_ID_FIELD']].numpy(),
        'class': map(lambda x: [el for el in x if el != 0], _train_data.dataset.item_feat['class'].numpy().tolist())
    })

    sensitive_maps = [_train_data.dataset.field2id_token[sens_attr] for sens_attr in sensitive_attrs]
    item_categories_map = _train_data.dataset.field2id_token['class']

    if history_matrix is None:
        history_matrix, _, history_len = _train_data.dataset.history_item_matrix()
        history_matrix, history_len = history_matrix.numpy(), history_len.numpy()
    else:
        clean_history_matrix(history_matrix)
        topk = len(history_matrix.iloc[0][pred_col])
        history_len = np.zeros(user_df.shape[0])
        history_len[history_matrix['user_id']] = topk
        history_df = user_df[['user_id']].join(history_matrix.set_index('user_id'))
        history_topk = history_df[pred_col]
        history_df[pred_col] = history_topk.apply(lambda x: np.zeros(topk, dtype=int) if np.isnan(x).all() else x)

        history_matrix = np.stack(history_df[pred_col].values)

    pref_matrix = compute_user_preference(history_matrix, item_df, n_categories=len(item_categories_map))
    uniform_categories_prob = compute_uniform_categories_prob(item_df, item_categories_map)

    bias_ratio = dict.fromkeys(sensitive_attrs)
    for attr, group_map in zip(sensitive_attrs, sensitive_maps):
        bias_ratio[attr] = dict.fromkeys(range(len(group_map)))
        for demo_group, demo_df in user_df.groupby(attr):
            gr_idx = demo_group if mapped_keys else group_map[demo_group]

            if group_map[demo_group] == '[PAD]':
                bias_ratio[attr][gr_idx] = None
                continue

            group_users = demo_df['user_id'].to_numpy()
            group_pref = pref_matrix[group_users, :].sum(dim=0)
            group_history_len = history_len[group_users].sum()
            bias_ratio[attr][gr_idx] = group_pref / group_history_len
            bias_ratio[attr][gr_idx] /= uniform_categories_prob

    return bias_ratio


def plot_bias_analysis_disparity(train_bias, rec_bias, _train_data, item_categories=None):
    plots_path = os.path.join(script_path, os.pardir,
                              f'bias_disparity_plots{"_analysis" if args.hops_analysis else ""}', config['dataset'])
    if config['explain_fairness']:
        fair_metadata = "_".join(config["sensitive_attributes"])
        fair_loss = 'FairNDCGApprox' if config["explain_fairness_NDCGApprox"] else 'FairBD'
        plots_path = os.path.join(plots_path, fair_loss, fair_metadata)
    else:
        plots_path = os.path.join(plots_path, 'pred_explain' if not args.bias_orig_pred else 'original_pred')
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

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


def clean_history_matrix(hist_m):
    for col in ['topk_pred', 'cf_topk_pred']:
        if isinstance(hist_m.iloc[0][col], str):
            hist_m[col] = hist_m[col].map(lambda x: np.array(x[1:-1].strip().split(), int))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', required=True)
    parser.add_argument('--explainer_config_file', default='../config/gcmc_explainer.yaml')
    parser.add_argument('--hops_analysis', action='store_true')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--best_exp_col', nargs="+", default=["loss_total"])
    parser.add_argument('--bias_orig_pred', action='store_true')

    args = parser.parse_args()

    script_path = os.path.abspath(os.path.dirname(inspect.getsourcefile(lambda: 0)))

    if args.model_file is None:
        raise FileNotFoundError("need to specify a saved file with `--model_file`")

    print(args)

    config, model, dataset, train_data, valid_data, test_data = utils.load_data_and_model(args.model_file,
                                                                                          args.explainer_config_file)

    train_bias_ratio = generate_bias_ratio(
        train_data,
        sensitive_attrs=config['sensitive_attributes'],
        mapped_keys=False if args.hops_analysis else True
    )

    if args.hops_analysis:
        loaded_diam, diam_info = load_diameter_info()

        diam, pref_data = hops_analysis(config, model, test_data, diam=loaded_diam)
        update_diameter_info(loaded_diam, diam_info, diam)
    else:
        base_exps_filepath = get_base_exps_filepath(config)

        if not args.load:
            explain(config, model, test_data, base_exps_filepath, train_bias_ratio=train_bias_ratio)
        else:
            with open(os.path.join(base_exps_filepath, "config.pkl"), 'rb') as config_file:
                config = pickle.load(config_file)

        exps_data = utils.load_exps_file(base_exps_filepath)
        save_exps_df(base_exps_filepath, exps_data)

        top_exp_col = [utils.EXPS_COLUMNS.index(be) for be in
                       args.best_exp_col] if args.best_exp_col is not None else None

        pref_data = []
        for user_id, user_exps in exps_data.items():
            u_exps = user_exps
            if top_exp_col is not None and user_exps:
                for tec in top_exp_col:
                    u_exps = sorted(u_exps, key=lambda x: x[tec])
                u_exps = [u_exps[0]]
            if u_exps:
                pref_data.append([user_id, u_exps[0][1], u_exps[0][2]])

        pref_data = pd.DataFrame(pref_data, columns=['user_id', 'topk_pred', 'cf_topk_pred'])

    if not pref_data.empty:
        rec_bias_ratio = generate_bias_ratio(
            train_data,
            pred_col='topk_pred' if args.bias_orig_pred else 'cf_topk_pred',
            sensitive_attrs=config['sensitive_attributes'],
            history_matrix=pref_data,
            mapped_keys=False if args.hops_analysis else True
        )

        plot_bias_analysis_disparity(
            train_bias_ratio,
            rec_bias_ratio,
            train_data,
            item_categories=train_data.dataset.field2id_token['class']
        )
    else:
        print("Pref Data is empty!")
