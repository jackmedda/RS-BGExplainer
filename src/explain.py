import os
import gc
import argparse
import logging
import inspect
import pickle
import json
from logging import getLogger

import yaml
import torch
import tqdm
import scipy
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from recbole.config import Config
from recbole.data import create_dataset, data_preparation, save_split_dataloaders, load_split_dataloaders
from recbole.utils import init_logger, get_model, get_trainer, init_seed, set_color

from utils.utils import damerau_levenshtein_distance
from explainers.explainer import BGExplainer


def load_data_and_model(model_file):
    r"""Load filtered dataset, split dataloaders and saved model.
    Args:
        model_file (str): The path of saved model file.
    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    checkpoint = torch.load(model_file)
    config = checkpoint['config']

    with open(args.explainer_config_file, 'r', encoding='utf-8') as f:
        explain_config_dict = yaml.load(f.read(), Loader=config.yaml_loader)
    config.final_config_dict.update(explain_config_dict)

    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)

    # specifying tot_item_num to the number of unique items makes the dataloader for evaluation to be batched
    # on interactions of one user at a time
    config['eval_batch_size'] = dataset.item_num
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))

    return config, model, dataset, train_data, valid_data, test_data


def explain(config, model, test_data, epochs, topk=10, dist_type="damerau_levenshtein", **kwargs):
    iter_data = (
        tqdm.tqdm(
            test_data,
            total=len(test_data),
            ncols=100,
            desc=set_color(f"Explaining   ", 'pink'),
        )
    )

    exps = {}
    pref_data = []
    for batch_idx, batched_data in enumerate(iter_data):
        gc.collect()
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             if 'cuda' in obj.device.type:
        #                 import pdb; pdb.set_trace()
        #                 print(type(obj), obj.size())
        #     except Exception as e:
        #         print(e)
        user_id = batched_data[0].interaction[model.USER_ID].squeeze()
        bge = BGExplainer(config, train_data.dataset, model, user_id, dist=dist_type, **kwargs)
        exp = bge.explain(batched_data, epochs, topk=topk)
        del bge
        exps[user_id.item()] = exp
        if config['explain_fairness'] and exp:
            pref_data.append([user_id.item(), exp[0][2]])

    exps_file = f'{"fair_" if config["explain_fairness"] else ""}{config["dataset"]}_exps_explain_{epochs}epochs.pkl'
    with open(exps_file, 'wb') as f:
        pickle.dump(exps, f)

    data = [exp[:-3] + [exp[-1]] for exp_list in exps.values() for exp in exp_list]
    data = list(map(lambda x: [x[0].item(), *x[1:]], data))
    df = pd.DataFrame(
        data,
        columns=['uid', 'topk', 'cf_topk_pred', 'topk_dist', 'loss_total', 'loss_pred', 'loss_dist',
                 'fair_loss', 'n_edges', 'first_fair_loss']
    )

    df.to_csv(
        f'{"fair_" if config["explain_fairness"] else ""}{config["dataset"]}_exps_explain_{epochs}epochs.csv',
        index=False
    )

    return exps, pd.DataFrame(pref_data, columns=['user_id', 'topk'])


def hops_analysis(config, model, test_data, epochs, topk=10, dist_type="damerau_levenshtein", diam=None):
    if diam is None:
        graph = get_adj_matrix(config, dataset)
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
                exps.append([n_hops, user_id.item(), len(set(exp[0][1]) & set(exp[0][2])), exp[0][3], exp[0][-8]//2])
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


def get_adj_matrix(config, dataset):
    USER_ID = config['USER_ID_FIELD']
    ITEM_ID = config['ITEM_ID_FIELD']
    n_users = dataset.num(USER_ID)
    n_items = dataset.num(ITEM_ID)
    inter_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
    num_all = n_users + n_items

    A = scipy.sparse.dok_matrix((num_all, num_all), dtype=np.float32)
    inter_M = inter_matrix
    inter_M_t = inter_matrix.transpose()
    data_dict = dict(zip(zip(inter_M.row, inter_M.col + n_users), [1] * inter_M.nnz))
    data_dict.update(dict(zip(zip(inter_M_t.row + n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
    A._update(data_dict)
    A = A.tocoo()
    return nx.Graph(A)


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

            
def generate_bias_ratio(_train_data, sensitive_attrs=None, history_matrix: pd.DataFrame = None, mapped_keys=False):
    sensitive_attrs = ['gender', 'age'] if sensitive_attrs is None else sensitive_attrs
    user_df = pd.DataFrame({
        'user_id': _train_data.dataset.user_feat['user_id'].numpy(),
        **{sens_attr: _train_data.dataset.user_feat[sens_attr].numpy() for sens_attr in sensitive_attrs}
    })

    item_df = pd.DataFrame({
        'item_id': _train_data.dataset.item_feat['item_id'].numpy(),
        'class': map(lambda x: [el for el in x if el != 0], _train_data.dataset.item_feat['class'].numpy().tolist())
    })

    sensitive_maps = [_train_data.dataset.field2id_token[sens_attr] for sens_attr in sensitive_attrs]
    item_categories_map = _train_data.dataset.field2id_token['class']
    
    if history_matrix is None:
        history_matrix, _, history_len = _train_data.dataset.history_item_matrix()
        history_matrix, history_len = history_matrix.numpy(), history_len.numpy()
    else:
        topk = len(history_matrix.iloc[0]['topk'])
        history_len = np.full((user_df.shape[0]), topk)

        history_df = user_df[['user_id']].join(history_matrix.set_index('user_id'))
        history_topk = history_df['topk']
        history_df['topk'] = history_topk.apply(lambda x: np.zeros(topk, dtype=int) if np.isnan(x).all() else x)

        history_matrix = np.stack(history_df['topk'].values)

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
        ax = sns.heatmap(df)
        ax.set_title(attr.title())
        plt.tight_layout()
        plt.savefig(os.path.join(plots_path, f'{attr}.png'))
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', default=None)
    parser.add_argument('--explainer_config_file', default='../config/gcmc_explainer.yaml')
    parser.add_argument('--hops_analysis', action='store_true')

    args = parser.parse_args()

    script_path = os.path.abspath(os.path.dirname(inspect.getsourcefile(lambda: 0)))

    if args.model_file is None:
        raise FileNotFoundError("need to specify a saved file with `--model_file`")

    print(args)

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(args.model_file)

    train_bias_ratio = None
    if config['explain_fairness']:
        train_bias_ratio = generate_bias_ratio(
            train_data,
            sensitive_attrs=config['sensitive_attributes'],
            mapped_keys=False if args.hops_analysis else True
        )

    if args.hops_analysis:
        filepath_bias = f'{config["dataset"]}_train_rec_bias_ratio_analysis.pkl'

        loaded_diam, diam_info = load_diameter_info()

        diam, pref_data = hops_analysis(config, model, test_data, config['cf_epochs'], diam=loaded_diam)
        update_diameter_info(loaded_diam, diam_info, diam)
    else:
        filepath_bias = f'{config["dataset"]}_epochs{config["cf_epochs"]}_train_rec_bias_ratio.pkl'

        exps_data, pref_data = explain(config, model, test_data, config['cf_epochs'], train_bias_ratio=train_bias_ratio)

    if not pref_data.empty:
        rec_bias_ratio = generate_bias_ratio(
            train_data,
            sensitive_attrs=config['sensitive_attributes'],
            history_matrix=pref_data,
            mapped_keys=False if args.hops_analysis else True
        )

        with open(filepath_bias, 'wb') as f:
            pickle.dump(
                {
                    'train_bias_ratio': train_bias_ratio,
                    'rec_bias_ratio': rec_bias_ratio
                },
                f
            )

        plot_bias_analysis_disparity(
            train_bias_ratio,
            rec_bias_ratio,
            train_data,
            item_categories=train_data.dataset.field2id_token['class']
        )
