import os
import gc
import re
import argparse
import inspect
import pickle
import json
from typing import Iterable

import torch
import tqdm
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from recbole.utils import set_color
from recbole.data.interaction import Interaction

import src.utils as utils
from src.explainers.explainer import BGExplainer


def load_already_done_exps_user_id(base_exps_file):
    """
    Only used for `individual` or 'group' explanations. It prevents the code from re-explaining already explained users.
    :param base_exps_file:
    :return:
    """
    files = [f for f in os.listdir(base_exps_file) if re.match(r'user_\d+', f) is not None]

    return [int(_id) for f in files for _id in f.split('_')[1].split('.')[0].split('#')]


def get_base_exps_filepath(config, config_id=-1):
    """
    return the filepath where explanations are saved
    :param config:
    :param config_id:
    :return:
    """
    epochs = config['cf_epochs']
    base_exps_file = os.path.join(script_path, 'explanations', config['dataset'])
    if config['explain_fairness']:
        fair_metadata = "_".join(config["sensitive_attributes"])
        fair_loss = 'FairNDCGApprox' if config["explain_fairness_NDCGApprox"] else 'FairBD'
        base_exps_file = os.path.join(base_exps_file, fair_loss, fair_metadata, f"epochs_{epochs}")
    else:
        base_exps_file = os.path.join(base_exps_file, 'pred_explain', f"epochs_{epochs}")

    explain_scope = config['explain_scope']
    exp_scopes = ['group', 'individual', 'group_explain']
    if explain_scope not in exp_scopes:
        raise ValueError(f"`{explain_scope}` is not in {exp_scopes}")
    base_exps_file = os.path.join(base_exps_file, explain_scope)

    if os.path.exists(base_exps_file):
        if config_id == -1:
            paths_c_ids = sorted(os.listdir(base_exps_file), key=int)
            for path_c in sorted(os.listdir(base_exps_file), key=int):
                config_path = os.path.join(base_exps_file, path_c, 'config.pkl')
                if os.path.exists(config_path):
                    with open(config_path, 'rb') as f:
                        _c = pickle.load(f)
                    if config.final_config_dict == _c.final_config_dict:
                        base_exps_file = os.path.join(base_exps_file, str(i))
                        break

            config_id = 1 if len(paths_c_ids) == 0 else max(paths_c_ids, key=int)

        base_exps_file = os.path.join(base_exps_file, str(config_id))
    else:
        base_exps_file = os.path.join(base_exps_file, "1")

    return base_exps_file


def save_exps_df(base_exps_file, exps):
    """
    Saves the pandas dataframe representation of explanations
    :param base_exps_file:
    :param exps:
    :return:
    """
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


def group_explain(config, model, test_data, base_exps_file, **kwargs):
    """
    Function that explains in `group_explain` mode, that is generating perturbed graphs for each group of users (or
    entire group of users). It is different from the `group` mode because this function explains each group even though
    the training procedure learns from the considered group in batches, while the `group` mode explain each batch. Then,
    `group_explain` is used to explain big groups.
    :param config:
    :param model:
    :param test_data:
    :param base_exps_file:
    :param kwargs:
    :return:
    """
    epochs = config['cf_epochs']
    topk = config['cf_topk']

    if not os.path.exists(base_exps_file):
        os.makedirs(base_exps_file)

    attr_group = config['group_explain_by']
    if attr_group is not None:
        user_df = pd.DataFrame({
            'user_id': train_data.dataset.user_feat[config['USER_ID_FIELD']].numpy(),
            attr_group: train_data.dataset.user_feat[attr_group].numpy()
        })
        user_data = {}
        for demo_gr, demo_df in user_df.groupby(attr_group):
            if demo_gr != 0:
                user_data[demo_gr] = torch.tensor(demo_df[config['USER_ID_FIELD']].to_numpy())
                user_data[demo_gr] = user_data[demo_gr][torch.randperm(user_data[demo_gr].shape[0])]
    else:
        user_data = test_data.user_df[config['USER_ID_FIELD']][torch.randperm(test_data.user_df[config['USER_ID_FIELD']].shape[0])]

    if not config["explain_fairness"]:
        kwargs['train_bias_ratio'], kwargs['train_pref_ratio'] = None, None

    with open(os.path.join(base_exps_file, "config.pkl"), 'wb') as config_file:
        pickle.dump(config, config_file)

    with open(os.path.join(base_exps_filepath, "config.json"), 'w') as config_json:
        json.dump(config.final_config_dict, config_json, indent=4, default=lambda x: str(x))

    sensitive_map = train_data.dataset.field2id_token[attr_group] if attr_group is not None else None
    if isinstance(user_data, dict):
        for attr in user_data:
            bge = BGExplainer(config, train_data.dataset, model, user_data, dist=config['cf_dist'], **kwargs)
            exp, scores = bge.explain((user_data[attr], test_data), epochs, topk=topk)
            del bge

            exps_file_user = os.path.join(base_exps_file, f"{attr_group}_users({sensitive_map[attr]}).pkl")

            with open(exps_file_user, 'wb') as f:
                pickle.dump(exp, f)
    else:
        bge = BGExplainer(config, train_data.dataset, model, user_data, dist=config['cf_dist'], **kwargs)
        exp, _ = bge.explain((user_data, test_data), epochs, topk=topk)
        del bge

        exps_file_user = os.path.join(base_exps_file, f"all_users.pkl")

        with open(exps_file_user, 'wb') as f:
            pickle.dump(exp, f)


def explain(config, model, test_data, base_exps_file, **kwargs):
    """
    This function explains for `individual` and `group` mode. Check *group_explain* function to see the difference
    between `group` and `group_explain`.
    :param config:
    :param model:
    :param test_data:
    :param base_exps_file:
    :param kwargs:
    :return:
    """
    epochs = config['cf_epochs']
    topk = config['cf_topk']

    if not os.path.exists(base_exps_file):
        os.makedirs(base_exps_file)

    user_data = test_data.user_df[config['USER_ID_FIELD']][torch.randperm(test_data.user_df[config['USER_ID_FIELD']].shape[0])]

    loaded_user_ids = load_already_done_exps_user_id(base_exps_file)
    user_data = user_data[~(user_data[..., None] == torch.tensor(loaded_user_ids)).any(-1)]

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
        kwargs['train_bias_ratio'], kwargs['train_pref_ratio'] = None, None

    loaded_scores, field2token_id = None, None
    loaded_scores_path = os.path.join(base_exps_file, os.path.splitext(args.model_file)[0] + '.pkl')
    if os.path.exists(loaded_scores_path):
        with open(loaded_scores_path, 'rb') as scores_file:
            loaded_scores, field2id_token = pickle.load(scores_file)

    with open(os.path.join(base_exps_file, "config.pkl"), 'wb') as config_file:
        pickle.dump(config, config_file)

    with open(os.path.join(base_exps_filepath, "config.json"), 'w') as config_json:
        json.dump(config.final_config_dict, config_json, indent=4, default=lambda x: str(x))

    orig_scores = {}
    for batch_idx, batched_user in enumerate(iter_data):
        # user_id = batched_data[0].interaction[model.USER_ID]
        user_id = batched_user
        user_df_mask = (test_data.user_df[test_data.uid_field][..., None] == user_id).any(-1)
        user_df = Interaction({k: v[user_df_mask] for k, v in test_data.user_df.interaction.items()})
        history_item = test_data.uid2history_item[user_id]

        if len(user_id) > 1:
            history_u = torch.cat([torch.full_like(hist_iid, i) for i, hist_iid in enumerate(history_item)])
            history_i = torch.cat(list(history_item))

            loaded_scores, field2id_token = None, None  # not supported
        else:
            history_u = torch.full_like(history_item, 0)
            history_i = history_item

        batched_data = (user_df, (history_u, history_i), None, None)

        gc.collect()
        bge = BGExplainer(config, train_data.dataset, model, user_id, dist=config['cf_dist'], **kwargs)
        exp, scores = bge.explain(batched_data, epochs, topk=topk, loaded_scores=loaded_scores, old_field2token_id=field2token_id)
        del bge

        if loaded_scores is None and len(user_id) == 1:
            orig_scores[user_id[0].item()] = scores[0]

        if len(user_id) == 1:
            exps_file_user = os.path.join(base_exps_file, f"user_{user_id[0].item()}.pkl")
        else:
            exps_file_user = os.path.join(base_exps_file, f"user_{'#'.join(map(str, user_id.numpy()))}.pkl")

        with open(exps_file_user, 'wb') as f:
            pickle.dump(exp, f)

    if loaded_scores is None:
        with open(loaded_scores_path, 'wb') as scores_file:
            pickle.dump((orig_scores, dataset.field2token_id[model.ITEM_ID_FIELD]), scores_file)


def hops_analysis(config, model, topk=10, dist_type="damerau_levenshtein", diam=None):
    """
    Functions used to make analysis when different number of hops is used, that is perturb subgraphs of different sizes.
    :param config:
    :param model:
    :param topk:
    :param dist_type:
    :param diam:
    :return:
    """
    if diam is None:
        graph = utils.get_nx_adj_matrix(config, dataset)
        max_cc_graph = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()
        # cc_graphs = [graph.subgraph(x).copy() for x in nx.connected_components(graph)]
        # for x in cc_graphs:
        #     print(x.number_of_nodes(), x.number_of_edges(), nx.diameter(x))
        diam = nx.diameter(max_cc_graph)

    user_data = test_data.user_df[config['USER_ID_FIELD']][torch.randperm(test_data.user_df[config['USER_ID_FIELD']].shape[0])]

    user_data = user_data.split(1)
    iter_data = (
        tqdm.tqdm(
            user_data,
            total=len(user_data),
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
    for batch_idx, batched_user in enumerate(iter_data):
        user_id = batched_user
        user_df_mask = (test_data.user_df[test_data.uid_field][..., None] == user_id).any(-1)
        user_df = Interaction({k: v[user_df_mask] for k, v in test_data.user_df.interaction.items()})
        history_item = test_data.uid2history_item[user_id]

        history_u = torch.full_like(history_item, 0)
        history_i = history_item

        batched_data = (user_df, (history_u, history_i), None, None)
        for n_hops in range(1, stop_hops):
            config['n_hops'] = n_hops

            # if the diam is odd and the current n_hops is the last, then use normal hops
            if config['neighbors_hops'] and diam % 2 == 1 and n_hops == (stop_hops - 1):
                config['n_hops'] = diam
                config['neighbors_hops'] = False

            try:
                bge = BGExplainer(config, train_data.dataset, model, user_id, dist=dist_type)
                exp, _ = bge.explain(batched_data, 1, topk=topk)
                exps.append([n_hops, user_id.item(), len(set(exp[0][1][0]) & set(exp[0][2][0])), exp[0][3], exp[0][-2] // 2])
                if fair_analysis and n_hops == (stop_hops - 1):
                    pref_data.append([user_id.item(), exp[0][1][0]])
            except Exception as e:
                print(e)
                pass

        if diam % 2 == 1:
            config['neighbors_hops'] = True if hops_info == 'neighbors_' else False  # reset if diam is odd

    df = pd.DataFrame(exps, columns=['n_hops', 'uid', 'intersection', 'edit_dist', 'n_edges'])

    df.to_csv(f'{config["dataset"]}_exps_test_1_epoch_{hops_info}hops_all_users.csv', index=False)

    pref_data = pd.DataFrame(pref_data, columns=['user_id', 'topk_pred'])
    pref_data.to_csv(os.path.join(script_path, 'pref_data_GCMC_original.csv'), index=None)

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

    return diam, pref_data


def load_diameter_info():
    """
    Load the diameter of the graph of the current dataset.
    :return:
    """
    _diam_info = {}
    diam_info_path = os.path.join(script_path, '../dataset', 'diam_info.json')
    if os.path.exists(diam_info_path):
        if os.stat(diam_info_path).st_size == 0:
            os.remove(diam_info_path)

        with open(diam_info_path, 'r') as f:
            _diam_info = json.load(f)

    return _diam_info.get(config['dataset'], None), _diam_info


def update_diameter_info(_loaded_diam, _diam_info, _diam):
    """
    Updates the dictionary (json format) that stores the diameter info of each dataset.
    :param _loaded_diam:
    :param _diam_info:
    :param _diam:
    :return:
    """
    diam_info_path = os.path.join(script_path, '../dataset', 'diam_info.json')

    if _loaded_diam is None:
        _diam_info[config['dataset']] = _diam
        with open(diam_info_path, 'w') as f:
            json.dump(_diam_info, f)


def plot_bias_analysis_disparity(train_bias, rec_bias, _train_data, item_categories=None):
    """
    Used to plot bias disparity as an heatmap when hops analysis is performed
    :param train_bias:
    :param rec_bias:
    :param _train_data:
    :param item_categories:
    :return:
    """
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

    bias_disparity = utils.compute_bias_disparity(train_bias, rec_bias, _train_data)
    for attr in bias_disparity:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', required=True)
    parser.add_argument('--explainer_config_file', default=os.path.join("config", "gcmc_explainer.yaml"))
    parser.add_argument('--hops_analysis', action='store_true')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--load_config_id', default=-1)
    parser.add_argument('--best_exp_col', default="loss_total")
    parser.add_argument('--bias_orig_pred', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    script_path = os.path.abspath(os.path.dirname(inspect.getsourcefile(lambda: 0)))

    if args.model_file is None:
        raise FileNotFoundError("need to specify a saved file with `--model_file`")

    print(args)

    # load trained model, config, dataset
    config, model, dataset, train_data, valid_data, test_data = utils.load_data_and_model(args.model_file,
                                                                                          args.explainer_config_file)

    import pdb; pdb.set_trace()

    map_cat = None
    filter_plot_str = ""
    cats_vs_all = config['cats_vs_all']
    if cats_vs_all is not None:
        if not isinstance(cats_vs_all, Iterable):
            cats_vs_all = [cats_vs_all]
        if isinstance(cats_vs_all[0], str):
            cats_vs_all = [(dataset.field2id_token['class'] == cat).nonzero().item() for cat in cats_vs_all]
        elif not isinstance(cats_vs_all[0], int):
            raise NotImplementedError(f"one_cat_vs_all = {cats_vs_all} is not supported")

        if 0 in cats_vs_all:
            raise ValueError("cats_vs_all cannot be or contain zero")

        item_df = pd.DataFrame({
            'item_id': train_data.dataset.item_feat['item_id'].numpy(),
            'class': map(lambda x: [el for el in x if el != 0], train_data.dataset.item_feat['class'].numpy().tolist())
        })

        map_cat = dict(zip(cats_vs_all + [-1], list(range(1, len(cats_vs_all) + 2))))
        item_df['class'] = item_df['class'].map(lambda x: np.unique([map_cat[cat] if cat in map_cat else map_cat[-1] for cat in x if cat != 0]))
        n_max_cat = max(map(len, item_df['class'].tolist()))
        item_cats = np.stack([np.pad(x, (0, n_max_cat - len(x)), mode='constant', constant_values=0) for x in item_df['class']])
        item_cats = torch.tensor(item_cats, dtype=int)

        filter_plot_str = f"_vs_all({'_'.join(map(str, cats_vs_all))})"
    elif config['filter_categories'] is not None and cats_vs_all is None:
        filter_cats = config['filter_categories']
        item_df = pd.DataFrame({
            'item_id': train_data.dataset.item_feat['item_id'].numpy(),
            'class': map(lambda x: [el for el in x if el != 0], train_data.dataset.item_feat['class'].numpy().tolist())
        })

        if config['filter_categories_mode'] is None or config['filter_categories_mode'] == "random":
            item_df['class'] = item_df['class'].map(lambda x: np.random.permutation(x)[:filter_cats])
        elif config['filter_categories_mode'] == "first":
            item_df['class'] = item_df['class'].map(lambda x: x[:1])
        item_cats = np.stack([np.pad(x, (0, filter_cats - len(x)), mode='constant', constant_values=0) for x in item_df['class']])
        item_cats = torch.tensor(item_cats, dtype=int)

        filter_plot_str = f"_filter({filter_cats})"
    else:
        item_cats = dataset.item_feat['class']

    n_cats = item_cats.max() + 1

    heat_data = []
    bar_data = []
    for x in range(n_cats):
        x_cats = item_cats[(item_cats == x).any(dim=1)]
        cat_count = torch.bincount(x_cats.flatten(), minlength=n_cats).numpy() / x_cats.shape[0]
        heat_data.append(cat_count)
        bar_data.append(x_cats[(x_cats == torch.tensor([x] + [0] * (x_cats.shape[1] - 1))).all(dim=1), :].shape[0] / x_cats.shape[0])
    heat_data = np.stack(heat_data).round(2)
    heat_data[:, 0] = 0
    heat_data[0] = 0

    g = sns.JointGrid(height=12, space=0.5)
    g.ax_marg_x.remove()
    if map_cat is not None:
        class_names = dataset.field2id_token['class'][[0] + list(map_cat.values())]
    else:
        class_names = dataset.field2id_token['class']
    bar_df = pd.DataFrame(zip(class_names, bar_data))
    sns.barplot(x=1, y=0, data=bar_df, ax=g.ax_marg_y, color="black")
    g.ax_marg_y.plot([1., 1.], g.ax_marg_y.get_ylim(), 'k--')
    sns.heatmap(
        pd.DataFrame(heat_data, index=class_names, columns=class_names),
        ax=g.ax_joint,
        center=0,
        linewidths=.5,
        annot=True
    )

    g.figure.tight_layout()
    plt.savefig(f"{config['dataset']}{filter_plot_str}_cats_share_distribution.png", bbox_inches="tight")
    plt.close()

    bias_pref_kwargs = dict(
        item_cats=item_cats if (config['filter_categories'] is not None or config['cats_vs_all'] is not None) else None
    )

    # measure bias ratio in training set
    if config['target_scope'] == 'group':
        train_bias_ratio, train_pref_ratio = utils.generate_bias_ratio(
            train_data,
            config,
            sensitive_attrs=config['sensitive_attributes'],
            mapped_keys=False,
            **bias_pref_kwargs
        )
    elif config['target_scope'] == 'steck':
        train_bias_ratio = None
        train_pref_ratio = utils.generate_steck_pref_ratio(
            train_data,
            config,
            sensitive_attrs=config['sensitive_attributes'],
            mapped_keys=False,
            **bias_pref_kwargs
        )
    elif config['target_scope'] == 'individual':
        train_bias_ratio, train_pref_ratio = utils.generate_individual_bias_ratio(
            train_data,
            config,
            **bias_pref_kwargs
        )
    else:
        raise NotImplementedError(f"target_scope = `{config['target_scope']}` is not supported")

    item_df = pd.DataFrame({
        'item_id': train_data.dataset.item_feat[config['ITEM_ID_FIELD']].numpy(),
        'class': map(lambda x: [el for el in x if el != 0], item_cats.numpy().tolist())
    })
    item_categories_map = train_data.dataset.field2id_token['class']
    cat_sharing_prob = utils.compute_category_sharing_prob(item_df, len(item_categories_map))
    cat_intersharing = utils.compute_category_intersharing_distribution(item_df, len(item_categories_map))
    attr_cat_distrib = utils.get_category_inter_distribution_over_attrs(
        train_data,
        config['sensitive_attributes'],
        norm=True,
        **bias_pref_kwargs
    )

    if args.hops_analysis:
        loaded_diam, diam_info = load_diameter_info()

        diam, pref_data = hops_analysis(config, model, diam=loaded_diam)
        update_diameter_info(loaded_diam, diam_info, diam)
    else:
        base_exps_filepath = get_base_exps_filepath(config, config_id=args.load_config_id)

        if not args.load:
            kwargs = dict(
                train_bias_ratio=train_bias_ratio,
                train_pref_ratio=train_pref_ratio,
                cat_sharing_prob=cat_sharing_prob,
                cat_intersharing_distrib=cat_intersharing,
                attr_cat_distrib=attr_cat_distrib,
                verbose=args.verbose
            )

            if config['filter_categories'] is not None:
                kwargs['item_cats'] = item_cats

            if not config['group_explain']:
                explain(
                    config,
                    model,
                    test_data,
                    base_exps_filepath,
                    **kwargs
                )
            else:
                group_explain(
                    config,
                    model,
                    test_data,
                    base_exps_filepath,
                    **kwargs
                )
        else:
            with open(os.path.join(base_exps_filepath, "config.pkl"), 'rb') as config_file:
                config = pickle.load(config_file)

        if config['filter_categories'] is not None or map_cat is not None:
            torch.save(item_cats, os.path.join(base_exps_filepath, 'item_cats.pt'))

        if map_cat is not None:
            with open(os.path.join(base_exps_filepath, 'map_cats.pkl'), 'wb') as f:
                pickle.dump(map_cat, f)

        exps_data = utils.load_exps_file(base_exps_filepath)
        save_exps_df(base_exps_filepath, exps_data)

        top_exp_col = utils.EXPS_COLUMNS.index(args.best_exp_col) if args.best_exp_col is not None else None

        pref_data = []
        for user_id, user_exps in exps_data.items():
            u_exps = user_exps
            if top_exp_col is not None and user_exps:
                u_exps = sorted(u_exps, key=lambda x: x[top_exp_col])
                u_exps = [u_exps[0]]
            if u_exps:
                pref_data.append([user_id, u_exps[0][1].squeeze(), u_exps[0][2].squeeze()])

        pref_data = pd.DataFrame(pref_data, columns=['user_id', 'topk_pred', 'cf_topk_pred'])

    if not pref_data.empty:
        rec_bias_ratio, rec_pref_ratio = utils.generate_bias_ratio(
            train_data,
            config,
            pred_col='topk_pred' if args.bias_orig_pred else 'cf_topk_pred',
            sensitive_attrs=config['sensitive_attributes'],
            history_matrix=pref_data,
            mapped_keys=False if args.hops_analysis else True
        )

        plot_bias_path = plot_bias_analysis_disparity(
            train_bias_ratio,
            rec_bias_ratio,
            train_data,
            item_categories=train_data.dataset.field2id_token['class']
        )
    else:
        print("Pref Data is empty!")
