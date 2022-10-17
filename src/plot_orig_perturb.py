# %%
import os
import inspect

import torch
import numpy as np
import pandas as pd
import networkx as nx
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from recbole.evaluator import Evaluator

import src.utils as utils
import src.explainers as explainers
import src.utils.plot_utils as plot_utils

script_path = os.path.abspath(os.path.dirname(inspect.getsourcefile(lambda: 0)))
script_path = os.path.join(script_path, 'src') if 'src' not in script_path else script_path


# %%
def get_plots_path(dataset, model_name, epochs, sens_attr):
    plots_path = os.path.join(
        script_path,
        os.pardir,
        f'dp_plots',
        dataset.dataset_name,
        model_name,
        'comparison_orig_pert',
        f"epochs_{epochs}",
        sens_attr,
    )

    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    return plots_path


def barplot_annotate_brackets(num1, num2, diff, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
    """
    https://stackoverflow.com/a/52333561

    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    p_005 = '*'
    p_001 = '^'

    text = f"Diff: {diff:.6f}"
    text += p_001 if data < 0.01 else (p_005 if data < 0.05 else '')

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y0 = ly + dh
    y1 = ry + dh

    barhl = dict.fromkeys([0, 1])
    if y0 > y1:
        barhl[1] = y0 + barh - y1
        barhl[0] = barh
    else:
        barhl[0] = y1 + barh - y0
        barhl[1] = barh

    barx = [lx, lx, rx, rx]
    bary = [y0, y0+barhl[0], y1+barhl[1], y1]
    mid = ((lx+rx)/2, max(y0, y1)+barh)

    plt.plot(barx, bary, c='black')
    plt.plot([lx + 0.001, rx - 0.001], [max(y0, y1), max(y0, y1)], c='black', ls='--', lw=0.6)

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)


def plot_barplot_orig_pert(orig_m_ndcg, orig_f_ndcg, pert_m_ndcg, pert_f_ndcg, user_data, sens_attr, plots_path="", **kwargs):
    model_name, pert_model_name = kwargs.get("model_name", ""), kwargs.get("pert_model_name", "")
    pert_model_file = kwargs.get("pert_model_file", "_perturbed")

    if sens_attr == 'age':
        m_idx, f_idx = "Y", "O"
    else:
        m_idx, f_idx = "M", "F"

    df = pd.DataFrame(
        zip(
            np.concatenate([orig_m_ndcg, orig_f_ndcg, pert_m_ndcg, pert_f_ndcg]),
            np.tile([m_idx] * orig_m_ndcg.shape[0] + [f_idx] * orig_f_ndcg.shape[0], 2),
            ["Original"] * user_data.shape[0] + ["Perturbed"] * user_data.shape[0]
        ),
        columns=["NDCG", "Group", "Graph Type"]
    )

    ax = sns.barplot(x="Graph Type", y="NDCG", hue="Group", order=["Original", "Perturbed"], hue_order=[m_idx, f_idx], data=df)
    x_tick_mean = []
    for cp_bars, graph_type in zip([[0, 2], [1, 3]], ["Original", "Perturbed"]):
        gt_df = df[df["Graph Type"] == graph_type]
        m_vals, f_vals = gt_df.loc[gt_df["Group"] == m_idx, "NDCG"].values, gt_df.loc[gt_df["Group"] == f_idx, "NDCG"].values
        stat = stats.mannwhitneyu(m_vals, f_vals)
        mean_diff = abs(m_vals.mean() - f_vals.mean())

        barplot_annotate_brackets(
            0, 1, mean_diff, stat.pvalue,
            [ax.patches[i].get_x() + ax.patches[i].get_width() / 2 for i in cp_bars],
            [ax.patches[i].get_height() for i in cp_bars],
            yerr=[np.abs(np.nan_to_num(np.diff(ax.lines[i].get_ydata(True)), nan=0)) / 2 for i in cp_bars],
            dh=0.03,
            barh=0.01
        )
        x_tick_mean.append(graph_type + f" {mean_diff:.6f}")

    ax.set_xticklabels(x_tick_mean)
    ax.minorticks_on()
    ax.grid(axis='y', ls=':')

    plt.savefig(os.path.join(plots_path, f"{model_name}_orig_{pert_model_name}_{pert_model_file}.png"))
    plt.tight_layout()
    plt.close()


def graph_statistics(pert_config,
                     orig_train_dataset,
                     orig_valid_dataset,
                     orig_test_dataset,
                     pert_dataset,
                     orig_model_name,
                     sens_attr,
                     c_id,
                     exp_type,
                     exp_value,
                     short_head=0.05):
    orig_model_name = os.path.splitext(os.path.basename(orig_model_name))[0].split('-')[0] if '-' in orig_model_name else orig_model_name

    group_name_map = {
        "M": "Males",
        "F": "Females",
        "Y": "Younger",
        "O": "Older"
    }
    m_label, f_label = ("M", "F") if sens_attr == "gender" else ("Y", "O")
    m_idx, f_idx = (orig_train_dataset.dataset.field2token_id[sens_attr][lab] for lab in ["M", "F"])

    user_feat = orig_train_dataset.dataset.user_feat
    user_df = pd.DataFrame({'user_id': user_feat['user_id'].numpy(), sens_attr: user_feat[sens_attr].numpy()})

    m_group, f_group = (user_feat[sens_attr] == m_idx).nonzero().T[0].numpy() - 1,\
                       (user_feat[sens_attr] == f_idx).nonzero().T[0].numpy() - 1

    sens_attr_map = dict(zip(np.concatenate([m_group, f_group]), [m_label] * len(m_group) + [f_label] * len(f_group)))

    _, _, orig_item_pop = orig_train_dataset.dataset.history_user_matrix()
    _, _, pert_item_pop = pert_dataset.dataset.history_user_matrix()

    orig_item_pop, pert_item_pop = orig_item_pop[1:].numpy(), pert_item_pop[1:].numpy()

    orig_item_pop, pert_item_pop = np.argsort(orig_item_pop)[::-1], np.argsort(pert_item_pop)[::-1]

    orig_sh_n, pert_sh_n = round(len(orig_item_pop) * short_head), round(len(pert_item_pop) * short_head)
    orig_short_head, orig_long_tail = np.split(orig_item_pop, [orig_sh_n])
    pert_short_head, pert_long_tail = np.split(pert_item_pop, [pert_sh_n])

    orig_sh_pop = dict(zip(
        np.concatenate([orig_short_head, orig_long_tail]),
        ["Short Head"] * len(orig_short_head) + ["Long Tail"] * len(orig_long_tail)
    ))
    pert_sh_pop = dict(zip(
        np.concatenate([pert_short_head, pert_long_tail]),
        ["Short Head"] * len(pert_short_head) + ["Long Tail"] * len(pert_long_tail)
    ))

    evaluator = Evaluator(pert_config)
    edge_additions = pert_config['edge_additions']
    cf_epochs = pert_config['cf_epochs']
    exp_rec_data = pert_config['exp_rec_data']
    delete_adv_group = pert_config['delete_adv_group']
    rec_data = locals()[f"orig_{exp_rec_data}_dataset"]

    exp_path = os.path.join(script_path, 'dp_ndcg_explanations', orig_train_dataset.dataset.dataset_name,
                            orig_model_name, "FairDP", sens_attr, f"epochs_{cf_epochs}", str(c_id))

    # Does not matter which explanation we take if we evaluate just the recommendations of the original model
    exp_rec_df, rec_result_data = plot_utils.extract_best_metrics(
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
    print(orig_train_dataset.dataset.field2id_token[sens_attr])
    suptitle = f"{'Addition' if edge_additions else 'Deletions'} of Edges Connected to " \
               f"{group_name_map[orig_train_dataset.dataset.field2id_token[sens_attr][group_edge_del]]}"

    orig_nx = utils.get_nx_biadj_matrix(orig_train_dataset.dataset, remove_first_row_col=True)
    pert_nx = utils.get_nx_biadj_matrix(pert_dataset.dataset, remove_first_row_col=True)

    orig_top = {n for n, d in orig_nx.nodes(data=True) if d["bipartite"] == 0}
    orig_bottom = set(orig_nx) - orig_top
    pert_top = {n for n, d in pert_nx.nodes(data=True) if d["bipartite"] == 0}
    pert_bottom = set(pert_nx) - pert_top

    orig_node_type_map = dict(zip(orig_top, ["users"] * len(orig_top)))
    pert_node_type_map = dict(zip(pert_top, ["users"] * len(pert_top)))
    orig_node_type_map.update(dict(zip(orig_bottom, ["items"] * len(orig_bottom))))
    pert_node_type_map.update(dict(zip(pert_bottom, ["items"] * len(pert_bottom))))

    orig_centr = nx.bipartite.degree_centrality(orig_nx, orig_top)
    pert_centr = nx.bipartite.degree_centrality(pert_nx, pert_top)

    orig_top, orig_bottom = list(orig_top), list(orig_bottom)
    pert_top, pert_bottom = list(pert_top), list(pert_bottom)
    orig_df_data = zip(orig_top + orig_bottom, [orig_centr[n] for n in (orig_top + orig_bottom)])
    pert_df_data = zip(pert_top + pert_bottom, [pert_centr[n] for n in (pert_top + pert_bottom)])

    orig_df = pd.DataFrame(orig_df_data, columns=["node_id_minus_1", "Centrality"])
    pert_df = pd.DataFrame(pert_df_data, columns=["node_id_minus_1", "Centrality"])

    orig_df["Node Type"] = orig_df["node_id_minus_1"].map(orig_node_type_map)
    pert_df["Node Type"] = pert_df["node_id_minus_1"].map(pert_node_type_map)

    orig_user_df, orig_item_df = orig_df[orig_df["Node Type"] == "users"], orig_df[orig_df["Node Type"] == "items"]
    pert_user_df, pert_item_df = pert_df[pert_df["Node Type"] == "users"], pert_df[pert_df["Node Type"] == "items"]

    orig_user_df["Group"] = orig_user_df["node_id_minus_1"].map(sens_attr_map)
    pert_user_df["Group"] = pert_user_df["node_id_minus_1"].map(sens_attr_map)

    orig_item_df["Popularity"] = orig_item_df["node_id_minus_1"].map(lambda x: orig_sh_pop[int(x) - len(user_feat) + 1])
    pert_item_df["Popularity"] = pert_item_df["node_id_minus_1"].map(lambda x: pert_sh_pop[int(x) - len(user_feat) + 1])

    orig_user_df["Graph Type"], pert_user_df["Graph Type"] = "Original", "Perturbed"
    user_df = pd.concat([orig_user_df, pert_user_df], ignore_index=True)

    orig_item_df["Graph Type"], pert_item_df["Graph Type"] = "Original", "Perturbed"
    item_df = pd.concat([orig_item_df, pert_item_df], ignore_index=True)

    fig, axs = plt.subplots(2, 2, sharey=True)
    sns.kdeplot(x="Centrality", data=user_df[user_df["Group"] == m_label], hue="Graph Type", ax=axs[0, 0])
    sns.kdeplot(x="Centrality", data=user_df[user_df["Group"] == f_label], hue="Graph Type", ax=axs[0, 1])

    axs[0, 0].set_title(group_name_map[m_label])
    axs[0, 1].set_title(group_name_map[f_label])

    sns.kdeplot(x="Centrality", data=item_df[item_df["Popularity"] == "Short Head"], hue="Graph Type", ax=axs[1, 0])
    sns.kdeplot(x="Centrality", data=item_df[item_df["Popularity"] == "Long Tail"], hue="Graph Type", ax=axs[1, 1])

    axs[1, 0].set_title("Short Head")
    axs[1, 1].set_title("Long Tail")

    fig.suptitle(suptitle)

    fig.tight_layout()
    fig.savefig(os.path.join(
        get_plots_path(orig_train_dataset.dataset, orig_model_name, pert_config['cf_epochs'], sens_attr),
        f"kdeplot_orig_perturbed_{orig_model_name}_{c_id}__{exp_type}_{exp_value}.png"
    ))
    plt.close()


def prepare_data(config, model, pert_model, dataset, pert_dataset, train_data, test_data, topk=10, **kwargs):
    model.eval()
    pert_model.eval()

    pert_model_file = kwargs.get("perturbed_model_file", "_perturbed")

    model_name = model.__class__.__name__
    pert_model_name = pert_model.__class__.__name__

    epochs, batch_exp = config['cf_epochs'], config['user_batch_exp']
    sens_attr = pert_model_file.split('(')[1].split('_')[0]

    user_df = pd.DataFrame({
        'user_id': train_data.dataset.user_feat['user_id'].numpy(),
        sens_attr: train_data.dataset.user_feat[sens_attr].numpy()
    })

    attr_map = dataset.field2id_token[sens_attr]
    f_idx, m_idx = (attr_map == 'F').nonzero()[0][0], (attr_map == 'M').nonzero()[0][0]
    evaluator = Evaluator(config)

    orig_tot_item_num = dataset.item_num
    orig_item_tensor = dataset.get_item_feature().to(model.device)
    orig_test_batch_size = orig_tot_item_num

    pert_tot_item_num = pert_dataset.item_num
    pert_item_tensor = pert_dataset.get_item_feature().to(model.device)
    pert_test_batch_size = pert_tot_item_num

    exp_object = explainers.DPBGExplainer

    if dataset.field2id_token[dataset.iid_field].shape[0] >= pert_dataset.field2id_token[pert_dataset.iid_field].shape[0]:
        orig_item_data = torch.tensor([dataset.field2token_id[dataset.iid_field][i] for i in
                                       pert_dataset.field2token_id[pert_dataset.iid_field]])
        pert_item_data = torch.tensor(list(pert_dataset.field2token_id[pert_dataset.iid_field].values()))
    else:
        orig_item_data = torch.tensor(list(dataset.field2token_id[dataset.iid_field].values()))
        pert_item_data = torch.tensor([pert_dataset.field2token_id[pert_dataset.iid_field][i] for i in
                                       dataset.field2token_id[dataset.iid_field]])

    user_data = test_data.user_df[test_data.uid_field][torch.randperm(test_data.user_df.length)]
    orig_batched_data = exp_object.prepare_batched_data(user_data, test_data, item_data=orig_item_data)
    pert_batched_data = exp_object.prepare_batched_data(user_data, test_data, item_data=pert_item_data)

    orig_scores_args = [orig_batched_data, orig_tot_item_num, orig_test_batch_size, orig_item_tensor]
    pert_scores_args = [pert_batched_data, pert_tot_item_num, pert_test_batch_size, pert_item_tensor]
    topk_args = {'topk': topk}

    with torch.no_grad():
        orig_model_scores = exp_object.get_scores(model, *orig_scores_args, pred=None).detach().cpu()
        orig_model_scores_topk, orig_model_topk_idx = exp_object.get_top_k(orig_model_scores, **topk_args)

        pert_model_scores = exp_object.get_scores(pert_model, *pert_scores_args, pred=None).detach().cpu()
        pert_model_scores_topk, pert_model_topk_idx = exp_object.get_top_k(pert_model_scores, **topk_args)

    orig_pref_df = pd.DataFrame(zip(user_data.numpy(), orig_model_topk_idx), columns=['user_id', 'topk_pred'])
    pert_pref_df = pd.DataFrame(zip(user_data.numpy(), pert_model_topk_idx), columns=['user_id', 'topk_pred'])

    test_orig_m_ndcg = utils.compute_metric(
        evaluator,
        test_data.dataset,
        orig_pref_df.set_index('user_id').loc[user_df.loc[user_df[sens_attr] == m_idx, 'user_id']].reset_index(),
        'topk_pred',
        'ndcg'
    )[:, -1]
    test_orig_f_ndcg = utils.compute_metric(
        evaluator,
        test_data.dataset,
        orig_pref_df.set_index('user_id').loc[user_df.loc[user_df[sens_attr] == f_idx, 'user_id']].reset_index(),
        'topk_pred',
        'ndcg'
    )[:, -1]

    test_pert_m_ndcg = utils.compute_metric(
        evaluator,
        test_data.dataset,
        pert_pref_df.set_index('user_id').loc[user_df.loc[user_df[sens_attr] == m_idx, 'user_id']].reset_index(),
        'topk_pred',
        'ndcg'
    )[:, -1]
    test_pert_f_ndcg = utils.compute_metric(
        evaluator,
        test_data.dataset,
        pert_pref_df.set_index('user_id').loc[user_df.loc[user_df[sens_attr] == f_idx, 'user_id']].reset_index(),
        'topk_pred',
        'ndcg'
    )[:, -1]

    plot_barplot_orig_pert(
        test_orig_m_ndcg,
        test_orig_f_ndcg,
        test_pert_m_ndcg,
        test_pert_f_ndcg,
        user_data,
        sens_attr,
        model_name=model_name,
        pert_model_name=pert_model_name,
        plots_path=get_plots_path(dataset, model_name, epochs, sens_attr),
        pert_model_file=pert_model_file
    )
