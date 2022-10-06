# %%
import os
import inspect

import torch
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from recbole.evaluator import Evaluator

import src.utils as utils
import src.explainers as explainers

script_path = os.path.abspath(os.path.dirname(inspect.getsourcefile(lambda: 0)))
script_path = os.path.join(script_path, 'src') if 'src' not in script_path else script_path


# %%
def get_plots_path(dataset, model_name, epochs, sens_attrs):
    plots_path = os.path.join(
        script_path,
        os.pardir,
        f'dp_plots',
        dataset.dataset_name,
        model_name,
        'comparison_orig_pert',
        f"epochs_{epochs}",
        '_'.join(sens_attrs),
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

    # if type(data) is str:
    #     text = data
    # else:
    #     # * is p < 0.05
    #     # ** is p < 0.005
    #     # *** is p < 0.0005
    #     # etc.
    #     text = ''
    #     p_005 = '*'
    #     p_001 = '^'
    #     p = .05
    #
    #     text = p_001 if data < p else (p_005 if data < p else '')
    #
    #     if len(text) == 0:
    #         text = 'n. s.'

    text = f"Diff: {diff:.4f}"

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


def plot_barplot_orig_pert(orig_m_ndcg, orig_f_ndcg, pert_m_ndcg, pert_f_ndcg, user_data, plots_path="", **kwargs):
    model_name, pert_model_name = kwargs.get("model_name", ""), kwargs.get("pert_model_name", "")
    pert_model_file = kwargs.get("pert_model_file", "_perturbed")

    df = pd.DataFrame(
        zip(
            np.concatenate([orig_m_ndcg, orig_f_ndcg, pert_m_ndcg, pert_f_ndcg]),
            np.tile(["M"] * orig_m_ndcg.shape[0] + ["F"] * orig_f_ndcg.shape[0], 2),
            ["Original"] * user_data.shape[0] + ["Perturbed"] * user_data.shape[0]
        ),
        columns=["NDCG", "Group", "Graph Type"]
    )

    ax = sns.barplot(x="Graph Type", y="NDCG", hue="Group", order=["Original", "Perturbed"], hue_order=["M", "F"], data=df)
    for cp_bars, graph_type in zip([[0, 2], [1, 3]], ["Original", "Perturbed"]):
        gt_df = df[df["Graph Type"] == graph_type]
        m_vals, f_vals = gt_df.loc[gt_df["Group"] == "M", "NDCG"].values, gt_df.loc[gt_df["Group"] == "F", "NDCG"].values
        stat = stats.f_oneway(m_vals, f_vals)
        mean_diff = abs(m_vals.mean() - f_vals.mean())

        barplot_annotate_brackets(
            0, 1, mean_diff, stat.pvalue,
            [ax.patches[i].get_x() + ax.patches[i].get_width() / 2 for i in cp_bars],
            [ax.patches[i].get_height() for i in cp_bars],
            yerr=[np.abs(np.nan_to_num(np.diff(ax.lines[i].get_ydata(True)), nan=0)) / 2 for i in cp_bars],
            dh=0.03,
            barh=0.01
        )

    ax.minorticks_on()
    ax.grid(axis='y', ls=':')

    plt.savefig(os.path.join(plots_path, f"{model_name}_orig_{pert_model_name}_{pert_model_file}.png"))
    plt.tight_layout()
    plt.close()


def prepare_data(config, model, pert_model, dataset, pert_dataset, train_data, test_data, topk=10, **kwargs):
    model.eval()
    pert_model.eval()

    pert_model_file = kwargs.get("perturbed_model_file", "_perturbed")

    model_name = model.__class__.__name__
    pert_model_name = pert_model.__class__.__name__

    gender_map = dataset.field2id_token['gender']
    female_idx, male_idx = (gender_map == 'F').nonzero()[0][0], (gender_map == 'M').nonzero()[0][0]
    evaluator = Evaluator(config)

    sens_attrs, epochs, batch_exp = config['sensitive_attributes'], config['cf_epochs'], config['user_batch_exp']

    user_df = pd.DataFrame({
        'user_id': train_data.dataset.user_feat['user_id'].numpy(),
        **{sens_attr: train_data.dataset.user_feat[sens_attr].numpy() for sens_attr in sens_attrs}
    })

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

    test_orig_males_ndcg = utils.compute_metric(
        evaluator,
        test_data.dataset,
        orig_pref_df.set_index('user_id').loc[user_df.loc[user_df['gender'] == male_idx, 'user_id']].reset_index(),
        'topk_pred',
        'ndcg'
    )[:, -1]
    test_orig_females_ndcg = utils.compute_metric(
        evaluator,
        test_data.dataset,
        orig_pref_df.set_index('user_id').loc[user_df.loc[user_df['gender'] == female_idx, 'user_id']].reset_index(),
        'topk_pred',
        'ndcg'
    )[:, -1]

    test_pert_males_ndcg = utils.compute_metric(
        evaluator,
        test_data.dataset,
        pert_pref_df.set_index('user_id').loc[user_df.loc[user_df['gender'] == male_idx, 'user_id']].reset_index(),
        'topk_pred',
        'ndcg'
    )[:, -1]
    test_pert_females_ndcg = utils.compute_metric(
        evaluator,
        test_data.dataset,
        pert_pref_df.set_index('user_id').loc[user_df.loc[user_df['gender'] == female_idx, 'user_id']].reset_index(),
        'topk_pred',
        'ndcg'
    )[:, -1]

    plot_barplot_orig_pert(
        test_orig_males_ndcg,
        test_orig_females_ndcg,
        test_pert_males_ndcg,
        test_pert_females_ndcg,
        user_data,
        model_name=model_name,
        pert_model_name=pert_model_name,
        plots_path=get_plots_path(dataset, model_name, epochs, sens_attrs),
        pert_model_file=pert_model_file
    )
