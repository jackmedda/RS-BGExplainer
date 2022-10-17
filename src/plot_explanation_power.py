# %%
import os
import pickle
import argparse
import inspect

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
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


# %%
parser = argparse.ArgumentParser()
parser.add_argument('--model_files', nargs='+', required=True)
parser.add_argument('--explainer_config_file', default=os.path.join("config", "explainer.yaml"))
parser.add_argument('--load_config_ids', nargs="+", type=str, required=True, help="ids of configurations/explanations")
parser.add_argument('--n_datasets', type=int, default=2)
parser.add_argument('--th_edges_epochs', type=float, default=0.1)

args = parser.parse_args()

assert len(args.model_files) == len(args.load_config_ids), \
    "Pass the same number of perturbed model files and configuration ids to be loaded"

script_path = os.path.abspath(os.path.dirname(inspect.getsourcefile(lambda: 0)))
script_path = os.path.join(script_path, 'src') if 'src' not in script_path else script_path

print(args)

# %%
axs = {}
colors_map = {}
model_files_names = []
markers = ['*', 's']
color_palette = sns.color_palette("colorblind")
fig = plt.figure(figsize=(10, 6))
for model_file, c_id in zip(args.model_files, args.load_config_ids):
    config, model, dataset, train_data, valid_data, test_data = utils.load_data_and_model(model_file,
                                                                                          args.explainer_config_file)

    if dataset.dataset_name not in axs:
        sub_kwargs = {'sharey': list(axs.values())[0]} if len(axs) > 0 else {}
        axs[dataset.dataset_name] = fig.add_subplot(1, args.n_datasets, len(axs) + 1, **sub_kwargs)

    model_name = model.__class__.__name__
    model_files_names.append(model_name)
    sens_attr, epochs, batch_exp = config['sensitive_attribute'], config['cf_epochs'], config['user_batch_exp']

    if model_name not in colors_map:
        colors_map[model_name] = color_palette[len(colors_map)]

    user_df = pd.DataFrame({
        'user_id': train_data.dataset.user_feat['user_id'].numpy(),
        sens_attr: train_data.dataset.user_feat[sens_attr].numpy()
    })

    attr_map = dataset.field2id_token[sens_attr]
    f_idx, m_idx = (attr_map == 'F').nonzero()[0][0], (attr_map == 'M').nonzero()[0][0]
    user_num, item_num = dataset.user_num, dataset.item_num
    evaluator = Evaluator(config)

    metrics_names = evaluator.metrics
    model_dp_s = f'{model_name}+FairDP'

    exp_paths = {
        model_dp_s: os.path.join(script_path, 'dp_ndcg_explanations', dataset.dataset_name, model_name, 'FairDP',
                                 sens_attr, f"epochs_{epochs}", c_id)
    }

    with open(os.path.join(exp_paths[model_dp_s], 'config.pkl'), 'rb') as f:
        exp_config = pickle.load(f)

    edge_additions = exp_config['edge_additions']
    exp_rec_data = exp_config['exp_rec_data']
    delete_adv_group = exp_config['delete_adv_group']
    rec_data = locals()[f"{exp_rec_data}_data"]

    all_exp_test_dfs, test_result_all_data, test_n_users_data_all, test_topk_dist_all = plot_utils.extract_all_exp_metrics_data(
        exp_paths,
        train_data,
        test_data.dataset,
        evaluator,
        sens_attr,
        rec=False
    )
    all_exp_rec_dfs, rec_result_all_data, rec_n_users_data_all, rec_topk_dist_all = plot_utils.extract_all_exp_metrics_data(
        exp_paths,
        train_data,
        rec_data.dataset,
        evaluator,
        sens_attr,
        rec=True
    )

    exp_test_df = all_exp_test_dfs[model_dp_s][all_exp_test_dfs[model_dp_s]["epoch"] == all_exp_test_dfs[model_dp_s]["epoch"].unique()[0]]
    exp_rec_df = all_exp_rec_dfs[model_dp_s][all_exp_rec_dfs[model_dp_s]["epoch"] == all_exp_rec_dfs[model_dp_s]["epoch"].unique()[0]]

    test_orig_total_ndcg = utils.compute_metric(evaluator, test_data.dataset, exp_test_df, 'topk_pred', 'ndcg')
    rec_orig_total_ndcg = utils.compute_metric(evaluator, rec_data.dataset, exp_rec_df, 'topk_pred', 'ndcg')

    test_orig_m_ndcg, test_orig_f_ndcg = utils.compute_metric_per_group(
        evaluator,
        test_data,
        user_df,
        all_exp_test_dfs[model_dp_s],
        sens_attr,
        (m_idx, f_idx),
        metric="ndcg"
    )
    rec_orig_m_ndcg, rec_orig_f_ndcg = utils.compute_metric_per_group(
        evaluator,
        rec_data,
        user_df,
        all_exp_rec_dfs[model_dp_s],
        sens_attr,
        (m_idx, f_idx),
        metric="ndcg"
    )

    if rec_orig_m_ndcg >= rec_orig_f_ndcg:
        if delete_adv_group is not None:
            group_edge_del = m_idx if delete_adv_group else f_idx
        else:
            group_edge_del = m_idx
    else:
        if delete_adv_group is not None:
            group_edge_del = f_idx if delete_adv_group else m_idx
        else:
            group_edge_del = f_idx

    df_over_epochs = all_exp_rec_dfs[model_dp_s].groupby(["epoch", "n_del_edges"]).first().reset_index().sort_values("epoch")
    df_over_epochs["Edges/Epochs"] = np.diff(df_over_epochs["n_del_edges"].values, prepend=[0]) / train_data.dataset.inter_num

    axs[dataset.dataset_name].plot(
        df_over_epochs["epoch"].values, df_over_epochs["fair_loss"].values, c=colors_map[model_name], label=model_name
    )

    for _, row in df_over_epochs[["epoch", "fair_loss", "Edges/Epochs"]].iterrows():
        axs[dataset.dataset_name].scatter(
            [row["epoch"]],
            [row["fair_loss"]],
            color=colors_map[model_name],
            marker=markers[0] if row["Edges/Epochs"] < args.th_edges_epochs else markers[1]
        )

marker_legend_labels = [f'< {args.th_edges_epochs}%', f'≥ {args.th_edges_epochs}%']
for d_name, ax in axs.items():
    title_proxy = Rectangle((0, 0), 0, 0, color='w')
    ls_legend_handles = [
        Line2D([0], [0], ls="", color='k', marker=markers[0], markersize=10),
        Line2D([0], [0], ls="", color='k', marker=markers[1], markersize=10)
    ]

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        [title_proxy] + handles + [title_proxy] + ls_legend_handles,
        ["Model"] + labels + ["Edges/Epochs"] + marker_legend_labels
    )

    ax.set_title(d_name)
    ax.set_xlabel('Epochs')
    ax.set_ylabel(f'{sens_attr.title()} ApproxNDCG Diff')

plots_path = os.path.join(
    get_plots_path('_'.join(list(axs.keys())), '_'.join(model_files_names)),
    'comparison',
    f"epochs_{epochs}",
    '_'.join(args.load_config_ids),
    sens_attr
)
if not os.path.exists(plots_path):
    os.makedirs(plots_path)

title = "Edge Additions " if edge_additions else "Edge Deletions "
if sens_attr == "gender":
    title += "of Males " if group_edge_del == m_idx else "of Females "
else:
    title += "of Younger " if group_edge_del == m_idx else "of Older "
title += "Optimized on " + f"{exp_rec_data.title()} Data"

fig.suptitle(title)
fig.tight_layout()
fig.savefig(os.path.join(plots_path, f'fair_diff_over_epochs_{sens_attr}.png'))
