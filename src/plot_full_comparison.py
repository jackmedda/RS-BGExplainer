# %%
import os
import argparse
import inspect

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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


def update_plot_data(_test_df_data, _rec_df_data):
    test_orig_total_metric = best_test_exp_result[model_name]
    rec_orig_total_metric = best_rec_exp_result[model_name]

    test_pert_total_metric = best_test_exp_result[model_dp_s]
    rec_pert_total_metric = best_rec_exp_result[model_dp_s]

    m_group_mask = best_rec_exp_df[model_name].user_id.isin(user_df.loc[user_df[sens_attr] == m_idx, 'user_id'])
    f_group_mask = best_rec_exp_df[model_name].user_id.isin(user_df.loc[user_df[sens_attr] == f_idx, 'user_id'])

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

    _test_df_data.extend(list(zip(
        test_uid,
        user_df.set_index('user_id').loc[test_uid, sens_attr],
        [model_name] * len(test_uid),
        [dataset.dataset_name] * len(test_uid),
        [metric.upper()] * len(test_uid),
        test_orig_total_metric[:, -1],
        ["NoPolicy"] * len(test_uid)
    )))

    _test_df_data.extend(list(zip(
        test_uid,
        user_df.set_index('user_id').loc[test_uid, sens_attr],
        [model_name] * len(test_uid),
        [dataset.dataset_name] * len(test_uid),
        [metric.upper()] * len(test_uid),
        test_pert_total_metric[:, -1],
        [policy] * len(test_uid)
    )))

    _rec_df_data.extend(list(zip(
        rec_uid,
        user_df.set_index('user_id').loc[rec_uid, sens_attr],
        [model_name] * len(rec_uid),
        [dataset.dataset_name] * len(rec_uid),
        [metric.upper()] * len(rec_uid),
        rec_orig_total_metric[:, -1],
        ["NoPolicy"] * len(rec_uid)
    )))

    _rec_df_data.extend(list(zip(
        rec_uid,
        user_df.set_index('user_id').loc[rec_uid, sens_attr],
        [model_name] * len(rec_uid),
        [dataset.dataset_name] * len(rec_uid),
        [metric.upper()] * len(rec_uid),
        rec_pert_total_metric[:, -1],
        [policy] * len(rec_uid)
    )))

    return _group_edge_del


# %%
parser = argparse.ArgumentParser()
parser.add_argument('--model_files', nargs='+', required=True)
parser.add_argument('--explainer_config_files', required=True, nargs='+', type=str)
parser.add_argument('--th_edges_epochs', type=float, default=0.1)
parser.add_argument('--iterations', default=100)

args = parser.parse_args()

assert len(args.model_files) == len(args.explainer_config_files), \
    "Pass the same number of perturbed model files and configuration files to be loaded"

script_path = os.path.abspath(os.path.dirname(inspect.getsourcefile(lambda: 0)))
script_path = os.path.join(script_path, 'src') if 'src' not in script_path else script_path

print(args)

config_ids = [os.path.basename(os.path.dirname(p)) for p in args.explainer_config_files]

policy_map = {
    'increase_disparity': 'IncDisp',  # Increase Disparity
    'force_removed_edges': 'MonDel',   # Monotonic Deletions
    'group_deletion_constraint': 'DelCons'  # Deletion Constraint
}

# %%
axs = {}
test_df_data, rec_df_data = [], []
test_del_df_data, rec_del_df_data = [], []
for model_file, exp_config_file in zip(args.model_files, args.explainer_config_files):
    config, model, dataset, train_data, valid_data, test_data = utils.load_data_and_model(model_file, exp_config_file)

    model_name = model.__class__.__name__
    sens_attr, epochs, batch_exp = config['sensitive_attribute'], config['cf_epochs'], config['user_batch_exp']
    policy = '+'.join([pm for p, pm in policy_map.items() if config[p]])

    edge_additions = config['edge_additions']
    exp_rec_data = config['exp_rec_data']
    delete_adv_group = config['delete_adv_group']
    rec_data = locals()[f"{exp_rec_data}_data"]

    user_df = pd.DataFrame({
        'user_id': train_data.dataset.user_feat['user_id'].numpy(),
        sens_attr: train_data.dataset.user_feat[sens_attr].numpy()
    })

    attr_map = dataset.field2id_token[sens_attr]
    f_idx, m_idx = (attr_map == 'F').nonzero()[0][0], (attr_map == 'M').nonzero()[0][0]
    user_num, item_num = dataset.user_num, dataset.item_num
    evaluator = Evaluator(config)

    metrics = evaluator.metrics
    model_dp_s = f'{model_name}+FairDP'

    exp_paths = {model_dp_s: os.path.dirname(exp_config_file)}

    best_test_exp_df, best_test_exp_result = plot_utils.extract_best_metrics(
        exp_paths,
        'auto',
        evaluator,
        test_data.dataset,
        config=config
    )
    best_rec_exp_df, best_rec_exp_result = plot_utils.extract_best_metrics(
        exp_paths,
        'auto',
        evaluator,
        rec_data.dataset,
        config=config
    )

    test_uid = best_test_exp_df[model_dp_s]['user_id'].to_numpy()
    rec_uid = best_rec_exp_df[model_dp_s]['user_id'].to_numpy()

    all_exp_test_dfs, test_result_all_data, _, _ = plot_utils.extract_all_exp_metrics_data(
        exp_paths,
        train_data,
        test_data.dataset,
        evaluator,
        sens_attr,
        rec=False
    )

    all_exp_rec_dfs, rec_result_all_data, _, _ = plot_utils.extract_all_exp_metrics_data(
        exp_paths,
        train_data,
        rec_data.dataset,
        evaluator,
        sens_attr,
        rec=True
    )

    import pdb; pdb.set_trace()

    for metric in metrics:
        group_edge_del = update_plot_data(test_df_data, rec_df_data)


cols = ['user_id', sens_attr.title(), 'Model', 'Dataset', 'Metric', 'Value', 'Policy']
test_df = pd.DataFrame(test_df_data, columns=cols)
rec_df = pd.DataFrame(rec_df_data, columns=cols)

print(test_df)
print(rec_df)

datasets_list, models_list = test_df['Dataset'].unique().tolist(), test_df['Model'].unique().tolist()
plots_path = os.path.join(get_plots_path('_'.join(datasets_list), '_'.join(models_list)), '_'.join(config_ids), sens_attr)
if not os.path.exists(plots_path):
    os.makedirs(plots_path)

with open(os.path.join(plots_path, 'test_df.csv'), 'w') as f:
    f.write(f'# model_files {" ".join(args.model_files)}\n')
    f.write(f'# explainer_config_files {" ".join(args.explainer_config_files)}\n')
    test_df.to_csv(f)
with open(os.path.join(plots_path, 'rec_df.csv'), 'w') as f:
    f.write(f'# model_files {" ".join(args.model_files)}\n')
    f.write(f'# explainer_config_files {" ".join(args.explainer_config_files)}\n')
    rec_df.to_csv(f)

for df, exp_data_name in zip([test_df, rec_df], ["test", exp_rec_data]):
    for metric, metric_df in df.groupby("Metric"):
        plot_df_data = []
        y_col = f"{metric.upper()} DP"
        plot_columns = ["Model", "Dataset", "Policy", y_col]
        palette = dict(zip(np.concatenate([["NoPolicy"], df["Policy"].unique()]), sns.color_palette("colorblind")))
        for (_model, _dataset, _policy), sub_df in metric_df.groupby(["Model", "Dataset", "Policy"]):
            dp_samples = utils.compute_DP_across_random_samples(
                sub_df, sens_attr.title(), 'Value', batch_size=batch_exp, iterations=args.iterations
            )

            plot_df_data.extend(list(zip(
                [_model] * args.iterations,
                [_dataset] * args.iterations,
                [_policy] * args.iterations,
                dp_samples
            )))

        plot_df_line_gby = metric_df.groupby(["Dataset", "Model"])
        fig_line = plt.figure(figsize=(15, 15), constrained_layout=True)
        subfigs = fig_line.subfigures(len(datasets_list), 1)

        plot_df_bar = pd.DataFrame(plot_df_data, columns=plot_columns)
        plot_df_bar_gby = plot_df_bar.groupby("Dataset")
        fig_bar, axs_bar = plt.subplots(1, len(datasets_list), figsize=(10, 6))
        axs_bar = [axs_bar] if not isinstance(axs_bar, list) else axs_bar
        for i, (dset, ax_bar) in enumerate(zip(datasets_list, axs_bar)):
            dset_bar_df = plot_df_bar_gby.get_group(dset)
            sns.barplot(x="Model", y=y_col, data=dset_bar_df, hue="Policy", ax=ax_bar, palette=palette)
            ax_bar.set_title(dset.upper())

            subfigs[i].set_title(dset.upper())
            axs_line = subfigs[i].subplots(1, len(models_list))
            for m, ax_line in zip(models_list, axs_line):
                dset_model_line_df = plot_df_line_gby.get_group((dset, m))
                sns.barplot(x="DEL EDGES", y=y_col, data=dset_model_line_df, hue="Policy", ax=ax_line, palette=palette)
                ax_bar.set_title(dset.upper())

        fig_line.suptitle(sens_attr.title())
        fig_line.tight_layout()
        fig_line.savefig(os.path.join(plots_path, f"lineplot_{exp_data_name}_{metric}_DP_random_samples.png"))

        fig_bar.suptitle(sens_attr.title())
        fig_bar.tight_layout()
        fig_bar.savefig(os.path.join(plots_path, f"barplot_{exp_data_name}_{metric}_DP_random_samples.png"))
        plt.close("all")
