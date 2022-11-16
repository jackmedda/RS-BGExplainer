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

# %%
axs = {}
test_df_data, rec_df_data = [], []
test_df_dp_data, rec_df_dp_data = [], []
for model_file, exp_config_file in zip(args.model_files, args.explainer_config_files):
    config, model, dataset, train_data, valid_data, test_data = utils.load_data_and_model(model_file, exp_config_file)

    model_name = model.__class__.__name__
    sens_attr, epochs, batch_exp = config['sensitive_attribute'], config['cf_epochs'], config['user_batch_exp']

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

    for metric in metrics:
        test_orig_total_metric = utils.compute_metric(evaluator, test_data.dataset, best_test_exp_df[model_dp_s], 'topk_pred', metric)
        rec_orig_total_metric = utils.compute_metric(evaluator, rec_data.dataset, best_rec_exp_df[model_dp_s], 'topk_pred', metric)

        rec_orig_m_metric, rec_orig_f_metric = utils.compute_metric_per_group(
            evaluator,
            rec_data,
            user_df,
            best_rec_exp_df[model_dp_s],
            sens_attr,
            (m_idx, f_idx),
            metric=metric
        )

        test_pert_total_metric = utils.compute_metric(evaluator, test_data.dataset, best_test_exp_df[model_dp_s], 'cf_topk_pred', metric)
        rec_pert_total_metric = utils.compute_metric(evaluator, rec_data.dataset, best_rec_exp_df[model_dp_s], 'cf_topk_pred', metric)

        if rec_orig_m_metric >= rec_orig_f_metric:
            if delete_adv_group is not None:
                group_edge_del = m_idx if delete_adv_group else f_idx
            else:
                group_edge_del = m_idx
        else:
            if delete_adv_group is not None:
                group_edge_del = f_idx if delete_adv_group else m_idx
            else:
                group_edge_del = f_idx

        test_df_data.extend(list(zip(
            test_uid,
            user_df.set_index('user_id').loc[test_uid, sens_attr],
            [model_name] * len(test_uid),
            [dataset.dataset_name] * len(test_uid),
            [metric.upper()] * len(test_uid),
            test_orig_total_metric[:, -1],
            ["Original"] * len(test_uid)
        )))

        test_df_data.extend(list(zip(
            test_uid,
            user_df.set_index('user_id').loc[test_uid, sens_attr],
            [model_name] * len(test_uid),
            [dataset.dataset_name] * len(test_uid),
            [metric.upper()] * len(test_uid),
            test_pert_total_metric[:, -1],
            ["Perturbed"] * len(test_uid)
        )))

        rec_df_data.extend(list(zip(
            rec_uid,
            user_df.set_index('user_id').loc[rec_uid, sens_attr],
            [model_name] * len(rec_uid),
            [dataset.dataset_name] * len(rec_uid),
            [metric.upper()] * len(rec_uid),
            rec_orig_total_metric[:, -1],
            ["Original"] * len(rec_uid)
        )))

        rec_df_data.extend(list(zip(
            rec_uid,
            user_df.set_index('user_id').loc[rec_uid, sens_attr],
            [model_name] * len(rec_uid),
            [dataset.dataset_name] * len(rec_uid),
            [metric.upper()] * len(rec_uid),
            rec_pert_total_metric[:, -1],
            ["Perturbed"] * len(rec_uid)
        )))

cols = ['user_id', sens_attr.title(), 'Model', 'Dataset', 'Metric', 'Value', 'Graph Type']
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
        plot_columns = ["Model", "Dataset", "Graph Type", y_col]
        palette = dict(zip(["Original", "Perturbed"], sns.color_palette("colorblind", n_colors=2)))
        fig, axs = plt.subplots(1, len(datasets_list), figsize=(10, 6))
        axs = [axs] if not isinstance(axs, list) else axs
        for (_model, _dataset, graph_type), sub_df in metric_df.groupby(["Model", "Dataset", "Graph Type"]):
            dp_samples = utils.compute_DP_across_random_samples(
                sub_df, sens_attr.title(), 'Value', batch_size=batch_exp, iterations=args.iterations
            )

            plot_df_data.extend(list(zip(
                [_model] * args.iterations,
                [_dataset] * args.iterations,
                [graph_type] * args.iterations,
                dp_samples
            )))

        plot_df = pd.DataFrame(plot_df_data, columns=plot_columns)
        plot_df_gby = plot_df.groupby("Dataset")
        for dset, ax in zip(datasets_list, axs):
            dset_df = plot_df_gby.get_group(dset)
            sns.barplot(x="Model", y=y_col, data=dset_df, hue="Graph Type", ax=ax, palette=palette)
            ax.set_title(dset.upper())

        fig.suptitle(sens_attr.title())
        fig.tight_layout()
        fig.savefig(os.path.join(plots_path, f"{exp_data_name}_{metric}_DP_random_samples.png"))
        plt.close()

