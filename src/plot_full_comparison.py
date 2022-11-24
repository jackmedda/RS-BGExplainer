# %%
import os
import argparse
import inspect

import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_tick
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
    test_orig_total_metric = best_test_exp_result[model_name][metric]
    rec_orig_total_metric = best_rec_exp_result[model_name][metric]

    test_pert_total_metric = best_test_exp_result[model_dp_s][metric]
    rec_pert_total_metric = best_rec_exp_result[model_dp_s][metric]

    m_group_mask = best_rec_exp_df[model_dp_s].user_id.isin(user_df.loc[user_df[sens_attr] == m_idx, 'user_id'])
    f_group_mask = best_rec_exp_df[model_dp_s].user_id.isin(user_df.loc[user_df[sens_attr] == f_idx, 'user_id'])

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

    if sens_attr == "gender":
        real_group_map = {'M': 'M', 'F': 'F'}
    else:
        real_group_map = {'M': 'Y', 'F': 'O'}

    _test_df_data.extend(list(zip(
        test_uid,
        user_df.set_index('user_id').loc[test_uid, sens_attr].map(attr_map.__getitem__).map(real_group_map),
        [model_name] * len(test_uid),
        [dataset.dataset_name] * len(test_uid),
        [metric.upper()] * len(test_uid),
        test_orig_total_metric[:, -1],
        ["NoPolicy"] * len(test_uid)
    )))

    _test_df_data.extend(list(zip(
        test_uid,
        user_df.set_index('user_id').loc[test_uid, sens_attr].map(attr_map.__getitem__).map(real_group_map),
        [model_name] * len(test_uid),
        [dataset.dataset_name] * len(test_uid),
        [metric.upper()] * len(test_uid),
        test_pert_total_metric[:, -1],
        [policy] * len(test_uid)
    )))

    _rec_df_data.extend(list(zip(
        rec_uid,
        user_df.set_index('user_id').loc[rec_uid, sens_attr].map(attr_map.__getitem__).map(real_group_map),
        [model_name] * len(rec_uid),
        [dataset.dataset_name] * len(rec_uid),
        [metric.upper()] * len(rec_uid),
        rec_orig_total_metric[:, -1],
        ["NoPolicy"] * len(rec_uid)
    )))

    _rec_df_data.extend(list(zip(
        rec_uid,
        user_df.set_index('user_id').loc[rec_uid, sens_attr].map(attr_map.__getitem__).map(real_group_map),
        [model_name] * len(rec_uid),
        [dataset.dataset_name] * len(rec_uid),
        [metric.upper()] * len(rec_uid),
        rec_pert_total_metric[:, -1],
        [policy] * len(rec_uid)
    )))

    return _group_edge_del


def update_plot_del_data(_test_df_del_data, _rec_df_del_data):
    filter_cols = ['user_id', 'epoch', 'n_del_edges', 'fair_loss']

    exp_test_df = all_exp_test_dfs[model_dp_s][filter_cols]
    exp_rec_df = all_exp_rec_dfs[model_dp_s][filter_cols]
    uid_list_test = next(exp_test_df.groupby('n_del_edges').__iter__())[1].user_id
    uid_list_rec = next(exp_rec_df.groupby('n_del_edges').__iter__())[1].user_id

    result_test_df_data, result_rec_df_data = [], []
    for n_del, res in test_result_all_data[model_dp_s].items():
        result_test_df_data.extend(list(zip(
            [n_del] * len(uid_list_test),
            uid_list_test,
            res[metric][:, -1],
            [metric.upper()] * len(uid_list_test)
        )))
    for n_del, res in rec_result_all_data[model_dp_s].items():
        result_rec_df_data.extend(list(zip(
            [n_del] * len(uid_list_rec),
            uid_list_rec,
            res[metric][:, -1],
            [metric.upper()] * len(uid_list_rec)
        )))

    exp_test_df = exp_test_df.join(
        pd.DataFrame(
            result_test_df_data, columns=['n_del_edges', 'user_id', 'Value', 'Metric']
        ).set_index(['n_del_edges', 'user_id']),
        on=['n_del_edges', 'user_id']
    ).join(user_df.set_index('user_id'), on='user_id')
    exp_rec_df = exp_rec_df.join(
        pd.DataFrame(
            result_rec_df_data, columns=['n_del_edges', 'user_id', 'Value', 'Metric']
        ).set_index(['n_del_edges', 'user_id']),
        on=['n_del_edges', 'user_id']
    ).join(user_df.set_index('user_id'), on='user_id')

    _test_result = exp_test_df.pop("Value")
    _rec_result = exp_rec_df.pop("Value")

    test_orig_total_metric = best_test_exp_result[model_name][metric][:, -1]
    rec_orig_total_metric = best_rec_exp_result[model_name][metric][:, -1]

    unique_test_del_edges = len(test_result_all_data[model_dp_s])
    unique_rec_del_edges = len(rec_result_all_data[model_dp_s])

    _test_df_del_data.extend(
        np.c_[
            exp_test_df.values,
            [model_name] * len(exp_test_df),
            [dataset.dataset_name] * len(exp_test_df),
            np.tile(test_orig_total_metric, unique_test_del_edges),
            ["NoPolicy"] * len(exp_test_df)
        ].tolist()
    )
    _test_df_del_data.extend(
        np.c_[
            exp_test_df.values,
            [model_name] * len(exp_test_df),
            [dataset.dataset_name] * len(exp_test_df),
            _test_result.to_numpy(),
            [policy] * len(exp_test_df)
        ].tolist()
    )

    _rec_df_del_data.extend(
        np.c_[
            exp_rec_df.values,
            [model_name] * len(exp_rec_df),
            [dataset.dataset_name] * len(exp_rec_df),
            np.tile(rec_orig_total_metric, unique_rec_del_edges),
            ["NoPolicy"] * len(exp_rec_df)
        ].tolist()
    )
    _rec_df_del_data.extend(
        np.c_[
            exp_rec_df.values,
            [model_name] * len(exp_rec_df),
            [dataset.dataset_name] * len(exp_rec_df),
            _rec_result.to_numpy(),
            [policy] * len(exp_rec_df)
        ].tolist()
    )


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

dataset_map = {
    "ml-100k": "ML 100K",
    "ml-1m": "ML 1M",
    "lastfm-1k": "Last.FM 1K"
}

group_name_map = {
    "M": "Males",
    "F": "Females",
    "Y": "Younger",
    "O": "Older"
}

# %%
axs = {}
datasets_train_inter_sizes = {}
test_df_data, rec_df_data = [], []
test_del_df_data, rec_del_df_data = [], []
for model_file, exp_config_file in zip(args.model_files, args.explainer_config_files):
    config, model, dataset, train_data, valid_data, test_data = utils.load_data_and_model(model_file, exp_config_file)

    datasets_train_inter_sizes[dataset.dataset_name] = train_data.dataset.inter_num

    model_name = model.__class__.__name__
    sens_attr, epochs, batch_exp = config['sensitive_attribute'], config['cf_epochs'], config['user_batch_exp']
    policy = '+'.join([pm for p, pm in policy_map.items() if config['explainer_policies'][p]])

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

    for metric in metrics:
        group_edge_del = update_plot_data(test_df_data, rec_df_data)
        update_plot_del_data(test_del_df_data, rec_del_df_data)

cols = ['user_id', sens_attr.title(), 'Model', 'Dataset', 'Metric', 'Value', 'Policy']
test_df = pd.DataFrame(test_df_data, columns=cols)
rec_df = pd.DataFrame(rec_df_data, columns=cols)

del_cols = ['user_id', 'Epoch', '# Del Edges', 'Fair Loss', 'Metric', sens_attr.title(), 'Model', 'Dataset', 'Value', 'Policy']
test_del_df = pd.DataFrame(test_del_df_data, columns=del_cols)
rec_del_df = pd.DataFrame(rec_del_df_data, columns=del_cols)

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

for df, del_df, exp_data_name in zip([test_df, rec_df], [test_del_df, rec_del_df], ["test", exp_rec_data]):
    _metr_df_gby = df.groupby("Metric")
    _metr_del_df_gby = del_df.groupby("Metric")

    _metrics = list(_metr_df_gby.groups.keys())
    for metric in _metrics:
        metric_df = _metr_df_gby.get_group(metric)
        metric_del_df = _metr_del_df_gby.get_group(metric)

        nop_mask = metric_df["Policy"] == "NoPolicy"
        metr_df_nop = metric_df[nop_mask].copy()
        metr_df_p = metric_df[~nop_mask].copy()

        metr_df_nop["State"] = "Before"
        metr_df_p["State"] = "After"

        metr_df = pd.concat(
            [metr_df_p] + [metr_df_nop.copy().replace('NoPolicy', p) for p in metr_df_p.Policy.unique()],
            ignore_index=True
        )

        metr_df = pd.melt(
            metr_df, metr_df.columns[~metr_df.columns.isin(["Gender", "Age"])],
            var_name="Sens Attr",
            value_name="Demo Group"
        )

        metr_df_mean = metr_df.groupby(
            ["Dataset", "Model", "Policy", "State", "Sens Attr", "Demo Group"]
        ).mean().reset_index()
        metr_df_pivot = metr_df_mean.pivot(
            index=["Dataset", "Model", "Policy"],
            columns=["Sens Attr", "Demo Group", "State"],
            values="Value"
        )
        table_df = metr_df_pivot.reindex(
            ["Gender", "Age"], axis=1, level=0
        ).reindex(
            ["M", "F", "Y", "O"], axis=1, level=1
        ).reindex(
            ['Before', 'After'], axis=1, level=2
        )
        for level_attr, demo_groups in zip(["Gender", "Age"], [["M", "F"], ["Y", "O"]]):
            if level_attr in table_df:
                table_dp_df = (table_df[(level_attr, demo_groups[0])] - table_df[(level_attr, demo_groups[1])]).abs()
                table_dp_df.columns = pd.MultiIndex.from_product([[level_attr], ["DP"], ["Before", "After"]])
                table_df = pd.concat([table_df, table_dp_df], axis=1)
        table_df.columns.names = [''] * len(table_df.columns.names)
        table_df.columns = table_df.columns.map(lambda x: (x[0], group_name_map.get(x[1], x[1]), x[2]))
        table_df.round(3).to_latex(
            os.path.join(plots_path, f"table_{exp_data_name}_{metric}_best_epoch.tex")
        )

        plot_df_data = []
        plot_del_df_data = []
        y_col = f"{metric.upper()} DP"
        plot_columns = ["Model", "Dataset", "Policy", y_col]
        plot_del_columns = ["Model", "Dataset", "Policy", "% Del Edges", y_col]
        palette = dict(zip(np.concatenate([["NoPolicy"], df["Policy"].unique()]), sns.color_palette("colorblind")))
        _m_dset_pol_df = metric_df.groupby(["Model", "Dataset", "Policy"])
        _m_dset_pol_del_df = metric_del_df.groupby(["Model", "Dataset", "Policy"])

        m_dset_pol = list(_m_dset_pol_df.groups.keys())
        for (_model, _dataset, _policy) in tqdm.tqdm(m_dset_pol, desc="Extracting DP across random samples"):
            sub_df = _m_dset_pol_df.get_group((_model, _dataset, _policy))
            sub_del_df = _m_dset_pol_del_df.get_group((_model, _dataset, _policy))

            dp_samples = utils.compute_DP_across_random_samples(
                sub_df, sens_attr.title(), 'Value', batch_size=batch_exp, iterations=args.iterations
            )

            plot_df_data.extend(list(zip(
                [_model] * args.iterations,
                [_dataset] * args.iterations,
                [_policy] * args.iterations,
                dp_samples
            )))

            n_del_df_gby = sub_del_df.groupby("# Del Edges")
            sorted_del_edges = sub_del_df.sort_values("# Del Edges")["# Del Edges"].unique()
            for n_del in sorted_del_edges:
                n_del_df = n_del_df_gby.get_group(n_del)
                del_dp_samples = utils.compute_DP_across_random_samples(
                    n_del_df, sens_attr.title(), 'Value', batch_size=batch_exp, iterations=args.iterations
                )

                plot_del_df_data.extend(list(zip(
                    [_model] * args.iterations,
                    [_dataset] * args.iterations,
                    [_policy] * args.iterations,
                    [n_del] * args.iterations,
                    del_dp_samples
                )))

        plot_del_df_line = pd.DataFrame(plot_del_df_data, columns=plot_del_columns)
        plot_del_df_line_gby = plot_del_df_line.groupby(["Dataset", "Model"])
        fig_line = plt.figure(figsize=(15, 15), constrained_layout=True)
        subfigs = fig_line.subfigures(len(datasets_list), 1)
        subfigs = [subfigs] if not isinstance(subfigs, np.ndarray) else subfigs

        plot_df_bar = pd.DataFrame(plot_df_data, columns=plot_columns)
        plot_df_bar_gby = plot_df_bar.groupby("Dataset")
        fig_bar, axs_bar = plt.subplots(1, len(datasets_list), figsize=(10, 6))
        axs_bar = [axs_bar] if not isinstance(axs_bar, np.ndarray) else axs_bar
        for i, (dset, ax_bar) in enumerate(zip(datasets_list, axs_bar)):
            if dset in plot_df_bar_gby.groups:
                dset_bar_df = plot_df_bar_gby.get_group(dset)
                sns.barplot(x="Model", y=y_col, data=dset_bar_df, hue="Policy", ax=ax_bar, palette=palette)
                ax_bar.set_title(dset.upper())

            subfigs[i].suptitle(dset.upper())
            axs_line = subfigs[i].subplots(1, len(models_list))
            axs_line = [axs_line] if not isinstance(axs_line, np.ndarray) else axs_line
            for m, ax_line in zip(models_list, axs_line):
                if (dset, m) in plot_del_df_line_gby.groups:
                    dset_model_line_df = plot_del_df_line_gby.get_group((dset, m))
                    sns.lineplot(x="% Del Edges", y=y_col, data=dset_model_line_df, hue="Policy", ax=ax_line, palette=palette, ci=None)
                    ax_line.set_title(m.upper())
                    ax_line.xaxis.set_major_formatter(mpl_tick.FuncFormatter(lambda x, pos: f"{x / datasets_train_inter_sizes[dset] * 100:.2f}%"))

        fig_line.suptitle(sens_attr.title())
        fig_line.tight_layout()
        fig_line.savefig(os.path.join(plots_path, f"lineplot_{exp_data_name}_{metric}_DP_random_samples.png"))

        fig_bar.suptitle(sens_attr.title())
        fig_bar.tight_layout()
        fig_bar.savefig(os.path.join(plots_path, f"barplot_{exp_data_name}_{metric}_DP_random_samples.png"))
        plt.close("all")
