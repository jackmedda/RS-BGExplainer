# %%
import os
import ast
import pickle
import argparse
import inspect

import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_tick
import matplotlib.patches as mpatches
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

    if (dataset_name, model_name, sens_attr) not in no_policies:
        _test_df_data.extend(list(zip(
            test_uid,
            [sens_attr.title()] * len(test_uid),
            user_df.set_index('user_id').loc[test_uid, sens_attr].map(attr_map.__getitem__).map(real_group_map[sens_attr]),
            [model_name] * len(test_uid),
            [dataset_name] * len(test_uid),
            [metric.upper()] * len(test_uid),
            test_orig_total_metric[:, -1],
            ["NoPolicy"] * len(test_uid)
        )))

    _test_df_data.extend(list(zip(
        test_uid,
        [sens_attr.title()] * len(test_uid),
        user_df.set_index('user_id').loc[test_uid, sens_attr].map(attr_map.__getitem__).map(real_group_map[sens_attr]),
        [model_name] * len(test_uid),
        [dataset_name] * len(test_uid),
        [metric.upper()] * len(test_uid),
        test_pert_total_metric[:, -1],
        [policy] * len(test_uid)
    )))

    if (dataset_name, model_name, sens_attr) not in no_policies:
        _rec_df_data.extend(list(zip(
            rec_uid,
            [sens_attr.title()] * len(rec_uid),
            user_df.set_index('user_id').loc[rec_uid, sens_attr].map(attr_map.__getitem__).map(real_group_map[sens_attr]),
            [model_name] * len(rec_uid),
            [dataset_name] * len(rec_uid),
            [metric.upper()] * len(rec_uid),
            rec_orig_total_metric[:, -1],
            ["NoPolicy"] * len(rec_uid)
        )))

    _rec_df_data.extend(list(zip(
        rec_uid,
        [sens_attr.title()] * len(rec_uid),
        user_df.set_index('user_id').loc[rec_uid, sens_attr].map(attr_map.__getitem__).map(real_group_map[sens_attr]),
        [model_name] * len(rec_uid),
        [dataset_name] * len(rec_uid),
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

    if (dataset_name, model_name, sens_attr) not in no_policies:
        _test_df_del_data.extend(
            np.c_[
                exp_test_df.values,
                [sens_attr.title()] * len(exp_test_df),
                [model_name] * len(exp_test_df),
                [dataset_name] * len(exp_test_df),
                np.tile(test_orig_total_metric, unique_test_del_edges),
                ["NoPolicy"] * len(exp_test_df)
            ].tolist()
        )
    _test_df_del_data.extend(
        np.c_[
            exp_test_df.values,
            [sens_attr.title()] * len(exp_test_df),
            [model_name] * len(exp_test_df),
            [dataset_name] * len(exp_test_df),
            _test_result.to_numpy(),
            [policy] * len(exp_test_df)
        ].tolist()
    )

    if (dataset_name, model_name, sens_attr) not in no_policies:
        _rec_df_del_data.extend(
            np.c_[
                exp_rec_df.values,
                [sens_attr.title()] * len(exp_rec_df),
                [model_name] * len(exp_rec_df),
                [dataset_name] * len(exp_rec_df),
                np.tile(rec_orig_total_metric, unique_rec_del_edges),
                ["NoPolicy"] * len(exp_rec_df)
            ].tolist()
        )
    _rec_df_del_data.extend(
        np.c_[
            exp_rec_df.values,
            [sens_attr.title()] * len(exp_rec_df),
            [model_name] * len(exp_rec_df),
            [dataset_name] * len(exp_rec_df),
            _rec_result.to_numpy(),
            [policy] * len(exp_rec_df)
        ].tolist()
    )


def create_table_best_explanations(_metric_df):
    nop_mask = _metric_df["Policy"] == "NoPolicy"
    metr_df_nop = _metric_df[nop_mask].copy()
    metr_df_p = _metric_df[~nop_mask].copy()

    metr_df_nop["Status"] = "Before"
    metr_df_p["Status"] = "After"

    metr_df = pd.concat(
        [metr_df_p] + [metr_df_nop.copy().replace('NoPolicy', p) for p in metr_df_p.Policy.unique()],
        ignore_index=True
    )

    metr_df_mean = metr_df.groupby(
        ["Dataset", "Model", "Policy", "Status", "Sens Attr", "Demo Group"]
    ).mean().reset_index()
    metr_df_pivot = metr_df_mean.pivot(
        index=["Dataset", "Model", "Policy"],
        columns=["Sens Attr", "Demo Group", "Status"],
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
            table_dp_df.columns = pd.MultiIndex.from_product([[level_attr], ["$\Delta$"], ["Before", "After"]])
            table_df = pd.concat([table_df, table_dp_df], axis=1)
    table_df.columns = table_df.columns.map(lambda x: (x[0], group_name_map.get(x[1], x[1]), x[2]))
    table_out_bar_df = pd.melt(table_df, ignore_index=False).reset_index()
    table_df.columns.names = [''] * len(table_df.columns.names)
    table_df.round(3).to_latex(
        os.path.join(plots_path, f"table_{exp_data_name}_{metric}_best_epoch.tex")
    )

    return table_out_bar_df


def create_fig_bar2_legend(fig, _palette, _hatches, demo_groups, loc="upper left"):
    policies, pol_colors = zip(*_palette.items())
    fig.legend(
        [mpatches.Rectangle((0, 0), width=0, height=0)] +
        [mpatches.Rectangle((0, 0), width=2, height=0.7, edgecolor=c, facecolor=c) for c in pol_colors] +
        [mpatches.Rectangle((0, 0), width=0, height=0)] +
        [mpatches.Patch(facecolor='w', alpha=0, hatch=h) for h in _hatches],
        ['Policy'] + list(policies) + ['Demographic Groups'] + demo_groups,
        # bbox_to_anchor=(1, 1)
        loc=loc
    )


# %%
parser = argparse.ArgumentParser()
parser.add_argument('--model_files', nargs='+', required=True)
parser.add_argument('--explainer_config_files', required=True, nargs='+', type=str)
parser.add_argument('--iterations', default=100)

args = parser.parse_args()

assert len(args.model_files) == len(args.explainer_config_files), \
    "Pass the same number of perturbed model files and configuration files to be loaded"

script_path = os.path.abspath(os.path.dirname(inspect.getsourcefile(lambda: 0)))
script_path = os.path.join(script_path, 'src') if 'src' not in script_path else script_path

print(args)

policy_map = {
    'force_removed_edges': 'MonDel',   # Monotonic Deletions
    'group_deletion_constraint': 'DelCons'  # Deletion Constraint
}

dataset_map = {
    "ml-100k": "ML 100K",
    "ml-1m": "ML 1M",
    "lastfm-1k": "Last.FM 1K"
}


real_group_map = {
    'gender': {'M': 'M', 'F': 'F'},
    'age': {'M': 'Y', 'F': 'O'}
}

group_name_map = {
    "M": "Males",
    "F": "Females",
    "Y": "Younger",
    "O": "Older"
}

colors = {
    "Gender": {"M": "#0173b2", "F": "#de8f05"},
    "Age": {"Y": "#0173b2", "O": "#de8f05"}
}

exp_epochs, config_ids, datasets_list, models_list, sens_attrs = [], [], [], [], []
for exp_config_file in args.explainer_config_files:
    _, dset, model, _, s_attr, eps, cid, _ = exp_config_file.split('dp_ndcg_explanations')[1].split(os.sep)
    datasets_list.append(dset)
    models_list.append(model)
    sens_attrs.append(s_attr)
    exp_epochs.append(eps)
    config_ids.append(cid)

unique_datasets, unique_models, unique_sens_attrs = \
    np.unique(datasets_list).tolist(), np.unique(models_list).tolist(), np.unique(sens_attrs).tolist()
plots_path = os.path.join(
    get_plots_path('_'.join(unique_datasets), '_'.join(unique_models)),
    '_'.join(exp_epochs),
    '_'.join(config_ids),
    '_'.join(sens_attrs)
)
if not os.path.exists(plots_path):
    os.makedirs(plots_path)

if os.path.exists(os.path.join(plots_path, 'rec_df.csv')) and os.path.exists(os.path.join(plots_path, 'incdisp.pkl')):
    test_rows, rec_rows = 2, 3

    test_df = pd.read_csv(os.path.join(plots_path, 'test_df.csv'), skiprows=test_rows)
    rec_df = pd.read_csv(os.path.join(plots_path, 'rec_df.csv'), skiprows=rec_rows)
    test_del_df = pd.read_csv(os.path.join(plots_path, 'test_del_df.csv'), skiprows=test_rows)
    rec_del_df = pd.read_csv(os.path.join(plots_path, 'rec_del_df.csv'), skiprows=rec_rows)

    with open(os.path.join(plots_path, 'incdisp.pkl'), 'rb') as f:
        incdisp = pickle.load(f)

    with open(os.path.join(plots_path, 'del_edges.pkl'), 'rb') as f:
        del_edges = pickle.load(f)

    with open(os.path.join(plots_path, 'all_batch_exps.pkl'), 'rb') as f:
        all_batch_exps = pickle.load(f)

    with open(os.path.join(plots_path, 'train_datasets.pkl'), 'rb') as f:
        train_datasets = pickle.load(f)

    with open(os.path.join(plots_path, 'datasets_train_inter_sizes.pkl'), 'rb') as f:
        datasets_train_inter_sizes = pickle.load(f)

    with open(os.path.join(plots_path, 'rec_df.csv'), 'r') as f:
        metadata = [next(f) for _ in range(rec_rows)]
        exp_rec_data = metadata[2].split(': ')[1].strip()
else:
    # %%
    incdisp = {}
    del_edges = {}
    no_policies = set()
    all_batch_exps = {}
    train_datasets = {}
    datasets_train_inter_sizes = {}
    test_df_data, rec_df_data = [], []
    test_del_df_data, rec_del_df_data = [], []
    for model_file, exp_config_file in zip(args.model_files, args.explainer_config_files):
        config, model, dataset, train_data, valid_data, test_data = utils.load_data_and_model(model_file, exp_config_file)

        dataset_name = dataset.dataset_name
        model_name = model.__class__.__name__
        sens_attr, epochs, batch_exp = config['sensitive_attribute'], config['cf_epochs'], config['user_batch_exp']
        policy = '+'.join([pm for p, pm in policy_map.items() if config['explainer_policies'][p]])
        incdisp[(dataset_name, model_name)] = 'IncDisp' if config['explainer_policies']['increase_disparity'] else ''  # Increase Disparity
        all_batch_exps[dataset_name] = batch_exp

        train_datasets[dataset_name] = train_data.dataset
        datasets_train_inter_sizes[dataset_name] = train_data.dataset.inter_num

        exp_epochs.append(epochs)

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

        del model
        del dataset

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

        # the deleted edges are repeated for each row, so take the first is the same
        test_del_edges = best_test_exp_df[model_dp_s]['del_edges'].iloc[0].tolist()
        rec_del_edges = best_rec_exp_df[model_dp_s]['del_edges'].iloc[0].tolist()
        for exp_data_name, exp_del_edges in zip(["test", exp_rec_data], [test_del_edges, rec_del_edges]):
            for policy_type in ["NoPolicy", policy]:
                del_edges[(exp_data_name, dataset_name, model_name, policy_type, sens_attr.title())] = exp_del_edges

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

        no_policies.add((dataset_name, model_name, sens_attr))

    cols = ['user_id', 'Sens Attr', 'Demo Group', 'Model', 'Dataset', 'Metric', 'Value', 'Policy']
    duplicated_cols_subset = [c for c in cols if c not in ['Value']]
    test_df = pd.DataFrame(test_df_data, columns=cols).drop_duplicates(subset=duplicated_cols_subset, ignore_index=True)
    rec_df = pd.DataFrame(rec_df_data, columns=cols).drop_duplicates(subset=duplicated_cols_subset, ignore_index=True)

    del_cols = ['user_id', 'Epoch', '# Del Edges', 'Fair Loss', 'Metric', 'Demo Group', 'Sens Attr', 'Model', 'Dataset', 'Value', 'Policy']
    duplicated_del_cols_subset = [c for c in del_cols if c not in ['Value', 'Fair Loss', 'Epoch']]
    test_del_df = pd.DataFrame(test_del_df_data, columns=del_cols).drop_duplicates(subset=duplicated_del_cols_subset, ignore_index=True)
    rec_del_df = pd.DataFrame(rec_del_df_data, columns=del_cols).drop_duplicates(subset=duplicated_del_cols_subset, ignore_index=True)

    with open(os.path.join(plots_path, 'test_df.csv'), 'w') as f:
        f.write(f'# model_files {" ".join(args.model_files)}\n')
        f.write(f'# explainer_config_files {" ".join(args.explainer_config_files)}\n')
        test_df.to_csv(f, index=None)
    with open(os.path.join(plots_path, 'rec_df.csv'), 'w') as f:
        f.write(f'# model_files {" ".join(args.model_files)}\n')
        f.write(f'# explainer_config_files {" ".join(args.explainer_config_files)}\n')
        f.write(f'exp_rec_data: {exp_rec_data}\n')
        rec_df.to_csv(f, index=None)

    with open(os.path.join(plots_path, 'test_del_df.csv'), 'w') as f:
        f.write(f'# model_files {" ".join(args.model_files)}\n')
        f.write(f'# explainer_config_files {" ".join(args.explainer_config_files)}\n')
        test_del_df.to_csv(f, index=None)
    with open(os.path.join(plots_path, 'rec_del_df.csv'), 'w') as f:
        f.write(f'# model_files {" ".join(args.model_files)}\n')
        f.write(f'# explainer_config_files {" ".join(args.explainer_config_files)}\n')
        f.write(f'exp_rec_data: {exp_rec_data}\n')
        rec_del_df.to_csv(f, index=None)

    with open(os.path.join(plots_path, 'incdisp.pkl'), 'wb') as f:
        pickle.dump(incdisp, f)

    with open(os.path.join(plots_path, 'del_edges.pkl'), 'wb') as f:
        pickle.dump(del_edges, f)

    with open(os.path.join(plots_path, 'all_batch_exps.pkl'), 'wb') as f:
        pickle.dump(all_batch_exps, f)

    with open(os.path.join(plots_path, 'train_datasets.pkl'), 'wb') as f:
        pickle.dump(train_datasets, f)

    with open(os.path.join(plots_path, 'datasets_train_inter_sizes.pkl'), 'wb') as f:
        pickle.dump(datasets_train_inter_sizes, f)

unique_policies = sorted(test_df['Policy'].unique(), key=lambda x: 0 if x == "NoPolicy" else len(x))
for df, del_df, exp_data_name in zip([test_df, rec_df], [test_del_df, rec_del_df], ["test", exp_rec_data]):
    _metr_df_gby = df.groupby("Metric")
    _metr_del_df_gby = del_df.groupby("Metric")

    _metrics = list(_metr_df_gby.groups.keys())
    for metric in _metrics:
        metric_df = _metr_df_gby.get_group(metric)
        metric_del_df = _metr_del_df_gby.get_group(metric)

        table_bar_df = create_table_best_explanations(metric_df)

        plot_df_data = []
        plot_del_df_data = []
        y_col = f"$\Delta$ {metric.upper()}"
        plot_columns = ["Model", "Dataset", "Policy", "Sens Attr", y_col]
        plot_del_columns = ["Model", "Dataset", "Policy", "% Del Edges", "Sens Attr", y_col]
        palette = dict(zip(unique_policies, sns.color_palette("colorblind")))
        _m_dset_pol_df = metric_df.groupby(["Model", "Dataset", "Policy", "Sens Attr"])
        _m_dset_pol_del_df = metric_del_df.groupby(["Model", "Dataset", "Policy", "Sens Attr"])

        fig_qnt, axs_qnt = {}, {}
        for pol in metric_df["Policy"].unique():
            fig_qnt[pol] = plt.figure(figsize=(15, 15), constrained_layout=True)
            fig_qnt[pol].subfigures(len(unique_datasets), 1)
            for dset, subfig in zip(unique_datasets, fig_qnt[pol].subfigs):
                subfig.suptitle(dataset_map[dset])
                subfig.subplots(1, len(unique_models))

        fig_pca, axs_pca = {}, {}
        for pca_s_attr in unique_sens_attrs:
            fig_pca[pca_s_attr] = plt.figure(figsize=(15, 15), constrained_layout=True)
            fig_pca[pca_s_attr].subfigures(len(unique_datasets), 1)
            for dset, subfig in zip(unique_datasets, fig_pca[pca_s_attr].subfigs):
                subfig.suptitle(dataset_map[dset])
                subfig.subplots(len(unique_models), len(unique_policies), sharex=True, sharey=True)

        rel_th = 1e-1
        qnt_size = 100
        m_dset_pol = list(_m_dset_pol_df.groups.keys())
        for (_model, _dataset, _policy, _s_attr) in tqdm.tqdm(m_dset_pol, desc="Extracting DP across random samples"):
            sub_df = _m_dset_pol_df.get_group((_model, _dataset, _policy, _s_attr))
            sub_del_df = _m_dset_pol_del_df.get_group((_model, _dataset, _policy, _s_attr))

            pca_ax = fig_pca[_s_attr.lower()].subfigs[unique_datasets.index(_dataset)].axes[
                unique_models.index(_model) * len(unique_policies) + unique_policies.index(_policy)
            ]

            train_pca, pert_train_pca = utils.get_decomposed_adj_matrix(
                del_edges[(exp_data_name, _dataset, _model, _policy, _s_attr)],
                train_datasets[_dataset]
            )

            f_idx = (train_datasets[_dataset].field2id_token[_s_attr.lower()] == 'F').nonzero()[0][0]
            sens_data = train_datasets[_dataset].user_feat[_s_attr.lower()].numpy()[1:]
            sens_data = np.array([group_name_map[real_group_map[_s_attr.lower()]['F' if idx == f_idx else 'M']] for idx in sens_data])

            if _policy != 'NoPolicy':
                changes = np.abs(train_pca - pert_train_pca)
                mask = changes > np.percentile(changes, 95)
                rel_chs, = np.bitwise_or.reduce(mask, axis=1).nonzero()
                irrel_chs, = np.bitwise_or.reduce(~mask, axis=1).nonzero()

                sns.scatterplot(x=pert_train_pca[irrel_chs, 0], y=pert_train_pca[irrel_chs, 1], hue=sens_data[irrel_chs],
                                marker='o', ax=pca_ax, zorder=1, alpha=0.2)
                sns.scatterplot(x=pert_train_pca[rel_chs, 0], y=pert_train_pca[rel_chs, 1], hue=sens_data[rel_chs],
                                marker='o', zorder=3, ax=pca_ax)

                for rel_tr_pca, rel_pert_tr_pca in zip(train_pca[rel_chs], pert_train_pca[rel_chs]):
                    pca_ax.annotate(
                        "",
                        xy=rel_pert_tr_pca, xytext=rel_tr_pca,
                        arrowprops=dict(arrowstyle='->', lw=0.5, connectionstyle='arc3', zorder=2)
                    )

                utils.legend_without_duplicate_labels(pca_ax)
            else:
                sns.scatterplot(x=train_pca[:, 0], y=train_pca[:, 1], hue=sens_data, marker='o', ax=pca_ax)

            pca_ax.set_title(_policy)

            for dg_i, (dg, dg_df) in enumerate(sub_df.groupby("Demo Group")):
                ax = fig_qnt[_policy].subfigs[unique_datasets.index(_dataset)].axes[unique_models.index(_model)]

                qnt_values = dg_df.sort_values("Value", ascending=False)["Value"]
                qnt_values = [pct.mean() * (1 if dg_i == 0 else -1) for pct in np.array_split(qnt_values, qnt_size)]

                color = [colors[_s_attr][dg]] * qnt_size

                ax.bar(
                    np.arange(qnt_size),
                    qnt_values,
                    color=color,
                    align='edge',
                    label=dg
                )

                ax.xaxis.set_visible(False)
                ax.set_title(_model)

            dp_samples = utils.compute_DP_across_random_samples(
                sub_df, _s_attr, "Demo Group", _dataset, 'Value', batch_size=all_batch_exps[_dataset], iterations=args.iterations
            )

            plot_df_data.extend(list(zip(
                [_model] * args.iterations,
                [_dataset] * args.iterations,
                [_policy] * args.iterations,
                [_s_attr] * args.iterations,
                dp_samples
            )))

            n_del_df_gby = sub_del_df.groupby("# Del Edges")
            sorted_del_edges = sub_del_df.sort_values("# Del Edges")["# Del Edges"].unique()
            for n_del in sorted_del_edges:
                n_del_df = n_del_df_gby.get_group(n_del)
                del_dp_samples = utils.compute_DP_across_random_samples(
                    n_del_df, _s_attr, "Demo Group", _dataset, 'Value', batch_size=all_batch_exps[_dataset], iterations=args.iterations
                )

                plot_del_df_data.extend(list(zip(
                    [_model] * args.iterations,
                    [_dataset] * args.iterations,
                    [_policy] * args.iterations,
                    [n_del] * args.iterations,
                    [_s_attr] * args.iterations,
                    del_dp_samples
                )))

        for pol in fig_qnt:
            for subfig in fig_qnt[pol].subfigs:
                for ax in subfig.axes:
                    ax.legend(loc='upper right')
            fig_qnt[pol].savefig(os.path.join(plots_path, f'percentile_plot_{exp_data_name}_{metric}_{pol}.png'))

        for pca_s_attr in fig_pca:
            for subfig in fig_pca[pca_s_attr].subfigs:
                for ax in subfig.axes:
                    ax.legend(loc='upper right')
            fig_pca[pca_s_attr].savefig(os.path.join(plots_path, f'{pca_s_attr}_adj_matrix_decomposition_{exp_data_name}_{metric}.png'))
        plt.close("all")

        hatches = ['//', 'o']
        fig_bar2, axs_bar2 = plt.subplots(len(unique_sens_attrs), len(unique_datasets), figsize=(10, 6))
        axs_bar2 = [axs_bar2] if not isinstance(axs_bar2, np.ndarray) else axs_bar2
        table_bar_df.loc[table_bar_df["Status"] == "Before", "Policy"] = "NoPolicy"
        table_bar_df = table_bar_df.drop("Status", axis=1).rename(columns={'value': y_col})

        plot_del_df_line = pd.DataFrame(plot_del_df_data, columns=plot_del_columns)
        plot_df_bar = pd.DataFrame(plot_df_data, columns=plot_columns)

        plot_table_df_bar_gby = table_bar_df.groupby(["Sens Attr", "Dataset"])
        plot_del_df_line_gby = plot_del_df_line.groupby(["Sens Attr", "Dataset", "Model"])
        plot_df_bar_gby = plot_df_bar.groupby(["Sens Attr", "Dataset"])
        for s_attr_i, orig_sens_attr in enumerate(unique_sens_attrs):
            sens_attr = orig_sens_attr.title()
            fig_line = plt.figure(figsize=(15, 15), constrained_layout=True)
            subfigs = fig_line.subfigures(len(unique_datasets), 1)
            subfigs = [subfigs] if not isinstance(subfigs, np.ndarray) else subfigs

            fig_bar, axs_bar = plt.subplots(1, len(unique_datasets), figsize=(10, 6))
            axs_bar = [axs_bar] if not isinstance(axs_bar, np.ndarray) else axs_bar

            s_attr_dgs = []

            for i, (dset, ax_bar) in enumerate(zip(unique_datasets, axs_bar)):
                if (sens_attr, dset) in plot_df_bar_gby.groups:
                    dset_bar_sattr_df = plot_df_bar_gby.get_group((sens_attr, dset))
                    sns.barplot(x="Model", y=y_col, data=dset_bar_sattr_df, hue="Policy", ax=ax_bar, palette=palette)
                    ax_bar.set_title(dataset_map[dset])

                    if i == len(axs_bar) - 1:
                        handles, labels = ax_bar.get_legend_handles_labels()
                        fig_bar.legend(handles, labels, loc='upper center', ncol=len(labels))
                    ax_bar.get_legend().remove()

                if (sens_attr, dset) in plot_table_df_bar_gby.groups:
                    plot_tdf_bar_sattr_df = plot_table_df_bar_gby.get_group((sens_attr, dset))
                    _ax = axs_bar2[s_attr_i, i] if len(unique_sens_attrs) > 1 else axs_bar2[i]
                    dg_df_gby = plot_tdf_bar_sattr_df.groupby("Demo Group")
                    s_attr_dgs = [x for x in sorted(dg_df_gby.groups) if 'Delta' not in x]
                    for dg, hatch in zip(s_attr_dgs, hatches):
                        dg_df = dg_df_gby.get_group(dg)
                        sns.barplot(x="Model", y=y_col, data=dg_df, hue="Policy", ax=_ax, palette=palette, edgecolor='black', alpha=0.6, hatch=hatch)
                        _ax.get_legend().remove()
                        _ax.set_title(dataset_map[dset])

                subfigs[i].suptitle(dset.upper())
                axs_line = subfigs[i].subplots(1, len(unique_models))
                axs_line = [axs_line] if not isinstance(axs_line, np.ndarray) else axs_line
                for m, ax_line in zip(unique_models, axs_line):
                    if (sens_attr, dset, m) in plot_del_df_line_gby.groups:
                        dset_model_line_df = plot_del_df_line_gby.get_group((sens_attr, dset, m))
                        sns.lineplot(x="% Del Edges", y=y_col, data=dset_model_line_df, hue="Policy", ax=ax_line, palette=palette, ci=None)
                        ax_line.set_title(m.upper() + (f'+{incdisp[(dset, m)]}' if incdisp[(dset, m)] else ''))
                        ax_line.xaxis.set_major_formatter(mpl_tick.FuncFormatter(lambda x, pos: f"{x / datasets_train_inter_sizes[dset] * 100:.2f}%"))

            create_fig_bar2_legend(fig_bar2, palette, hatches, s_attr_dgs, loc="upper left")

            fig_line.suptitle(sens_attr.title())
            fig_line.savefig(os.path.join(plots_path, f"{sens_attr}_lineplot_{exp_data_name}_{metric}_DP_random_samples.png"))

            fig_bar.suptitle(sens_attr.title())
            fig_bar.tight_layout()
            fig_bar.savefig(os.path.join(plots_path, f"{sens_attr}_barplot_{exp_data_name}_{metric}_DP_random_samples.png"))

        fig_bar2.tight_layout()
        fig_bar2.savefig(os.path.join(plots_path, f"overlapping_barplot_{exp_data_name}_{metric}_DP_random_samples.png"))
        plt.close("all")
