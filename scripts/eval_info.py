import os
import sys
import yaml
import pickle
import argparse

import scipy
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from recbole.evaluator import Evaluator
from recbole.data import create_dataset

sys.path.append(os.path.dirname(sys.path[0]))
import biga.utils as utils
import biga.evaluation as eval_utils


if __name__ == "__main__":
    """It works only when called from outside of the scripts folder as a script (not as a module)."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_path', '--e', required=True)
    parser.add_argument('--iterations', '--it', default=100, type=int)
    args = parser.parse_args()

    real_group_map = {
        'gender': {'M': 'M', 'F': 'F'},
        'age': {'M': 'Y', 'F': 'O'},
        'user_wide_zone': {'M': 'America', 'F': 'Other'}
    }

    group_name_map = {
        "M": "Males",
        "F": "Females",
        "Y": "Younger",
        "O": "Older",
        "America": "America",
        "Other": "Other"
    }

    if args.exp_path[-1] != os.sep:
        args.exp_path += os.sep

    if 'experiments' in args.exp_path:
        _, dset, mod, _, _, s_attr, eps, cid, _ = args.exp_path.split('dp_explanations')[1].split(os.sep)
    else:
        _, dset, mod, _, s_attr, eps, cid, _ = args.exp_path.split('dp_ndcg_explanations')[1].split(os.sep)
    eps = eps.replace('epochs_', '')

    model_files = os.scandir(os.path.join(os.path.dirname(sys.path[0]), 'saved'))
    model_file = [f.path for f in model_files if mod in f.name and dset.upper() in f.name][0]

    config, model, dataset, train_data, valid_data, test_data = utils.load_data_and_model(
        model_file,
        os.path.join(args.exp_path, 'config.yaml')
    )

    mondel_pol = ''
    delcons_pol = 'CN'  # used by all the augmentation experiments (meaningless if only disadvantaged users are used)
    zerousers_pol = 'ZN'
    lowdegree_pol = 'LD'
    furthestusers_pol = 'F'
    sparseusers_pol = 'S'
    itemspref_pol = 'IP'
    nicheitems_pol = 'NI'  # not used because too similar with sparseusers_pol

    random_pol = 'RND-P'
    no_pert_col = 'NP'  # NoPerturbation

    policy_order_base = [
        mondel_pol,
        delcons_pol,
        zerousers_pol,
        lowdegree_pol,
        furthestusers_pol,
        sparseusers_pol,
        itemspref_pol,
        nicheitems_pol,
        random_pol,
        no_pert_col
    ]

    palette = dict(zip(policy_order_base, sns.color_palette("colorblind")))
    pol_hatches = dict(zip(policy_order_base, ['X', '.', '/', 'O', '*']))

    policy_map = {
        'force_removed_edges': mondel_pol,
        'group_deletion_constraint': delcons_pol,
        'random_perturbation': random_pol,
        'users_zero_constraint': zerousers_pol,
        'users_low_degree': lowdegree_pol,
        'users_furthest_constraint': furthestusers_pol,
        'sparse_users_constraint': sparseusers_pol,
        'items_preference_constraint': itemspref_pol,
        'niche_items_constraint': nicheitems_pol
    }

    exp_policies = [policy_map[k] for k, v in config['explainer_policies'].items() if v and k in policy_map]
    curr_policy = '+'.join(exp_policies)
    if int(cid) == 99:
        curr_policy = "L-" + curr_policy
    if int(cid) == 100:
        curr_policy = "G-" + curr_policy
    if int(cid) == 101:
        curr_policy = "N-" + curr_policy

    plots_path = os.path.join('scripts', 'plots', dset, mod, s_attr, f"{cid}_{curr_policy}")
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    checkpoint = torch.load(model_file)
    orig_test_pref_data = eval_utils.pref_data_from_checkpoint(config, checkpoint, train_data, test_data)
    orig_valid_pref_data = eval_utils.pref_data_from_checkpoint(config, checkpoint, train_data, valid_data)

    demo_group_map = dataset.field2id_token[s_attr]

    evaluator = Evaluator(config)
    for _pref_data, _eval_data in zip([orig_test_pref_data, orig_valid_pref_data], [test_data.dataset, valid_data.dataset]):
        _pref_data['Demo Group'] = [
            demo_group_map[dg] for dg in dataset.user_feat[s_attr][_pref_data['user_id']].numpy()
        ]
        _pref_data["Demo Group"] = _pref_data["Demo Group"].map(real_group_map[s_attr.lower()]).to_numpy()

        metric_result = eval_utils.compute_metric(evaluator, _eval_data, _pref_data, 'cf_topk_pred', 'ndcg')
        _pref_data['Value'] = metric_result[:, -1]
        _pref_data['Quantile'] = _pref_data['Value'].map(lambda x: np.ceil(x * 10) / 10 if x > 0 else 0.1)

    batch_exp = config['user_batch_exp']
    exps, rec_model_preds, test_model_preds = utils.load_dp_exps_file(args.exp_path)
    best_exp = utils.get_best_exp_early_stopping(exps[0], config)

    pert_edges = best_exp[utils.exp_col_index('del_edges')]

    def pert_edges_mapper(pe, rec_dset):
        return pe

    test_pert_df, valid_pert_df = eval_utils.extract_metrics_from_perturbed_edges(
        {(dset, s_attr): pert_edges},
        models=[mod],
        metrics=["NDCG"],  # ["NDCG", "Precision", "Recall", "Hit"],
        models_path=os.path.join(os.path.dirname(sys.path[0]), 'saved'),
        on_bad_models='ignore',
        remap=pert_edges_mapper
    )

    test_pert_df = test_pert_df[test_pert_df['Metric'].str.upper() == 'NDCG']
    valid_pert_df = valid_pert_df[valid_pert_df['Metric'].str.upper() == 'NDCG']
    for _pert_df in [test_pert_df, valid_pert_df]:
        _pert_df['Quantile'] = _pert_df['Value'].map(lambda x: np.ceil(x * 10) / 10 if x > 0 else 0.1)
        _pert_df["Demo Group"] = _pert_df["Demo Group"].map(real_group_map[s_attr.lower()]).to_numpy()

    # print(f'{"*" * 15} Test {"*" * 15}')
    # print(f'{"*" * 15} {s_attr.title()} {"*" * 15}')
    # for dg, sa_dg_df in test_pert_df.groupby('Demo Group'):
    #     print(f'\n{"*" * 15} {dg.title()} {"*" * 15}')
    #     print(sa_dg_df.describe())

    dgs = list(real_group_map[s_attr.lower()].values())
    orig_pert_pval_dict = {}
    plot_df_data = []
    for orig_dp_df, pert_dp_df, split in zip(
        [orig_test_pref_data, orig_valid_pref_data],
        [test_pert_df, valid_pert_df],
        ['Test', 'Valid']
    ):
        total = orig_dp_df['Value'].mean()
        metr_dg1 = orig_dp_df.loc[orig_dp_df['Demo Group'] == dgs[0], 'Value'].to_numpy()
        metr_dg2 = orig_dp_df.loc[orig_dp_df['Demo Group'] == dgs[1], 'Value'].to_numpy()
        _dp = eval_utils.compute_DP(metr_dg1.mean(), metr_dg2.mean())
        pval = scipy.stats.mannwhitneyu(metr_dg1, metr_dg2).pvalue
        plot_df_data.append([_dp, split, 'Orig', metr_dg1.mean(), metr_dg2.mean(), total, pval])

        total = orig_dp_df['Value'].mean()
        metr_dg1 = pert_dp_df.loc[pert_dp_df['Demo Group'] == dgs[0], 'Value'].to_numpy()
        metr_dg2 = pert_dp_df.loc[pert_dp_df['Demo Group'] == dgs[1], 'Value'].to_numpy()
        _dp = eval_utils.compute_DP(metr_dg1.mean(), metr_dg2.mean())
        pval = scipy.stats.mannwhitneyu(metr_dg1, metr_dg2).pvalue
        plot_df_data.append([_dp, split, curr_policy, metr_dg1.mean(), metr_dg2.mean(), total, pval])

        try:
            orig_pert_pval_dict[split] = scipy.stats.wilcoxon(
                orig_dp_df.sort_values('user_id')['Value'].to_numpy(),
                pert_dp_df.sort_values('user_id')['Value'].to_numpy()
            ).pvalue
        except ValueError:  # zero_method 'wilcox' and 'pratt' do not work if x - y is zero for all elements.
            # highest pvalue because the distributions are equal
            orig_pert_pval_dict[split] = 1.0

    dp_plot_df = pd.DataFrame(plot_df_data, columns=['$\Delta$NDCG', 'Split', 'Policy', *dgs, 'NDCG', 'pvalue'])
    dp_plot_df.to_markdown(os.path.join(plots_path, 'DP_barplot.md'), index=False)
    dp_plot_df.to_latex(os.path.join(plots_path, 'DP_barplot.tex'), index=False)
    with open(os.path.join(plots_path, 'orig_pert_pval_dict.pkl'), 'wb') as f:
        pickle.dump(orig_pert_pval_dict, f)
    print(dp_plot_df)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.barplot(x='Split', y='$\Delta$NDCG', data=dp_plot_df, hue='Policy', ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_path, 'DP_barplot.png'), bbox_inches="tight", pad_inches=0, dpi=200)
    plt.close()

    exit()

    dp_samples, dgs_order = eval_utils.compute_DP_across_random_samples(
        test_pert_df, s_attr, "Demo Group", dset, 'Value', batch_size=batch_exp, iterations=args.iterations
    )

    orig_dp_samples, orig_dgs_order = eval_utils.compute_DP_across_random_samples(
        orig_pref_data, s_attr, "Demo Group", dset, 'Value', batch_size=batch_exp, iterations=args.iterations
    )

    plot_df = pd.DataFrame(
        zip(
            np.concatenate([dp_samples[:, -1], orig_dp_samples[:, -1]]),
            ['Perturbed'] * args.iterations + ['Orig'] * args.iterations
        ),
        columns=['NDCG DP', 'Policy']
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.boxplot(x='Policy', y='NDCG DP', data=plot_df, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_path, f'DP_across_samples.png'), bbox_inches="tight", pad_inches=0, dpi=200)

    fig, axs = plt.subplots(2, 1, figsize=(15, 12))
    for i, (pref_df, pref_title) in enumerate(zip([test_pert_df, orig_pref_data], ['Perturbed', 'Orig'])):
        sns.boxplot(x='Quantile', y='Value', data=pref_df, hue='Demo Group', ax=axs[i])
        axs[i].set_title(pref_title)
    fig.tight_layout()
    fig.savefig(
        os.path.join(plots_path, f'quantile_boxplot.png'),
        bbox_inches="tight", pad_inches=0, dpi=200
    )

    perc_data = []
    for pref_df, pref_title in zip([test_pert_df, orig_pref_data], ['Perturbed', 'Orig']):
        for quant, qnt_df in pref_df.groupby('Quantile'):
            perc_data.append([
                'M',
                quant,
                len(qnt_df[qnt_df['Demo Group'] == 'M']) / len(pref_df[pref_df['Demo Group'] == 'M']),
                pref_title
            ])
            perc_data.append([
                'F',
                quant,
                len(qnt_df[qnt_df['Demo Group'] == 'F']) / len(pref_df[pref_df['Demo Group'] == 'F']),
                pref_title
            ])

    perc_df = pd.DataFrame(perc_data, columns=['Demo Group', 'Part', 'Value', 'Method'])
    fig, axs = plt.subplots(2, 1, figsize=(15, 12))
    for i, method in enumerate(['Perturbed', 'Orig']):
        sns.barplot(x='Part', y='Value', data=perc_df[perc_df.Method == method], hue='Demo Group', ax=axs[i])
        axs[i].set_title(method)
    fig.tight_layout()
    fig.savefig(
        os.path.join(plots_path, f'perc_barplot.png'),
        bbox_inches="tight", pad_inches=0, dpi=200
    )
