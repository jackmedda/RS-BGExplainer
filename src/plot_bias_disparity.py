import os
import argparse
import inspect

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import utils


def get_plots_path():
    plots_path = os.path.join(script_path, os.pardir, f'bias_disparity_plots', config['dataset'])

    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    return plots_path


def plot_bias_analysis_disparity(train_bias, rec_bias, _train_data, item_categories=None):
    plots_path = get_plots_path()

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
        ax = sns.heatmap(df, vmin=-2.0, vmax=2.0)
        ax.set_title(attr.title())
        plt.tight_layout()
        plt.savefig(os.path.join(plots_path, f'{attr}.png'))
        plt.close()

    return plots_path


def clean_history_matrix(hist_m):
    for col in ['topk_pred', 'cf_topk_pred']:
        if isinstance(hist_m.iloc[0][col], str):
            hist_m[col] = hist_m[col].map(lambda x: np.array(x[1:-1].strip().split(), int))


def extract_bias_disparity(_exp_paths, _train_bias_ratio, _train_data, _config, best_exp_col):
    bias_disparity_all = {}
    for e_type, e_path in _exp_paths.items():
        if e_path is None:
            continue
        exps_data = utils.load_exps_file(e_path)

        top_exp_col = utils.EXPS_COLUMNS.index(best_exp_col) if best_exp_col is not None else None

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
            rec_bias_ratio = utils.generate_bias_ratio(
                _train_data,
                config,
                pred_col='cf_topk_pred',
                sensitive_attrs=_config['sensitive_attributes'],
                history_matrix=pref_data,
                mapped_keys=True
            )

            bias_disparity_all[e_type] = utils.compute_bias_disparity(_train_bias_ratio, rec_bias_ratio, _train_data)

            if 'original_pred' not in bias_disparity_all:
                rec_orig_bias = utils.generate_bias_ratio(
                    _train_data,
                    config,
                    pred_col='topk_pred',
                    sensitive_attrs=_config['sensitive_attributes'],
                    history_matrix=pref_data,
                    mapped_keys=True
                )

                bias_disparity_all['original_pred'] = utils.compute_bias_disparity(_train_bias_ratio, rec_orig_bias, _train_data)
        else:
            print("Pref Data is empty!")

    return bias_disparity_all


def plot_bias_disparity_diff_dumbbell(bd, sens_attrs, config_ids):
    plots_path = os.path.join(get_plots_path(), 'comparison', '_'.join(config_ids))
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    x, y = 'Bias Disparity', 'Category'
    item_cats = train_data.dataset.field2id_token['class']

    for attr in sens_attrs:
        attr_path = os.path.join(plots_path, attr)
        if not os.path.exists(attr_path):
            os.makedirs(attr_path)

        for demo_gr in bd['original_pred'][attr]:
            if bd['original_pred'][attr][demo_gr] is not None:
                for exp_type in bd:
                    if exp_type == 'original_pred' or bd[exp_type][attr][demo_gr] is None:
                        continue

                    orig_data = dict(zip(item_cats, bd['original_pred'][attr][demo_gr].numpy()))
                    exp_data = dict(zip(item_cats, bd[exp_type][attr][demo_gr].numpy()))

                    fig = plt.figure(figsize=(12, 12))
                    ax = fig.gca()

                    df_orig = pd.DataFrame.from_dict(orig_data, orient='index', columns=[x]).reset_index().dropna()
                    df_orig = df_orig.rename(columns={'index': y})
                    df_exp = pd.DataFrame.from_dict(exp_data, orient='index', columns=[x]).reset_index().dropna()
                    df_exp = df_exp.rename(columns={'index': y})

                    sns.scatterplot(x=x, y=y, color='#F5793A', data=df_exp, ax=ax, s=200, label=exp_type, zorder=2)
                    sns.scatterplot(x=x, y=y, color='#A95AA1', data=df_orig, ax=ax, s=200, label='original', zorder=2)

                    sorted_df = pd.concat([df_orig, df_exp]).sort_values([y, x]).reset_index(drop=True)
                    for i in range(0, len(sorted_df), 2):
                        ax.plot(sorted_df.loc[i:(i + 1), x], sorted_df.loc[i:(i + 1), y], 'k', zorder=1)

                    ax.plot([0., 0.], ax.get_ylim(), 'k--', zorder=1)

                    plt.savefig(os.path.join(attr_path, f"{demo_gr}#dumbbell_orig__{exp_type}.png"))
                    plt.close()


def plot_bias_disparity_barplot(bd, sens_attrs, config_ids):
    plots_path = os.path.join(get_plots_path(), 'comparison', '_'.join(config_ids))
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    for attr in sens_attrs:
        attr_path = os.path.join(plots_path, attr)
        if not os.path.exists(attr_path):
            os.makedirs(attr_path)

        rows = int(np.ceil((len(bd['original_pred'][attr]) - 1) / 3))
        cols = min(3, len(bd['original_pred'][attr]) - 1)

        fig, axs = plt.subplots(rows, cols, sharey=True, figsize=(12, 12))
        axs = axs.ravel()
        i = 0
        for demo_gr in bd['original_pred'][attr]:
            if bd['original_pred'][attr][demo_gr] is not None:
                plot_data = [[], []]
                for exp_type in bd:
                    if exp_type == 'original_pred' or bd[exp_type][attr][demo_gr] is None:
                        continue

                    if 'original_pred' not in plot_data[1]:
                        orig_data = bd['original_pred'][attr][demo_gr].numpy().tolist()
                        plot_data[0].extend(orig_data)
                        plot_data[1].extend(['original_pred'] * len(orig_data))

                    exp_data = bd[exp_type][attr][demo_gr].numpy().tolist()
                    plot_data[0].extend(exp_data)
                    plot_data[1].extend([exp_type] * len(exp_data))

                df = pd.DataFrame(plot_data, index=['Bias Disparity', 'Recommendations Type']).T.dropna()

                sns.boxplot(x='Recommendations Type', y='Bias Disparity', data=df, ax=axs[i])
                axs[i].plot(axs[i].get_xlim(), [0., 0.], 'k--')
                axs[1].set_xlabel("")

                axs[i].set_title(f"{attr.title()}: {demo_gr.title()}")
                i += 1

        fig.savefig(os.path.join(attr_path, f"barplot.png"))
        plt.close()


def plot_explanations_fairness_trend(_exp_paths, train_bias, orig_disparity, _train_data, _config, config_ids):
    item_cats = train_data.dataset.field2id_token['class']

    for e_type, e_path in _exp_paths.items():
        if e_path is None:
            continue
        exps_data = utils.load_exps_file(e_path)

        data = []
        for user_id, user_exps in exps_data.items():
            for u_exp in user_exps:
                data.append([user_id, u_exp[1].squeeze(), u_exp[2].squeeze(), int(u_exp[-5])])

        data_df = pd.DataFrame(data, columns=['user_id', 'topk_pred', 'cf_topk_pred', 'n_del_edges'])

        if data_df.empty:
            print(f"User explanations are empty for {e_type}")
            continue

        bd_data = {}
        n_users_data = {}
        for n_del, gr_df in data_df.groupby('n_del_edges'):
            rec_bias_ratio = utils.generate_bias_ratio(
                _train_data,
                config,
                pred_col='cf_topk_pred',
                sensitive_attrs=_config['sensitive_attributes'],
                history_matrix=gr_df[['user_id', 'topk_pred', 'cf_topk_pred']],
                mapped_keys=True
            )

            bd = utils.compute_bias_disparity(train_bias, rec_bias_ratio, _train_data)

            bd_data[n_del] = bd
            n_users_data[n_del] = len(gr_df['user_id'].unique())

        plots_path = os.path.join(get_plots_path(), 'comparison', '_'.join(config_ids))
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)

        for attr in _config['sensitive_attributes']:
            plot_data = []
            for n_del in bd_data:
                for demo_gr, bd_gr_data in bd_data[n_del][attr].items():
                    if bd_gr_data is not None:
                        l = len(item_cats)
                        plot_data.extend(list(zip([n_del] * l, [demo_gr] * l, bd_gr_data)))

            plot_data.extend(list(zip(
                np.repeat(list(bd_data.keys()), len(item_cats)),

            )))

            sns.lineplot(x='# Del Edges', y='Bias Disparity', hue='Attribute', data=plot_df)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', required=True)
    parser.add_argument('--explainer_config_file', default='../config/gcmc_explainer.yaml')
    parser.add_argument('--load_config_ids', nargs="+", type=int, default=[1, 1, 1],
                        help="follows the order ['pred_explain', 'FairBD', 'FairNDCGApprox'], set -1 to skip")
    parser.add_argument('--best_exp_col', default="loss_total")

    args = parser.parse_args()

    script_path = os.path.abspath(os.path.dirname(inspect.getsourcefile(lambda: 0)))

    if args.model_file is None:
        raise FileNotFoundError("need to specify a saved file with `--model_file`")

    print(args)

    config, model, dataset, train_data, valid_data, test_data = utils.load_data_and_model(args.model_file,
                                                                                          args.explainer_config_file)

    sens_attrs, epochs, batch_exp = config['sensitive_attributes'], config['cf_epochs'], config['user_batch_exp']
    batch_exp = 'individual' if batch_exp == 1 else 'group'
    if batch_exp == 'group':
        raise NotImplementedError()

    exp_paths = {}
    for c_id, exp_t in zip(args.load_config_ids, ['pred_explain', 'FairBD', 'FairNDCGApprox']):
        exp_paths[exp_t] = None
        if c_id != -1:
            exp_paths[exp_t] = os.path.join(script_path, 'explanations', dataset.dataset_name,
                                            exp_t, '_'.join(sens_attrs), f"epochs_{epochs}", str(batch_exp), str(c_id))

    train_bias_ratio = utils.generate_bias_ratio(
        train_data,
        config,
        sensitive_attrs=config['sensitive_attributes'],
        mapped_keys=True
    )

    user_df = pd.DataFrame({
        'user_id': train_data.dataset.user_feat['user_id'].numpy(),
        **{sens_attr: train_data.dataset.user_feat[sens_attr].numpy() for sens_attr in sens_attrs}
    })

    bias_disparity = extract_bias_disparity(exp_paths, train_bias_ratio, train_data, config, args.best_exp_col)

    plot_bias_disparity_diff_dumbbell(bias_disparity, sens_attrs, map(str, args.load_config_ids))

    plot_bias_disparity_barplot(bias_disparity, sens_attrs, map(str, args.load_config_ids))

    plot_explanations_fairness_trend(exp_paths, train_bias_ratio, bias_disparity['original_pred'], train_data, config. args.load_config_ids)
