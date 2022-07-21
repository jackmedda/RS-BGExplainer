# %%
import os
import argparse
import inspect

import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import src.utils as utils

# %%
def get_plots_path():
    plots_path = os.path.join(script_path, os.pardir, f'bias_disparity_plots', config['dataset'], args.best_exp_col)

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

        best_exp_col = best_exp_col.lower()
        top_exp_func = None
        if best_exp_col not in ["first", "last", "mid"]:
            top_exp_col = utils.EXPS_COLUMNS.index(best_exp_col) if best_exp_col is not None else None
            if top_exp_col is not None:
                top_exp_func = lambda exp: sorted(exp, key=lambda x: x[top_exp_col])[0]
        elif best_exp_col == "first":
            top_exp_func = lambda exp: exp[0]
        elif best_exp_col == "last":
            top_exp_func = lambda exp: exp[-1]
        elif best_exp_col == "mid":
            top_exp_func = lambda exp: exp[len(exp) // 2]

        pref_data = []
        for user_id, user_exps in exps_data.items():
            u_exps = user_exps
            if top_exp_func is not None and user_exps:
                u_exps = top_exp_func(user_exps)
                u_exps = [u_exps]
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

            if 'GCMC' not in bias_disparity_all:
                rec_orig_bias = utils.generate_bias_ratio(
                    _train_data,
                    config,
                    pred_col='topk_pred',
                    sensitive_attrs=_config['sensitive_attributes'],
                    history_matrix=pref_data,
                    mapped_keys=True
                )

                bias_disparity_all['GCMC'] = utils.compute_bias_disparity(_train_bias_ratio, rec_orig_bias, _train_data)
        else:
            print("Pref Data is empty!")

    return bias_disparity_all


def plot_bias_disparity_diff_dumbbell(bd, sens_attrs, config_ids, sort="dots"):
    plots_path = os.path.join(get_plots_path(), 'comparison', '_'.join(config_ids))
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    if sort not in ["dots", "barplot_side"]:
        raise NotImplementedError(f"sort = {sort} not supported for dumbbell")

    x, y = 'Bias Disparity', 'Category'
    item_cats = train_data.dataset.field2id_token['class']

    sensitive_maps = {sens_attr: train_data.dataset.field2id_token[sens_attr] for sens_attr in sens_attrs}

    for attr in sens_attrs:
        attr_path = os.path.join(plots_path, attr)
        if not os.path.exists(attr_path):
            os.makedirs(attr_path)

        cats_inter_counts_attr = {}
        for demo_gr, demo_df in joint_df.groupby(attr):
            class_counts = demo_df.explode('class').value_counts('class')
            class_counts.index = class_counts.index.map(item_cats.__getitem__)
            cats_inter_counts_attr[sensitive_maps[attr][demo_gr]] = class_counts.to_dict()

        for demo_gr in bd['GCMC'][attr]:
            if bd['GCMC'][attr][demo_gr] is not None:
                for exp_type in bd:
                    if exp_type == 'GCMC' or bd[exp_type][attr][demo_gr] is None:
                        continue

                    orig_data = dict(zip(item_cats, bd['GCMC'][attr][demo_gr].numpy()))
                    exp_data = dict(zip(item_cats, bd[exp_type][attr][demo_gr].numpy()))

                    df_orig = pd.DataFrame.from_dict(orig_data, orient='index', columns=[x]).reset_index().dropna()
                    df_orig = df_orig.rename(columns={'index': y})
                    df_exp = pd.DataFrame.from_dict(exp_data, orient='index', columns=[x]).reset_index().dropna()
                    df_exp = df_exp.rename(columns={'index': y})

                    if sort == "dots":
                        order = df_orig.sort_values(x)[y].to_list()
                        bar_data = pd.DataFrame(cats_inter_counts_attr[demo_gr].items(), columns=[y, x])
                    elif sort == "barplot_side":
                        order, vals = map(list, zip(*(sorted(cats_inter_counts_attr[demo_gr].items(), key=lambda x: x[1])[::-1])))
                        bar_data = pd.DataFrame(zip(order, vals), columns=[y, x])

                    g = sns.JointGrid(height=12, space=0.5)
                    g.ax_marg_x.remove()
                    sns.barplot(x=x, y=y, data=bar_data, ax=g.ax_marg_y, color="black", order=order)

                    sns.stripplot(x=x, y=y, color='#F5793A', data=df_exp, ax=g.ax_joint, jitter=False, s=10, label=exp_type, zorder=2, order=order)
                    sns.stripplot(x=x, y=y, color='#A95AA1', data=df_orig, ax=g.ax_joint, jitter=False, s=10, label='GCMC', zorder=2, order=order)

                    lines_df = df_orig.set_index('Category').join(df_exp.set_index('Category'), lsuffix='_orig').loc[order]
                    lines_styles = ((lines_df[x + '_orig'].abs() - lines_df[x].abs()) < 0).map(lambda x: ':' if x else '-').values.tolist()

                    lines_df['diff%'] = ((-(lines_df[x].abs() - lines_df[x + '_orig'].abs()) / lines_df[x + '_orig']) * 100).round(1)
                    lines_df['abs_diff'] = (lines_df[x + '_orig'] - lines_df[x]).abs().round(1)
                    lines_df['diff%'] = lines_df[['diff%', 'abs_diff']].apply(lambda row: f"{row['diff%']}% ({row['abs_diff']})", axis=1)
                    del lines_df['abs_diff']
                    for i, c in enumerate(lines_df.index):
                        g.ax_joint.plot(lines_df.loc[c, [x + '_orig', x]], [c, c], 'k', zorder=1, ls=lines_styles[i], label="")
                        g.ax_joint.text(lines_df.loc[c, [x + '_orig', x]].mean(), i - 0.3, lines_df.loc[c, 'diff%'], ha='center')

                    utils.legend_without_duplicate_labels(g.ax_joint)

                    g.ax_joint.plot([0., 0.], g.ax_joint.get_ylim(), 'k--', zorder=1)
                    g.ax_joint.set_title(f"{attr.title()}: {demo_gr}")

                    plt.tight_layout()
                    plt.savefig(os.path.join(attr_path, f"{demo_gr}#dumbbell_orig__{exp_type}.png"))
                    plt.close()


def plot_bias_disparity_boxplot(bd, sens_attrs, config_ids):
    plots_path = os.path.join(get_plots_path(), 'comparison', '_'.join(config_ids))
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    for attr in sens_attrs:
        attr_path = os.path.join(plots_path, attr)
        if not os.path.exists(attr_path):
            os.makedirs(attr_path)

        rows = int(np.ceil((len(bd['GCMC'][attr]) - 1) / 3))
        cols = min(3, len(bd['GCMC'][attr]) - 1)

        fig, axs = plt.subplots(rows, cols, sharey=True, figsize=(12, 12))
        axs = axs.ravel()
        i = 0
        for demo_gr in bd['GCMC'][attr]:
            if bd['GCMC'][attr][demo_gr] is not None:
                plot_data = [[], []]
                for exp_type in bd:
                    if exp_type == 'GCMC' or bd[exp_type][attr][demo_gr] is None:
                        continue

                    if 'GCMC' not in plot_data[1]:
                        orig_data = bd['GCMC'][attr][demo_gr].numpy().tolist()
                        plot_data[0].extend(orig_data)
                        plot_data[1].extend(['GCMC'] * len(orig_data))

                    exp_data = bd[exp_type][attr][demo_gr].numpy().tolist()
                    plot_data[0].extend(exp_data)
                    plot_data[1].extend([exp_type] * len(exp_data))

                df = pd.DataFrame(plot_data, index=['Bias Disparity', 'Recommendations Type']).T.dropna()

                sns.boxplot(x='Recommendations Type', y='Bias Disparity', data=df, ax=axs[i])
                axs[i].plot(axs[i].get_xlim(), [0., 0.], 'k--')
                axs[1].set_xlabel("")

                axs[i].set_title(f"{attr.title()}: {demo_gr.title()}")
                i += 1

        fig.savefig(os.path.join(attr_path, f"boxplot.png"))
        plt.close()


# %%
def extract_all_exp_bd_data(_exp_paths, train_bias, _train_data):
    sens_attributes = config['sensitive_attributes']
    sensitive_maps = {sens_attr: train_data.dataset.field2id_token[sens_attr] for sens_attr in sens_attributes}

    bd_data = {}
    n_users_data_all = {}
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

        bd_data[e_type] = {}
        n_users_data_all[e_type] = {}
        for n_del, gr_df in tqdm.tqdm(data_df.groupby('n_del_edges'), desc="Extracting BD from each explanation"):
            rec_bias_ratio = utils.generate_bias_ratio(
                _train_data,
                config,
                pred_col='cf_topk_pred',
                sensitive_attrs=sens_attributes,
                history_matrix=gr_df[['user_id', 'topk_pred', 'cf_topk_pred']],
                mapped_keys=True
            )

            bd = utils.compute_bias_disparity(train_bias, rec_bias_ratio, _train_data)
            bd_data[e_type][n_del] = bd

            gr_df_attr = gr_df['user_id'].drop_duplicates().to_frame().join(user_df.set_index('user_id'), on='user_id')
            n_users_data_all[e_type][n_del] = {attr: gr_df_attr[attr].value_counts().to_dict() for attr in sens_attributes}
            for attr in sens_attributes:
                n_users_del = n_users_data_all[e_type][n_del][attr]
                n_users_data_all[e_type][n_del][attr] = {sensitive_maps[attr][dg]: n_users_del[dg] for dg in n_users_del}

        return bd_data, n_users_data_all


def plot_explanations_fairness_trend(_bd_data_all, _n_users_data_all, orig_disparity, config_ids):
    item_cats = train_data.dataset.field2id_token['class']
    sens_attributes = config['sensitive_attributes']
    sensitive_maps = {sens_attr: train_data.dataset.field2id_token[sens_attr] for sens_attr in sens_attributes}

    dg_counts = {}
    for attr in config['sensitive_attributes']:
        dg_counts[attr] = user_df[attr].value_counts().to_dict()
        dg_counts[attr] = {sensitive_maps[attr][dg]: dg_counts[attr][dg] for dg in dg_counts[attr]}

    for e_type in _bd_data_all:
        bd_data = _bd_data_all[e_type]
        for attr in sens_attributes:
            sens_map = sensitive_maps[attr]
            plots_path = os.path.join(get_plots_path(), 'comparison', '_'.join(config_ids), attr)
            if not os.path.exists(plots_path):
                os.makedirs(plots_path)
            for d_gr in sens_map:
                if orig_disparity[attr][d_gr] is None:
                    continue

                n_users_data = {k: f"{v[attr][d_gr] / dg_counts[attr][d_gr] * 100:.1f}"
                                for k, v in _n_users_data_all.items() if d_gr in v[attr]}

                plot_data = []
                plot_bd_keys = []
                for n_del in bd_data:
                    bd_gr_data = bd_data[n_del][attr][d_gr].numpy()
                    if not np.isnan(bd_gr_data).all():
                        l = len(item_cats)
                        plot_data.extend(list(zip([n_del] * l, [e_type] * l, bd_gr_data)))
                        plot_bd_keys.append(n_del)

                plot_data.extend(list(zip(
                    np.repeat(plot_bd_keys, len(item_cats)),
                    np.repeat([f'GCMC'] * len(item_cats), len(plot_bd_keys)),
                    np.tile(orig_disparity[attr][d_gr].numpy(), len(plot_bd_keys))
                )))

                plot_df = pd.DataFrame(plot_data, columns=['# Del Edges', 'Attribute', 'Bias Disparity']).dropna()

                ax = sns.lineplot(x='# Del Edges', y='Bias Disparity', hue='Attribute', data=plot_df)
                n_ticks = len(ax.get_xticks())
                nud_keys = list(n_users_data.keys())
                xticks = np.linspace(1, len(nud_keys), n_ticks, dtype=int)
                xtick_labels = [f"{x} \n ({n_users_data[nud_keys[x - 1]]})" for x in xticks]
                ax.set_xticks(xticks)
                ax.set_xticklabels(xtick_labels)
                ax.set_title(f"{attr.title()}: {d_gr}")

                ax.plot(ax.get_xlim(), [0., 0.], 'k--')

                plt.tight_layout()
                plt.savefig(os.path.join(plots_path, f'{d_gr}#lineplot_over_del_edges_{e_type}.png'))
                plt.close()


# %%
def plot_explanations_fairness_trend_dumbbell(_bd_all_data, orig_disparity, config_ids, sort="dots", bin_size=10):
    sens_attributes = config["sensitive_attributes"]
    sensitive_maps = {sens_attr: train_data.dataset.field2id_token[sens_attr] for sens_attr in sens_attributes}
    item_cats = train_data.dataset.field2id_token['class']

    x, y = 'Bias Disparity', 'Category'

    if sort not in ["dots", "barplot_side"]:
        raise NotImplementedError(f"sort = {sort} not supported for dumbbell")

    for e_type in _bd_all_data:
        bd_data = _bd_all_data[e_type]
        for attr in sens_attributes:
            sens_map = sensitive_maps[attr]
            plots_path = os.path.join(get_plots_path(), 'comparison', '_'.join(config_ids), attr)
            if not os.path.exists(plots_path):
                os.makedirs(plots_path)

            cats_inter_counts_attr = {}
            for demo_gr, demo_df in joint_df.groupby(attr):
                class_counts = demo_df.explode('class').value_counts('class')
                class_counts.index = class_counts.index.map(item_cats.__getitem__)
                cats_inter_counts_attr[sensitive_maps[attr][demo_gr]] = class_counts.to_dict()

            for d_gr in sens_map:
                if orig_disparity[attr][d_gr] is None:
                    continue

                exp_data = []
                plot_bd_keys = []
                for n_del in bd_data:
                    bd_gr_data = bd_data[n_del][attr][d_gr].numpy()
                    if not np.isnan(bd_gr_data).all():
                        l = len(item_cats)
                        exp_data.extend(list(zip([n_del] * l, bd_gr_data, item_cats)))
                        plot_bd_keys.append(n_del)

                orig_data = list(zip(['GCMC'] * len(item_cats), orig_disparity[attr][d_gr].numpy(), item_cats))

                df_orig = pd.DataFrame(orig_data, columns=['Attribute', x, y]).dropna()
                df_exp = pd.DataFrame(exp_data, columns=['# Del Edges', x, y]).dropna()

                max_del_edges = max(bd_data)
                bin_map = {i: f"{e_type}: {i * bin_size + 1 if i != 0 else 1}-{(i + 1) * bin_size}" for i in
                           range(max_del_edges // bin_size + 1)}

                df_exp['# Del Edges lab'] = df_exp['# Del Edges'].map(lambda x: bin_map[x // bin_size])
                df_exp['# Del Edges'] = df_exp['# Del Edges'].map(lambda x: (x // bin_size) + 1) * 12

                if sort == "dots":
                    order = df_orig.sort_values(x)[y].to_list()
                    bar_data = pd.DataFrame(cats_inter_counts_attr[d_gr].items(), columns=[y, x])
                elif sort == "barplot_side":
                    order, vals = map(list, zip(*(sorted(cats_inter_counts_attr[d_gr].items(), key=lambda x: x[1])[::-1])))
                    bar_data = pd.DataFrame(zip(order, vals), columns=[y, x])

                g = sns.JointGrid(height=12, space=0.5)
                g.ax_marg_x.remove()
                sns.barplot(x=x, y=y, data=bar_data, ax=g.ax_marg_y, color="black", order=order)

                df_exp_plot = df_exp.groupby(['Category', '# Del Edges']).agg(**{
                    "Bias Disparity": pd.NamedAgg(column='Bias Disparity', aggfunc='mean'),
                    "# Del Edges lab": pd.NamedAgg(column='# Del Edges lab', aggfunc='first'),
                }).reset_index()

                print(df_exp_plot)

                sns.stripplot(x=x, y=y, color='black', data=df_orig, ax=g.ax_joint, s=12, marker="X", jitter=False,
                              label='GCMC', zorder=2, order=order)
                sns.scatterplot(x=x, y=y, hue="# Del Edges", size="# Del Edges", palette="colorblind",
                                data=df_exp_plot, ax=g.ax_joint, zorder=2, legend="full")  # , jitter=False, order=order)

                # lines_df = df_orig.set_index('Category').join(df_exp.set_index('Category'), lsuffix='_orig').loc[order]
                # lines_styles = ((lines_df[x + '_orig'].abs() - lines_df[x].abs()) < 0).map(lambda x: ':' if x else '-').values.tolist()
                #
                # lines_df['diff%'] = ((-(lines_df[x].abs() - lines_df[x + '_orig'].abs()) / lines_df[x + '_orig']) * 100).round(1)
                # lines_df['abs_diff'] = (lines_df[x + '_orig'] - lines_df[x]).abs().round(1)
                # lines_df['diff%'] = lines_df[['diff%', 'abs_diff']].apply(lambda row: f"{row['diff%']}% ({row['abs_diff']})", axis=1)
                # del lines_df['abs_diff']
                # for i, c in enumerate(lines_df.index):
                #     g.ax_joint.plot(lines_df.loc[c, [x + '_orig', x]], [c, c], 'k', zorder=1, ls=lines_styles[i])
                #     g.ax_joint.text(lines_df.loc[c, [x + '_orig', x]].mean(), i - 0.3, lines_df.loc[c, 'diff%'], ha='center')
                #
                # unique_labels = utils.legend_without_duplicate_labels(g.ax_joint)
                #
                # handles, labels = zip(*unique_labels)
                # leg_labels_order = sorted(range(len(labels)), key=lambda x: int(labels[x].split(': ')[1].split('-')[0]) if ':' in labels[x] else -1)
                # handles, labels = [handles[x] for x in leg_labels_order], [labels[x] for x in leg_labels_order]
                # g.ax_joint.legend(handles, labels)
                #
                handles, labels = zip(*utils.legend_without_duplicate_labels(g.ax_joint))
                df_exp_plot_sizes = df_exp_plot.set_index("# Del Edges")
                labels = [df_exp_plot_sizes.loc[int(l), "# Del Edges lab"].iloc[0] if l.isnumeric() else l for l in labels]
                g.ax_joint.legend(handles, labels)
                # for handle, label in zip(handles, labels):
                #     if label in df_exp_plot_sizes.index:
                #         handle.set_sizes([df_exp_plot_sizes.loc[label, "# Del Edges"].iloc[0]])

                g.ax_joint.plot([0., 0.], g.ax_joint.get_ylim(), 'k--', zorder=1)
                g.ax_joint.set_title(f"{attr.title()}: {d_gr}")
                g.ax_joint.grid(axis='y', ls=(0, (1, 3)))

                _ax_j = g.ax_joint.twinx()
                _ax_j.tick_params(right=False)
                _ax_j.set_yticklabels([])
                _ax_j.set_ylabel('# Interactions for each category of items', rotation=270, labelpad=15)
                # g.ax_marg_x.set_title('# Interactions for each category of items')

                plt.tight_layout()
                plt.savefig(os.path.join(plots_path, f'{d_gr}#dumbbell_over_del_edges_{e_type}.png'))
                plt.close()


def create_table_bias_disparity(bd, config_ids):
    sens_attributes = config['sensitive_attributes']
    sensitive_maps = {sens_attr: train_data.dataset.field2id_token[sens_attr] for sens_attr in sens_attributes}

    order = ['GCMC', 'GCMC+BD', 'GCMC+NDCG']

    for attr, sens_map in sensitive_maps.items():
        tables_path = os.path.join(get_plots_path(), 'comparison', '_'.join(config_ids), attr)
        if not os.path.exists(tables_path):
            os.makedirs(tables_path)

        vals = []
        for exp_type in order:
            vals.append([])
            for demo_gr in sens_map:
                data = bd[exp_type][attr][demo_gr]
                if exp_type == 'pred_explain' or data is None:
                    continue

                vals[-1].extend([np.nanmean(data.numpy()), np.nanstd(data.numpy())])

        d_grs = [x for x in sens_map if bd['GCMC'][attr][x] is not None]

        plot_vals = []
        for row in vals:
            plot_vals.append([])
            for i in range(len(d_grs)):
                plot_vals[-1].append(f"{row[2 * i]:.2f} ({row[2 * i + 1]:.2f})")

        df_attr = pd.DataFrame(plot_vals, columns=d_grs, index=order)
        df_attr.to_markdown(os.path.join(tables_path, f"comparison_table_{attr}.md"), tablefmt="github")


# %%
parser = argparse.ArgumentParser()
parser.add_argument('--model_file', required=True)
parser.add_argument('--explainer_config_file', default=os.path.join("config", "gcmc_explainer.yaml"))
parser.add_argument('--load_config_ids', nargs="+", type=int, default=[1, 1, 1],
                    help="follows the order ['Silvestri et al.', 'GCMC+BD', 'GCMC+NDCG'], set -1 to skip")
parser.add_argument('--best_exp_col', default="loss_total")

args = parser.parse_args(r"--model_file src\saved\GCMC-ML100K-Jun-01-2022_13-28-01.pth --explainer_config_file config\gcmc_explainer.yaml --load_config_ids -1 2 1".split())

script_path = os.path.abspath(os.path.dirname(inspect.getsourcefile(lambda: 0)))
script_path = os.path.join(script_path, 'src') if 'src' not in script_path else script_path

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
for c_id, exp_t, old_exp_t in zip(
    args.load_config_ids,
    ['Silvestri et al.', 'GCMC+BD', 'GCMC+NDCG'],
    ['pred_explain', 'FairBD', 'FairNDCGApprox']
):
    exp_paths[exp_t] = None
    if c_id != -1:
        exp_paths[exp_t] = os.path.join(script_path, 'explanations', dataset.dataset_name,
                                        old_exp_t, '_'.join(sens_attrs), f"epochs_{epochs}", str(batch_exp), str(c_id))

train_bias_ratio = utils.generate_bias_ratio(
    train_data,
    config,
    sensitive_attrs=config['sensitive_attributes'],
    mapped_keys=True
)

item_df = pd.DataFrame({
    'item_id': train_data.dataset.item_feat['item_id'].numpy(),
    'class': map(lambda x: [el for el in x if el != 0], train_data.dataset.item_feat['class'].numpy().tolist())
})

user_df = pd.DataFrame({
    'user_id': train_data.dataset.user_feat['user_id'].numpy(),
    **{sens_attr: train_data.dataset.user_feat[sens_attr].numpy() for sens_attr in sens_attrs}
})

train_df = pd.DataFrame(train_data.dataset.inter_feat.numpy())[["user_id", "item_id"]]
joint_df = train_df.join(item_df.set_index('item_id'), on='item_id').join(user_df.set_index('user_id'), on='user_id')

bias_disparity = extract_bias_disparity(exp_paths, train_bias_ratio, train_data, config, args.best_exp_col)

# %%
bd_all_data, n_users_data_all = extract_all_exp_bd_data(exp_paths, train_bias_ratio, train_data)

# create_table_bias_disparity(bias_disparity, list(map(str, args.load_config_ids)))
#
# plot_bias_disparity_diff_dumbbell(bias_disparity, sens_attrs, list(map(str, args.load_config_ids)), sort="barplot_side")

# plot_bias_disparity_boxplot(bias_disparity, sens_attrs, list(map(str, args.load_config_ids)))
# plot_explanations_fairness_trend(bd_data_all, n_users_data_all, bias_disparity['GCMC'], list(map(str, args.load_config_ids)))

    # %%
plot_explanations_fairness_trend_dumbbell(bd_all_data, bias_disparity['GCMC'], list(map(str, args.load_config_ids)),
                                          sort="barplot_side", bin_size=20)
