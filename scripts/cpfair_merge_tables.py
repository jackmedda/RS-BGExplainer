import os
import re
import math
import pickle
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mpl_lines
import matplotlib.colors as mpl_colors
import matplotlib.ticker as mpl_tickers
import matplotlib.patches as mpl_patches
import matplotlib.legend_handler as mpl_legend_handlers


def update_plt_rc():
    SMALL_SIZE = 16
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 20

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)


class HandlerEllipse(mpl_legend_handlers.HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = orig_handle.get_center()
        radius = orig_handle.get_radius()
        p = mpl_patches.Ellipse(
            xy=center, width=radius, height=radius
        )
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '--d', required=True)
    parser.add_argument('--plots_path_to_merge', '--pptm', default=os.path.join('scripts', 'cpfair_robust_plots'))
    parser.add_argument('--base_plots_path', '--bpp', default=os.path.join('scripts', 'cpfair_merged_plots'))
    parser.add_argument('--exclude', '--ex', nargs='+', help='Exclude certaing config ids', default=None)
    parser.add_argument('--hl_threshold', '--hl_th', type=float, default=10.0, help='Threshold to decide if a change in fairness estimate denotes a robust model')

    args = parser.parse_args()
    args.dataset = args.dataset.lower()
    args.exclude = args.exclude or []
    print(args)

    sns.set_context("paper")
    update_plt_rc()
    out_path = os.path.join(args.base_plots_path, args.dataset)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    def pval_symbol(pval):
        if pval < 0.01:
            return '^'
        elif pval < 0.05:
            return '*'

        return ''

    def remove_str_pval(str_pval):
        return str_pval.replace('^', '').replace('*', '')

    def hl(val):
        return "\hl{" + val + "}"

    dataset_order = ['INS', 'LF1K', 'ML1M']
    models_order = ['GCMC', 'LGCN', 'NGCF']
    setting_order = ['CP', 'CS', 'PE', 'PV']
    group_attr_order = ['Age', 'Gender', 'Pop']
    pert_type_order = ['Orig', '$\dotplus$ Del', '$\dotplus$ Add']

    loaded_dfs = []
    group_dfs = []
    dfs_across_exps = {'consumer': [], 'provider': []}
    edge_perturbation_impact = {}
    plots_path = os.path.join(args.plots_path_to_merge, args.dataset)
    for dirpath, dirnames, filenames in os.walk(plots_path):
        if filenames:
            for x in filenames:
                if x == 'DP_barplot.csv':
                    metadata = dirpath.split(args.dataset + os.sep)[1]
                    if 'consumer' in metadata:
                        mod, sk_holder, metric_loss, group_attribute, pert_type = metadata.split(os.sep)
                    else:
                        mod, sk_holder, metric_loss, pert_type = metadata.split(os.sep)
                        group_attribute = 'popularity'
                    metadata_map = {
                        'Dataset': args.dataset,
                        'Model': mod,
                        'stakeholder': sk_holder,
                        'MetricLoss': metric_loss,
                        'GroupAttribute': group_attribute,
                        'PerturbationType': pert_type
                    }
                    df = pd.read_csv(os.path.join(dirpath, x))

                    delta_col = df.columns[df.columns.str.contains('Delta')][0]
                    metric = delta_col.replace('$\Delta$', '')

                    df.rename(columns={delta_col: 'DP'}, inplace=True)
                    for key, val in metadata_map.items():
                        df[key] = val

                    rel_cols = ['Policy'] + list(metadata_map.keys())
                    loaded_dfs.append(df[rel_cols + ['DP', 'pvalue']])

                    g_df = df[df.columns[~df.columns.isin([delta_col, 'Split', 'DP', 'pvalue'])]].melt(rel_cols).rename(columns={
                        'variable': 'Metric', 'value': 'Value'
                    })
                    group_dfs.append(g_df)

                    if os.path.exists(os.path.join(dirpath, 'edge_perturbation_impact.pkl')):
                        with open(os.path.join(dirpath, 'edge_perturbation_impact.pkl'), 'rb') as f:
                            edge_perturbation_impact[
                                f"{args.dataset}_{mod}_{sk_holder}_{metric_loss}_{group_attribute}_{pert_type}"
                            ] = pickle.load(f)
                elif x == 'df_across_exps.csv':
                    metadata = dirpath.split(args.dataset + os.sep)[1]
                    if 'consumer' in metadata:
                        mod, sk_holder, metric_loss, group_attribute, pert_type = metadata.split(os.sep)
                    else:
                        mod, sk_holder, metric_loss, pert_type = metadata.split(os.sep)
                        group_attribute = 'popularity'

                    metadata_map = {
                        'Dataset': args.dataset,
                        'Model': mod,
                        'stakeholder': sk_holder,
                        'MetricLoss': metric_loss,
                        'GroupAttribute': group_attribute,
                        'PerturbationType': pert_type
                    }
                    df_across_exps = pd.read_csv(os.path.join(dirpath, x))

                    for key, val in metadata_map.items():
                        df_across_exps[key] = val

                    dfs_across_exps[sk_holder].append(df_across_exps)

    orig_pert_pval_data = []
    orig_pert_pval_cols = ['Dataset', 'Model', 'Stakeholder', 'MetricLoss', 'GroupAttribute', 'Perturbation Type', 'P_value']
    all_dsets_path = os.path.dirname(plots_path)
    for dirpath, dirnames, filenames in os.walk(all_dsets_path):
        if filenames:
            for x in filenames:
                if x == 'orig_pert_pval_dict.pkl':
                    metadata = dirpath.split(all_dsets_path + os.sep)[1]
                    if 'consumer' in metadata:
                        dset, mod, sk_holder, metric_loss, group_attribute, pert_type = metadata.split(os.sep)
                    else:
                        dset, mod, sk_holder, metric_loss, pert_type = metadata.split(os.sep)
                        group_attribute = "popularity"

                    with open(os.path.join(dirpath, 'orig_pert_pval_dict.pkl'), 'rb') as f:
                        pval = pickle.load(f)['Test']
                        orig_pert_pval_data.append(
                            [dset, mod, sk_holder, metric_loss, group_attribute, pert_type, pval]
                        )

    dataset_map = {
        'insurance': 'INS',
        'lastfm-1k': 'LF1K',
        'ml-1m': 'ML1M'
    }

    model_map = {
        'GCMC': 'GCMC',
        'LightGCN': 'LGCN',
        'NGCF': 'NGCF'
    }

    setting_map = {
        'consumer | ndcg': 'CP',
        'consumer | softmax': 'CS',
        'provider | exposure': 'PE',
        'provider | visibility': 'PV'
    }

    pert_type_map = {
        'Orig | deletion': 'Orig',
        'Orig | addition': 'Orig',
        'Perturbed | deletion': '$\dotplus$ Del',
        'Perturbed | addition': '$\dotplus$ Add'
    }

    group_attr_map = {
        'gender': 'Gender',
        'age': 'Age',
        'popularity': 'Pop'
    }

    first_df = pd.concat(loaded_dfs, ignore_index=True)
    orig_mask = first_df.Policy == 'Orig'
    first_df = pd.concat((
        first_df[~orig_mask],
        first_df[orig_mask].drop_duplicates(
            subset=['stakeholder', 'MetricLoss', 'GroupAttribute', 'Policy', 'Dataset', 'Model']
        )  # removes the duplicate Orig for deletion and addition
    ), ignore_index=True)

    first_df['Dataset'] = first_df['Dataset'].map(dataset_map)
    first_df['Model'] = first_df['Model'].map(model_map)
    first_df['GroupAttribute'] = first_df['GroupAttribute'].map(group_attr_map)
    first_df.sort_values(
        ['stakeholder', 'MetricLoss', 'GroupAttribute', 'Policy', 'PerturbationType']
    )
    first_df['Setting'] = first_df['stakeholder'].str.cat(first_df['MetricLoss'], sep=' | ')
    first_df['Setting'] = first_df['Setting'].map(setting_map)
    del first_df['stakeholder']
    del first_df['MetricLoss']

    first_df['PerturbationType'] = first_df['Policy'].str.cat(first_df['PerturbationType'], sep=' | ')
    first_df['PerturbationType'] = first_df['PerturbationType'].map(pert_type_map)
    del first_df['Policy']

    first_df['DP'] *= 100  # transform it to percentage
    first_pval_df = first_df.copy(deep=True)
    first_pval_df['DP'] = first_pval_df[['DP', 'pvalue']].apply(lambda row: f"{row['DP']:.2f}{pval_symbol(row['pvalue'])}", axis=1)
    del first_pval_df['pvalue']

    rel_delta_col = 'Rel. $Delta$ (%)'
    change_idx = ['Dataset', 'Model', 'Setting', 'GroupAttribute', 'PerturbationType']
    final_idx_df = first_pval_df.set_index(change_idx)
    for set_pert, gby_df in first_pval_df.groupby(change_idx[:-1]):
        gby_pt_df = gby_df.set_index('PerturbationType')
        orig_dp = float(remove_str_pval(gby_pt_df.loc['Orig', 'DP']))
        for pt in gby_pt_df.index:
            if pt != 'Orig':
                pt_dp = float(remove_str_pval(gby_pt_df.loc[pt, 'DP']))
                pt_impact = (pt_dp - orig_dp) / orig_dp * 100
                pt_impact_bound = " ($>$+100%)" if pt_impact > 100 else (" ($<$-100%)" if pt_impact < -100 else f" ({pt_impact:+05.1f}%)")

                final_idx_df.loc[tuple([*set_pert, pt]), 'DP'] += pt_impact_bound
                final_idx_df.loc[tuple([*set_pert, pt]), rel_delta_col] = f"{pt_impact:.2f}"

    final_df = final_idx_df.reset_index()

    pivot_df = final_df.pivot(
        columns=['Dataset', 'Model'],
        index=['Setting', 'GroupAttribute', 'PerturbationType'],
        values='DP'
    )

    pivot_df = pivot_df.reindex(
        dataset_order, axis=1, level=0
    ).reindex(
        models_order, axis=1, level=1
    ).reindex(
        setting_order, axis=0, level=0
    ).reindex(
        group_attr_order, axis=0, level=1
    ).reindex(
        pert_type_order, axis=0, level=2
    )

    pivot_df.to_csv(os.path.join(out_path, 'best_exp_change_table.csv'))

    merged_dfs_to_merge = []
    for dirpath, _, merged_csvs in os.walk(os.path.dirname(out_path)):
        if merged_csvs and '.ipynb_checkpoints' not in dirpath:
            for mc in merged_csvs:
                if mc == 'best_exp_change_table.csv':
                    merged_dfs_to_merge.append(pd.read_csv(os.path.join(dirpath, mc), index_col=[0,1,2], header=[0,1]))

    total_df = pd.concat(merged_dfs_to_merge, axis=1)

    set_gr_idx = list(set([x[:2] for x in total_df.index]))
    for col in total_df.columns:
        for sg_idx in set_gr_idx:
            col_setting_df = total_df.loc[sg_idx, col]
            to_hl = []
            for cs_df_idx in col_setting_df.index:
                if cs_df_idx != 'Orig':
                    change = float(re.search("([+-]\d+[.]\d+)|([+-]\d+)", col_setting_df.loc[cs_df_idx]).group(0))
                    if abs(change) < args.hl_threshold:
                        to_hl.append(True)
                    else:
                        to_hl.append(False)
            if all(to_hl):
                for cs_df_idx in col_setting_df.index:
                    if cs_df_idx != 'Orig':
                        col_setting_df.loc[cs_df_idx] = "\cellcolor{lightgray} " + col_setting_df.loc[cs_df_idx]

            for cs_df_idx in col_setting_df.index:
                if cs_df_idx != 'Orig':
                    val, ch = col_setting_df.loc[cs_df_idx].split('(')
                    col_setting_df.loc[cs_df_idx] = val + '(\textit{' + ch[:-1] + '})'

            # format cells with makecell
            # for cs_df_idx in col_setting_df.index:
            #     if cs_df_idx != 'Orig':
            #         left_val, right_val = col_setting_df.loc[cs_df_idx].split(' (')
            #         col_setting_df.loc[cs_df_idx] = "\makecell{" + left_val + r" \\ " + f"({right_val}" + "}"

    total_df.columns.names = ['' for x in total_df.columns.names]
    total_df.index.names = ['' for x in total_df.index.names]
    total_df.index = pd.MultiIndex.from_tuples([(f"{x[0]} - {x[1]}" if 'C' in x[0] else x[0], x[2]) for x in total_df.index])
    total_df.index = pd.MultiIndex.from_tuples([("\rotatebox[origin=c]{90}{" + x[0] + "}", x[1]) for x in total_df.index])

    # re.sub('(?<=\d)\*', '{\\\\scriptsize *}', aa)
    col_sep = "1.6cm"
    n_cols = total_df.columns.shape[0]
    with open(os.path.join(os.path.dirname(out_path), 'total_best_exp_change_table.tex'), 'w') as f:
        f.write(
            total_df.to_latex(
                column_format='cM{1.2cm}|' + ''.join(['M{' + col_sep + '}' + ('' if (i + 1) % 3 != 0 or (i + 1) == n_cols else '|') for i in range(n_cols)]),
                multicolumn_format='c',
                multirow=True,
                escape=False
            ).replace(
                '^', '\^{}'
            ).replace(
                '%', '\%'
            ).replace(
                '{*}', '{*}[-10pt]'
            ).replace(
                '{*}[-10pt]{\\rotatebox[origin=c]{90}{\\textbf{CS - Gender}', '{*}[-5pt]{\\rotatebox[origin=c]{90}{\\textbf{CS - Gender}'
            ).replace(
                '&       & \\multicolumn{3}{c}{INS}', '\\multicolumn{2}{c}{}       & \\multicolumn{3}{c}{INS}'
            ).replace(
                '\\cline{1-11}', '\\cline{2-11}\n\\cline{2-11}'
            ).replace(
                '\\bottomrule', '\\midrule\n\\bottomrule'
            ).replace(
                '\\toprule', '\\toprule\n\\midrule'
            )
        )

    # NEW RQ1
    P_VALUE_RQ1 = 0.05

    delta_pivot_df = final_df.pivot(
        columns=['Dataset', 'Model'],
        index=['Setting', 'GroupAttribute', 'PerturbationType'],
        values=rel_delta_col
    )

    delta_pivot_df = delta_pivot_df.reindex(
        dataset_order, axis=1, level=0
    ).reindex(
        models_order, axis=1, level=1
    ).reindex(
        setting_order, axis=0, level=0
    ).reindex(
        group_attr_order, axis=0, level=1
    ).reindex(
        pert_type_order, axis=0, level=2
    )

    delta_pivot_df.to_csv(os.path.join(out_path, 'best_exp_rel_delta_change_table.csv'))

    merged_delta_dfs_to_merge = []
    for dirpath, _, merged_csvs in os.walk(os.path.dirname(out_path)):
        if merged_csvs and '.ipynb_checkpoints' not in dirpath:
            for mc in merged_csvs:
                if mc == 'best_exp_rel_delta_change_table.csv':
                    merged_delta_dfs_to_merge.append(pd.read_csv(os.path.join(dirpath, mc), index_col=[0,1,2], header=[0,1]))

    delta_total_df = pd.concat(merged_delta_dfs_to_merge, axis=1)

    delta_total_idx = delta_total_df.index.map(lambda x: x if x[2] != 'Orig' else None)
    delta_total_df = delta_total_df.loc[delta_total_idx.dropna()].reset_index()

    delta_total_df = delta_total_df.melt(["Setting", "GroupAttribute", "PerturbationType"], value_name=rel_delta_col)
    delta_total_df["Stakeholder"] = delta_total_df["Setting"].map(lambda x: "Consumer" if "C" in x else "Provider")

    delta_total_df['FullSetting'] = delta_total_df['Setting'].str.cat(delta_total_df['GroupAttribute'], sep=' | ').map(lambda x: x.split(' | ')[0] if 'Pop' in x else x)
    delta_total_df['FullSetting'] = delta_total_df['FullSetting'].map(lambda x: f"{x.split(' | ')[0]} ({'A' if 'Age' in x else 'G'})" if ' | ' in x else x)
    delta_total_df.rename(columns={'PerturbationType': 'Perturbation Type'}, inplace=True)

    # delta_total_df = delta_total_df.pivot(
    #     columns="Model",
    #     index=["Dataset", "Setting", "GroupAttribute", "Perturbation Type", "Stakeholder", "FullSetting"],
    #     values=rel_delta_col
    # )

#     def pair_grid_sk_plotter(x, y, **kwargs):
#         consumer_mask = y.str.contains('C')
#         gridspec_row_start = plt.gca().get_subplotspec().rowspan.start

#         consumer_mask = consumer_mask if gridspec_row_start == 0 else ~consumer_mask
#         x = x[consumer_mask]
#         y = y[consumer_mask]

#         sns.stripplot(x, y, **kwargs)

    orig_pert_pval_df = pd.DataFrame(orig_pert_pval_data, columns=orig_pert_pval_cols)
    orig_pert_pval_df['Dataset'] = orig_pert_pval_df['Dataset'].map(dataset_map)
    orig_pert_pval_df['Model'] = orig_pert_pval_df['Model'].map(model_map)
    orig_pert_pval_df['GroupAttribute'] = orig_pert_pval_df['GroupAttribute'].map(group_attr_map)
    orig_pert_pval_df.sort_values(
        ['Stakeholder', 'MetricLoss', 'GroupAttribute', 'Perturbation Type']
    )
    orig_pert_pval_df['Setting'] = orig_pert_pval_df['Stakeholder'].str.cat(orig_pert_pval_df['MetricLoss'], sep=' | ')
    orig_pert_pval_df['Setting'] = orig_pert_pval_df['Setting'].map(setting_map)
    orig_pert_pval_df['Perturbation Type'] = orig_pert_pval_df['Perturbation Type'].map("Perturbed | {}".format)
    orig_pert_pval_df['Perturbation Type'] = orig_pert_pval_df['Perturbation Type'].map(pert_type_map)
    orig_pert_pval_df['Stakeholder'] = orig_pert_pval_df['Stakeholder'].str.title()
    del orig_pert_pval_df['MetricLoss']

    orig_pert_pval_df['FullSetting'] = orig_pert_pval_df['Setting'].str.cat(orig_pert_pval_df['GroupAttribute'], sep=' | ').map(lambda x: x.split(' | ')[0] if 'Pop' in x else x)
    orig_pert_pval_df['FullSetting'] = orig_pert_pval_df['FullSetting'].map(lambda x: f"{x.split(' | ')[0]} ({'A' if 'Age' in x else 'G'})" if ' | ' in x else x)

    pval_join_cols = ["Dataset", "Model", "FullSetting", "Perturbation Type"]
    delta_total_df = delta_total_df.join(orig_pert_pval_df[pval_join_cols + ["P_value"]].set_index(pval_join_cols), on=pval_join_cols)

    palette = dict(zip([pert_type_map['Perturbed | addition'], pert_type_map['Perturbed | deletion']], sns.color_palette("colorblind")))
    markers = dict(zip([pert_type_map['Perturbed | addition'], pert_type_map['Perturbed | deletion']], ["s", "o"]))
    p_val_hatch = '+'
    markersize=20

    sns.set_context("paper")
    update_plt_rc()
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=16)

    delta_total_df_gby = delta_total_df.groupby("Dataset")
    for dset in dataset_order:
        delta_total_dset_df = delta_total_df_gby.get_group(dset)
        delta_total_dset_df_gby = delta_total_dset_df.groupby("Stakeholder")

        handles, labels = None, None
        fig = plt.figure(figsize=(6 * len(models_order), 3), constrained_layout=True)
        height_ratios = (2, 1)
        gs = fig.add_gridspec(
            2, len(models_order),  height_ratios=height_ratios,
            # left=0.1, right=0.9, bottom=0.1, top=0.9,
            wspace=0.01
        )

        for plot_i, sk in enumerate(["Consumer", "Provider"]):
            delta_total_sk_df = delta_total_dset_df_gby.get_group(sk)
            delta_total_sk_gby = delta_total_sk_df.groupby("Model")
            for plot_j, mod in enumerate(models_order):
                delta_total_mod_df = delta_total_sk_gby.get_group(mod)

                kws = {}
                if plot_i > 0:
                    kws['sharex'] = fig.axes[plot_j]
                if plot_j > 0:
                    kws['sharey'] = fig.axes[gs.ncols * plot_i]
                ax = fig.add_subplot(gs[plot_i, plot_j], **kws)

                for pt_idx, (pt, delta_pt_df) in enumerate(delta_total_mod_df.groupby("Perturbation Type")):
                    sns.stripplot(
                        data=delta_pt_df, x="Rel. $Delta$ (%)", y="FullSetting", hue="Perturbation Type", palette={pt: palette[pt]},
                        marker=markers[pt], size=markersize, orient="h", jitter=False, linewidth=1, edgecolor="w", ax=ax
                    )

                    pt_collections = ax.collections[(pt_idx * len(delta_pt_df) + 1 * pt_idx):-1]
                    for collection, pval in zip(pt_collections, delta_pt_df['P_value'].values):
                        if pval < P_VALUE_RQ1:
                            collection.set_hatch(p_val_hatch)

                if handles is None and labels is None:
                    handles, labels = ax.get_legend_handles_labels()
                ax.get_legend().remove()

        for row in range(gs.nrows):
            for col in range(gs.ncols):
                ax = fig.axes[row * gs.ncols + col]
                if sk == 'Consumer':
                    ax.tick_params(labelbottom=False)
                    if col == 0:
                        ax.set(title=models_order[row])

                if row < (gs.nrows - 1):
                    ax.tick_params(bottom=False, labelbottom=False)
                if col > 0:
                    ax.tick_params(left=False, labelleft=False)

                if dset == dataset_order[0] and row == 0:
                    ax.set_title(models_order[col])
                ax.set(xlabel='', ylabel='', xscale='symlog')
                ax.xaxis.grid(True)
                ax.yaxis.grid(True)
                ax.autoscale(False)
                xlim = ax.get_xlim()
                ax.set_xlim((xlim[0] * (0.2 if dset == "INS" else 1), xlim[1] * 1.5))
                if dset == "ML1M":
                    ax.set_xlim((xlim[0] * 0.9, xlim[1] * 1.48))
                ax.plot([0, 0], list(ax.get_ylim()), "k", ls='-', lw=3, clip_on=False)

        handles, labels = map(list, zip(*[(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]))
        handles = [mpl_lines.Line2D([], [], color=hand.get_facecolor(), marker=markers[lab], linestyle='None', markersize=markersize) for hand, lab in zip(handles, labels)]
        handles.extend(
            [
                mpl_lines.Line2D([], [], alpha=0),
                mpl_patches.Circle((20, 4), radius=20, facecolor="w", edgecolor="k", hatch=p_val_hatch),
                mpl_patches.Circle((20, 4), radius=20, facecolor="w", edgecolor="k")
            ]
        )
        labels.extend(["Wilcoxon signed-rank\n test on $\Delta$ with 5% CI", "Significant", "Not Significant"])
        handles = [mpl_lines.Line2D([], [], alpha=0), *handles]
        labels = ['Perturbation\n      Type', *labels]

        figlegend = plt.figure(figsize=(len(labels), 1))
        figlegend.legend(
            handles, labels, loc='center', frameon=False, fontsize=15, ncol=len(labels),
            handler_map={mpl_patches.Circle: HandlerEllipse()}
            # markerscale=1.8, handletextpad=0.1, columnspacing=1, borderpad=0.1
        )
        figlegend.savefig(os.path.join(args.base_plots_path, 'legend_delta_dotplot.png'), dpi=300, bbox_inches="tight", pad_inches=0)

        fig.savefig(os.path.join(args.base_plots_path, f'{dset}_delta_dotplot.png'), dpi=300, bbox_inches="tight", pad_inches=0)

    # RQ2
    full_cols = ['DP', 'Policy', 'orig_pert_pvalue', '% Pert Edges', 'Dataset', 'Model', 'stakeholder', 'MetricLoss', 'GroupAttribute', 'PerturbationType']
    dfs_across_exps = {k: pd.concat(v, ignore_index=True) for k, v in dfs_across_exps.items()}

    full_df = pd.concat([_df[full_cols] for _df in dfs_across_exps.values()], ignore_index=True)
    full_df.rename(columns={'% Pert Edges': 'Perturbed Edges (%)'}, inplace=True)
    #full_df['DP'] *= 100
    full_df['Model'] = full_df['Model'].map(model_map)
    full_df['Dataset'] = full_df['Dataset'].map(dataset_map)
    full_df['GroupAttribute'] = full_df['GroupAttribute'].map(group_attr_map)
    full_df['PerturbationType'] = full_df['PerturbationType'].map({'addition': '$\dotplus$ Add', 'deletion': '$\dotplus$ Del'})

    full_df['Setting'] = full_df['stakeholder'].str.cat(full_df['MetricLoss'], sep=' | ')
    full_df['Setting'] = full_df['Setting'].map(setting_map)

    full_df['FullSetting'] = full_df['Setting'].str.cat(full_df['GroupAttribute'], sep='\n').map(lambda x: x.split('\n')[0] if 'Pop' in x else x)
    full_setting_dtype = pd.api.types.CategoricalDtype(
        categories=['CP\nAge', 'CP\nGender', 'CS\nAge', 'CS\nGender', 'PE', 'PV']
    )
    full_df['FullSetting'] = full_df['FullSetting'].astype(full_setting_dtype)

    sns.set_theme(style="whitegrid", palette="colorblind", rc={"axes.spines.right": False, "axes.spines.top": False})
    plt.rc("axes.grid", axis="y")
    plt.grid(ls=':')
    sns.set_context("paper")
    update_plt_rc()

    def select_best_sizes_mapping(_hdls, _labs, s_mapper):
        pt2_idx = _labs[1:].index('Perturbation\nType') + 1
        fixed_idxs = [0, 1, pt2_idx + 1, pt2_idx + 2, 2]

        _hdls = [_hdls[i] for i in fixed_idxs]
        _labs = [_labs[i] for i in fixed_idxs]

        space = pt2_idx - 3
        size_lookup_keys, _ = zip(*list(s_mapper.items()))
        size_lookup_keys = np.array(size_lookup_keys)
        size_lookup_keys.sort()

        # first point in legend is 5%
        first_point_shifter = size_lookup_keys[size_lookup_keys > 0.01].shape[0]
        # last point is 85% of the scale
        last_point_shifter = size_lookup_keys.shape[0] - round(size_lookup_keys.shape[0] * 0.9)

        s_idxs = np.linspace(0, first_point_shifter - 1 - last_point_shifter, space, dtype=int)
        s_idxs = s_idxs + (size_lookup_keys.shape[0] - first_point_shifter)
        for _idx in s_idxs:
            _hdls.append(mpl_lines.Line2D([0], [0], marker='o', lw=0, markersize=np.sqrt(s_mapper[size_lookup_keys[_idx]]), color='k'))
            _labs.append(f"{round(float(size_lookup_keys[_idx]) * 100)}%")

        return _hdls, _labs

    orig_line_lw = 2.5
    orig_info_df = first_df[first_df['PerturbationType'] == 'Orig'].set_index(['Dataset', 'Model', 'Setting', 'GroupAttribute'])
    delta_over_edges_df = full_df.join(orig_info_df['DP'], on=orig_info_df.index.names, rsuffix="_Orig")
    delta_over_edges_df['$\Delta$'] = (delta_over_edges_df.DP * 100 - delta_over_edges_df.DP_Orig)
    delta_over_edges_df_gby = delta_over_edges_df.groupby('Dataset')
    for dset in dataset_order:
        if not delta_over_edges_df['Dataset'].str.contains(dset).any():
            continue
        dset_df = delta_over_edges_df_gby.get_group(dset)

        fig = plt.figure(figsize=(18, 1.6 * len(models_order)), constrained_layout=True)
        width_ratios = (4, 4, 4, 2, 2, 2)
        gs = fig.add_gridspec(
            1, 6,  width_ratios=width_ratios,
            # left=0.1, right=0.9, bottom=0.1, top=0.9,
            wspace=0.01
        )

        # manually set as the maximum across all datasets => the maximum is based on INS, we then save only the legend created by INS
        size_norm = mpl_colors.Normalize(vmin=0, vmax=0.6583932070542129)

        size_mapper = {}
        axs = {"C": [], "P": []}
        for i, mod in enumerate(models_order):
            dset_mod_df = dset_df[dset_df['Model'] == mod]

            for sk in ["C", "P"]:
                kws = {'sharey': axs[sk][0]} if axs[sk] else {}
                if sk == "C":
                    ax = fig.add_subplot(gs[i], **kws)
                else:
                    ax = fig.add_subplot(gs[3 + i], **kws)
                axs[sk].append(ax)

                sk_dset_mod_df = dset_mod_df[dset_mod_df['Setting'].str[0] == sk].copy(deep=True)
                sk_dset_mod_df["FullSetting"] = sk_dset_mod_df.FullSetting.cat.remove_unused_categories()
                sk_dset_mod_df['FullSetting_codes'] = sk_dset_mod_df['FullSetting'].cat.codes + np.random.uniform(-0.2, 0.2, len(sk_dset_mod_df))
                sk_dset_mod_df.rename(columns={'PerturbationType': 'Perturbation\nType', 'Perturbed Edges (%)': 'Perturbed\nEdges (%)'}, inplace=True)

                for zorder_i, (pt, alpha) in enumerate(zip([pert_type_map['Perturbed | addition'], pert_type_map['Perturbed | deletion']], [1, 0.5])):
                    sns.scatterplot(
                        x='FullSetting_codes',
                        y='DP',
                        hue='Perturbation\nType',
                        palette=palette,
                        size='Perturbed\nEdges (%)',
                        sizes=(30, 250),
                        size_norm=size_norm,
                        data=sk_dset_mod_df[sk_dset_mod_df['Perturbation\nType'] == pt],
                        alpha=alpha,
                        zorder=zorder_i,
                        ax=ax
                    )

                    p = sns.relational._ScatterPlotter(
                        data=sk_dset_mod_df[sk_dset_mod_df['Perturbation\nType'] == pt],
                        variables=dict(
                            x='FullSetting_codes',
                            y='DP',
                            hue='Perturbation\nType',
                            size='Perturbed\nEdges (%)',
                            style=None
                        ),
                        legend="auto"
                    )

                    p.map_size(sizes=(30, 250), order=None, norm=size_norm)
                    size_mapper.update(p._size_map.lookup_table)

                ax.set_xticks(
                    np.arange(len(sk_dset_mod_df.FullSetting.cat.categories)),
                    sk_dset_mod_df.FullSetting.cat.categories.map(lambda x: x.replace('Age', '(A)').replace('Gender', '(G)'))
                )
                ax.set_xlabel('')
                if i != 0:
                    ax.tick_params(left=False, labelleft=False)
                    ax.set_ylabel('')

                if dset != 'ML1M':
                    ax.tick_params(bottom=False, labelbottom=False)
                if dset == 'INS':
                    ax.set_title(mod)

                for (sett, gr), sett_gr_df in sk_dset_mod_df.groupby(['Setting', 'GroupAttribute']):
                    orig_dp = orig_info_df.loc[(dset, mod, sett, gr), 'DP'] / 100  # we don't use percentage as earlier
                    full_sett = f"{sett}\n{gr}" if gr != 'Pop' else sett
                    sett_idx = (sk_dset_mod_df['FullSetting'].cat.categories == full_sett).nonzero()[0].item()

                    ax.plot([sett_idx - 0.4, sett_idx + 0.4], [orig_dp, orig_dp], 'k--', lw=orig_line_lw)

        handles, labels = axs["C"][0].get_legend_handles_labels()
        del_idx = labels.index('$\dotplus$ Del')
        labels.insert(del_idx + 1, 'Orig')
        handles.insert(del_idx + 1, mpl_lines.Line2D([0], [0], color='k', ls='--', lw=orig_line_lw))

        if dset == "INS":
            figlegend = plt.figure(figsize=(len(labels), 1))
            handles, labels = select_best_sizes_mapping(handles, labels, size_mapper)

            handles[1].set_sizes(handles[1].get_sizes() * 10)
            handles[2].set_sizes(handles[2].get_sizes() * 10)
            labels[3] = '  ' + labels[3]

            labels[0] = 'Perturbation\n      Type'
            figlegend.legend(
                handles, labels, loc='center', frameon=False, fontsize=15, ncol=len(labels),
                markerscale=1, handletextpad=0.1, columnspacing=1, borderpad=0.1
            )
            figlegend.savefig(os.path.join(args.base_plots_path, 'legend_DP_over_pert_edges.png'), dpi=300, bbox_inches="tight", pad_inches=0)

        for _ax in axs["C"] + axs["P"]:
            _ax.get_legend().remove()

        # fig.supylabel(dset)
        # fig.tight_layout()
        fig.savefig(os.path.join(args.base_plots_path, f'{dset}_DP_over_pert_edges_scatterplot.png'), pad_inches=0, bbox_inches="tight", dpi=300)

    # RQ2 (2)
    # delta_over_edges_df = full_df.join(orig_info_df['DP'], on=orig_info_df.index.names, rsuffix="_Orig")
    # delta_over_edges_df['$\Delta$'] = (delta_over_edges_df.DP * 100 - delta_over_edges_df.DP_Orig) # / delta_over_edges_df['Perturbed Edges (%)']
    # sns.lmplot(
    #     data=delta_over_edges_df, y='$\Delta$', x='Perturbed Edges (%)', col='Model', row='Dataset', hue='FullSetting',
    #     height=8, aspect=3, facet_kws=dict(sharex=False, sharey=True)
    # )
    # plt.savefig(f'{args.dataset}_lmplot_woo.png', dpi=300, pad_inches=0, bbox_inches='tight')

    #RQ3
    pt_map = {'addition': '$\dotplus$ Add', 'deletion': '$\dotplus$ Del', 'Orig': 'Orig'}

    grs_df = pd.concat(group_dfs, ignore_index=True)
    grs_df['Model'] = grs_df['Model'].map(model_map)
    grs_df['Dataset'] = grs_df['Dataset'].map(dataset_map)
    grs_df['GroupAttribute'] = grs_df['GroupAttribute'].map(group_attr_map)
    grs_df['PerturbationType'] = grs_df['PerturbationType'].map({k: v for k, v in pt_map.items() if k != 'Orig'})

    per_gr_df = grs_df[grs_df.Metric.isin(['SH', 'LT', 'Y', 'O', 'M', 'F'])]
    pert_per_gr_df = per_gr_df[per_gr_df.Policy == 'Perturbed']
    del pert_per_gr_df['Policy']

    pert_per_gr_df_idx = pert_per_gr_df.set_index(['Dataset', 'Model', 'stakeholder', 'MetricLoss', 'GroupAttribute', 'PerturbationType', 'Metric'])
    for key, group_ei in edge_perturbation_impact.items():
        for group_ei_gr, ei in group_ei.items():
            dset, mod, sk, m_loss, gr_attr, pt = key.split('_')
            dset = dataset_map[dset]
            mod = model_map[mod]
            gr_attr = group_attr_map[gr_attr]
            pt = pt_map[pt]
            pert_per_gr_df_idx.loc[(dset, mod, sk, m_loss, gr_attr, pt, group_ei_gr), 'EI'] = ei
    ei_df = pert_per_gr_df_idx.reset_index()

    orig_per_gr_df = per_gr_df[per_gr_df.Policy == 'Orig']
    del orig_per_gr_df['Policy']
    orig_per_gr_df = orig_per_gr_df[orig_per_gr_df.PerturbationType == '$\dotplus$ Add']  # Orig values are independent of the perturbation type
    del orig_per_gr_df['PerturbationType']

    adv_orig_grs_df = orig_per_gr_df.groupby(['Dataset', 'Model', 'stakeholder', 'MetricLoss', 'GroupAttribute']).max().set_index('Metric', append=True)
    ei_adv_idx_df = ei_df.set_index(['Dataset', 'Model', 'stakeholder', 'MetricLoss', 'GroupAttribute', 'Metric'])
    ei_adv_idx_df.loc[adv_orig_grs_df.index, 'Advantaged'] = True
    ei_adv_df = ei_adv_idx_df.reset_index()
    ei_adv_df.fillna(False, inplace=True)

    ei_adv_df['Setting'] = ei_adv_df['stakeholder'].str.cat(ei_adv_df['MetricLoss'], sep=' | ')
    ei_adv_df['Setting'] = ei_adv_df['Setting'].map(setting_map)
    del ei_adv_df['stakeholder']
    del ei_adv_df['MetricLoss']

    # hh = ei_adv_idx_df.reset_index().dropna(subset=['EI']).fillna(False).groupby(
    #     ['Dataset', 'Model', 'Setting', 'GroupAttribute', 'PerturbationType']
    # ).filter(
    #     lambda x: np.isin([True, False], x['Advantaged']).all()
    # ).groupby(
    #     ['Dataset', 'Model', 'Setting', 'GroupAttribute', 'PerturbationType']
    # ).apply(
    #     lambda x: x.loc[x['Advantaged'], 'EI'].item() - x.loc[~x['Advantaged'], 'EI'].item()
    # ).reset_index().rename(columns={0: '$\Delta$EI'})

    delta_ei_df = ei_adv_df.groupby(['Dataset', 'Model', 'Setting', 'GroupAttribute', 'PerturbationType']).apply(
        lambda x: x.loc[x['Advantaged'], 'EI'].item() - x.loc[~x['Advantaged'], 'EI'].item()
    ).reset_index().rename(columns={0: '$\Delta$EI'})

    def add_delta(_df):
        _orig_mask = _df['PerturbationType'] == 'Orig'
        _orig_dp = _df.loc[_orig_mask, 'DP'].item()
        _df['$\Delta$'] = _df['DP'] - _orig_dp
        return _df.loc[~_orig_mask]

    join_idx = ['Dataset', 'Model', 'Setting', 'GroupAttribute', 'PerturbationType']
    first_delta_df = first_df.groupby(join_idx[:-1]).apply(add_delta)

    delta_ei_dp_df = delta_ei_df.join(first_delta_df.set_index(join_idx), on=join_idx, how='left')
    delta_ei_dp_df['FullSetting'] = delta_ei_dp_df['Setting'].str.cat(
        delta_ei_dp_df['GroupAttribute'], sep='\n'
    ).map(
        lambda x: x.split('\n')[0] if 'Pop' in x else '_{'.join(x.split('\n')) + '}'
    )
    delta_ei_dtype = pd.api.types.CategoricalDtype(
        categories=['CP_{Age}', 'CP_{Gender}', 'CS_{Age}', 'CS_{Gender}', 'PE', 'PV']
    )
    delta_ei_dp_df['FullSetting'] = delta_ei_dp_df['FullSetting'].astype(delta_ei_dtype)
    del delta_ei_dp_df['pvalue']

    delta_ei_dp_df.to_csv(os.path.join(args.base_plots_path, f'{dset}_delta_EI_DP.csv'), index=False)
