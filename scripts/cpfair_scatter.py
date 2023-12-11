import os
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_tick
import matplotlib.ticker as mpl_tickers
import matplotlib.transforms as mpl_trans


def update_plt_rc():
    SMALL_SIZE = 16
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 20

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--merged_plots_path', '--mpp', default=os.path.join("scripts", "cpfair_merged_plots"))

    args = parser.parse_args()
    print(args)

    full_setting_map = {
        'CP_{Age}': 'CP (A)',
        'CP_{Gender}': 'CP (G)',
        'CS_{Age}': 'CS (A)',
        'CS_{Gender}': 'CS (G)',
        'PE': 'PE',
        'PV': 'PV'
    }

    sns.set_context('paper')
    update_plt_rc()

    dataset_order = ['INS', 'LF1K', 'ML1M']
    models_order = ['GCMC', 'LGCN', 'NGCF']
    setting_order = ['CP', 'CS', 'PE', 'PV']
    group_attr_order = ['Age', 'Gender', 'Pop']
    pert_type_order = ['Orig', '$\dotplus$ Del', '$\dotplus$ Add']

    df = pd.concat([pd.read_csv(f.path) for f in os.scandir(args.merged_plots_path) if '_delta_EI_DP.csv' in f.name], ignore_index=True)

    df['FullSetting'] = df['FullSetting'].map(full_setting_map)
    full_setting_dtype = pd.api.types.CategoricalDtype(categories=['CP (A)', 'CP (G)', 'CS (A)', 'CS (G)', 'PE', 'PV'])
    df['FullSetting'] = df['FullSetting'].astype(full_setting_dtype)

    df['DP'] /= 100
    df['$\Delta$'] /= 100

    handles, labels = None, None
    fig = plt.figure(figsize=(6 * len(models_order), 3 * len(models_order)), constrained_layout=True)
    width_ratios = (4, 4, 4, 2, 2, 2)
    gs = fig.add_gridspec(
        3, 6,  width_ratios=width_ratios,
        # left=0.1, right=0.9, bottom=0.1, top=0.9,
        wspace=0.15, hspace=0.15
    )

    axs = {"C": [], "P": []}
    df_idx = df.set_index(["Dataset", "Model"])
    for i, dset in enumerate(dataset_order):
        for sk_axs in axs.values():
            sk_axs.append([])
        for j, mod in enumerate(models_order):
            for sk in ["C", "P"]:
                kws = {}
                if j > 0:
                    kws['sharey'] = axs[sk][i][0]
                if i > 0:
                    kws['sharex'] = axs[sk][0][j]
                if sk == "C":
                    ax = fig.add_subplot(gs[i, j], **kws)
                else:
                    ax = fig.add_subplot(gs[i, 3 + j], **kws)

                axs[sk][i].append(ax)

                plot_df = df_idx.loc[(dset, mod)].copy(deep=True)
                plot_df = plot_df[plot_df["Setting"].str.startswith(sk)]

                # style_order = dict(zip(sens_df['Model'].values, ['o', 'v', '^', '<', '+', 'x', '*', 's', 'p', 'D']))
                # style_order = dict(zip(sens_df['Model'].values, ['$\mathbf{' + str(x) + '}$' for x in range(10)]))
                # style_order = dict(zip(sens_df['Model'].values, ['o', 's', 'd', 'X', 's', 'd', 'X', 's', 'd', 'X']))
                style_order = dict(zip(
                    plot_df['FullSetting'].cat.categories, ['$\maltese$', r'$\clubsuit$', r's', r'^', 'P', 'X']
                ))
                plot_df.rename(columns={'PerturbationType': 'Perturbation\n      Type', 'FullSetting': 'Fairness\n Setting'}, inplace=True)

                base_size = 21
                scatter_kws = dict(
                    x='$\Delta$', y=r'$\Delta$EI', data=plot_df,
                    hue='Perturbation\n      Type',
                    # s=s,
                    # marker=marker,
                    palette='colorblind',
                    style='Fairness\n Setting',
                    markers=style_order,
                    size='Fairness\n Setting',
                    sizes=dict(zip(plot_df['Fairness\n Setting'].cat.categories, [base_size**2] + [(base_size**2)/1.3] * 5))
                )

                sns.scatterplot(
                    ax=ax, **scatter_kws
                )

                if handles is None and labels is None:
                    handles, labels = ax.get_legend_handles_labels()
                ax.get_legend().remove()

                ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), linewidth=200, clip_on=False)
                ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), linewidth=200, clip_on=False)

                # ax.tick_params(axis='x', direction='in', pad=-15)
                ax.spines[['left', 'bottom']].set_position('zero')
                ax.spines[['top', 'right']].set_visible(False)

                ax.xaxis.grid(True, ls=':', lw=2)
                ax.yaxis.grid(True, ls=':', lw=2)

                if i == (len(dataset_order) - 1):
                    labelpad = 105 if sk == "C" else 30
                    ax.set_xlabel(ax.get_xlabel(), labelpad=labelpad)
                else:
                    ax.set_xlabel('')
                if j > 0 or sk == "P":
                    ax.set_ylabel('')
                else:
                    ax.set_ylabel(dset)
    #             plt.text(0.5, -0.05, axs[i, j].get_xlabel(), ha='center', va='center', fontsize=12, transform=axs[i, j].transAxes)
    #             axs[i, j].set_xlabel('')
                if i == 0:
                    ax.set_title(mod)
                if i < (len(dataset_order) - 1):
                    ax.tick_params(labelbottom=False)
                if j > 0:
                    ax.tick_params(labelleft=False)

                ax.xaxis.set_major_locator(mpl_tickers.MaxNLocator(4 if sk == "C" else 2))

    fig.savefig(os.path.join(args.merged_plots_path, 'cpfair_scatter_total.png'), dpi=300, bbox_inches="tight", pad_inches=0)

    figlegend = plt.figure(figsize=(4, 1))
    handles[1].set_sizes(handles[-1].get_sizes())
    handles[2].set_sizes(handles[-1].get_sizes())
    figlegend.legend(handles, labels, loc='center', frameon=False, prop={'size': 10}, ncol=len(labels), markerscale=0.8)
    figlegend.savefig(os.path.join(args.merged_plots_path, 'cpfair_scatter_legend.png'), dpi=300, bbox_inches="tight", pad_inches=0)
