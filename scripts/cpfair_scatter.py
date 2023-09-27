import os
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_tick
import matplotlib.transforms as mpl_trans


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--merged_plots_path', '--mpp', default=os.path.join("scripts", "cpfair_merged_plots"))

    df = pd.concat([f.path for f in os.scandir(args.merged_plots_path) if '_delta_EI_DP.csv' in f.name], ignore_index=True)

    args = parser.parse_args()
    print(args)

    i = 0
    handles, labels = None, None
    fig, axs = plt.subplots(3, 3, sharey=row, figsize=(15, 15))
    df_idx = df.set_index(["Dataset", "Model"])
    for dset in ["INS", "LF1K", "ML1M"]:
        for mod in ["GCMC", "LGCN", "NGCF"]:
            plot_df = df_idx.loc[(dset, mod)]
            import pdb; pdb.set_trace()

            # style_order = dict(zip(sens_df['Model'].values, ['o', 'v', '^', '<', '+', 'x', '*', 's', 'p', 'D']))
            # style_order = dict(zip(sens_df['Model'].values, ['$\mathbf{' + str(x) + '}$' for x in range(10)]))
            # style_order = dict(zip(sens_df['Model'].values, ['o', 's', 'd', 'X', 's', 'd', 'X', 's', 'd', 'X']))
            style_order = dict(zip(plot_df['Setting'].values, ['o', 'P', 'P', 'P', 'X', 'X', 'X', '*', '*', '*']))
            s = 250
            marker = '*'

            scatter_kws = dict(
                x='Rel. Diff. NDCG', y=r'Rel. Diff. $\Delta$NDCG', data=plot_df,
                hue='Paper',
                # s=s
                # marker=marker,
                style='Paper',
                markers=style_order,
                size='Paper',
                sizes=dict(zip(plot_df['Paper'].values, [17**2 if p == 'Ours' else (17**2)/1.7 for p in plot_df['Paper']]))
            )

            sns.scatterplot(
                ax=axs[i], **scatter_kws
            )

            if handles is None and labels is None:
                handles, labels = axs[i].get_legend_handles_labels()
            axs[i].get_legend().remove()

            axs[i].plot(1, 0, ">k", transform=axs[i].get_yaxis_transform(), clip_on=False)
            axs[i].plot(0, 1, "^k", transform=axs[i].get_xaxis_transform(), clip_on=False)

            axs[i].tick_params(axis='x', direction='in', pad=-15)
            axs[i].spines[['left', 'bottom']].set_position('zero')
            axs[i].spines[['top', 'right']].set_visible(False)

            axs[i].set_ylabel('')
            plt.text(0.5, -0.05, axs[i].get_xlabel(), ha='center', va='center', fontsize=12, transform=axs[i].transAxes)
            axs[i].set_xlabel('')

            i += 1

            # axs[i].xaxis.set_major_formatter(lambda x, pos: f'{x*100:.1f}%')
            # axs[i].yaxis.set_major_formatter(lambda x, pos: f'{x*100:.1f}%')

            # axs[i].set_title(f"{dset}: {sens_attr}", pad=12)

    # fig.supxlabel('Rel. Diff. NDCG')
    # fig.supylabel('Rel. Diff. $\Delta$NDCG')

    fig.text(-0.1, 0.4, 'Rel. Diff. $\Delta$NDCG', ha='center', va='center', rotation=90, fontsize=12, transform=axs[0].transAxes)
    fig.legend(handles, labels, loc='upper center', ncol=len(plot_df.Paper.unique()), bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    fig.savefig('aaa.png', dpi=250, bbox_inches="tight", pad_inches=0)

    figlegend = plt.figure(figsize=(4, 1))
    figlegend.legend(handles, labels, loc='center', frameon=False, prop={'size': 10}, ncol=len(plot_df.Paper.unique()))  # labelspacing=4, ncol=1)# ncol=len(plot_df.Paper.unique()))
    figlegend.savefig(os.path.join('scripts', 'legend.png'), dpi=300, bbox_inches="tight", pad_inches=0)

    i = 0
    axs[0].set_ylabel("")
    for dset in ["ML-1M", "LFM-1K"]:
        for sens_attr in ["Gender", "Age"]:
            extent = axs[i].get_window_extent().transformed(fig.dpi_scale_trans.inverted())

            # bbox_args = [1.7, 1.1] if i == 0 else [1.1, 1.1]
            width = extent.width
            height = extent.height
            deltaw = (1.04 * width - width) / 2.0
            deltah = (0.96 * height - height) / 2.0
            offsetw = 0.6 if i == 3 else -0.2
            offseth = -1.65
            a = np.array([[-deltaw - deltaw * offsetw, -deltah], [deltaw, deltah + deltah * offseth]])
            new_bbox = extent._points + a
            fig.savefig(os.path.join('scripts', f'ax_{dset}_{sens_attr}_expanded.png'), bbox_inches=mpl_trans.Bbox(new_bbox), dpi=300)
            i += 1
