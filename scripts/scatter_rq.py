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
    parser.add_argument('--full_table_path', '--tp', default=os.path.join("scripts", "full_table.csv"))
    parser.add_argument('--merged_plots_path', '--mpp', default=os.path.join("scripts", "merged_plots"))

    args = parser.parse_args()
    print(args)

    df = pd.read_csv(args.full_table_path, header=None)

    new_cols = [("Model", "", "")]
    for dset in ['ML-1M', 'LFM-1K']:
        new_cols.append((dset, "Policy", ""))
        for metr in ['NDCG', '$\Delta$NDCG']:
            for state in ['Base', 'Mit']:
                new_cols.append((dset, metr, state))
    df.columns = pd.MultiIndex.from_tuples(new_cols)

    df.loc[:9, 'Sens Attr'] = 'Gender'
    df.loc[10:, 'Sens Attr'] = 'Age'

    df = df.astype({x: float for x in df.columns if 'Base' in x or 'Mit' in x})
    df = df.set_index(['Model', 'Sens Attr'])
    print(df)
    for dset, dset_tab in zip(["ml-1m", "lastfm-1k"], ["ML-1M", "LFM-1K"]):
        for sens_attr in ["gender", "age"]:
            ecir_df = pd.read_csv(
                os.path.join(args.merged_plots_path, dset, sens_attr, 'ecir_table_GCMC_LightGCN_NGCF.csv'),
                index_col=0, header=[0,1]
            )
            ecir_df = ecir_df.applymap(
                lambda x: float(x.replace('^', '').replace('*', '')) if isinstance(x, str) and 'CN' not in x else x
            )
            ecir_df["Policy"] = ecir_df.Policy.applymap(lambda x: 'B' if x == 'CN' else x.replace('CN ', 'B ').replace('CN+', ''))

            for mod, mod_row in ecir_df.iterrows():
                df.loc[(mod, sens_attr.title()), (dset_tab, 'Policy', '')] = mod_row['Policy'].item()

                for ecir_status, scat_status in zip(['Before', 'After'], ['Base', 'Mit']):
                    df.loc[
                        (mod, sens_attr.title()), (dset_tab, "NDCG", scat_status)
                    ] = mod_row[('NDCG', ecir_status)]
                    df.loc[
                        (mod, sens_attr.title()), (dset_tab, "$\Delta$NDCG", scat_status)
                    ] = mod_row[('$\Delta$NDCG', ecir_status)]
    print(df)
    df = df.reset_index()

    i = 0
    fig, axs = plt.subplots(1, 4, sharey=True, figsize=(16, 4))
    handles, labels = None, None
    for dset in ["ML-1M", "LFM-1K"]:
        for sens_attr in ["Gender", "Age"]:
            plot_df = []
            sens_df = df.set_index("Sens Attr").loc[sens_attr]
            for col in ["NDCG", "$\Delta$NDCG"]:
                col_df = sens_df[(dset, col)].copy(deep=True).abs()
                plot_df.append((col_df["Mit"] - col_df["Base"]) / col_df["Base"])
            plot_df = pd.concat(plot_df, keys=['Rel. Diff. NDCG', 'Rel. Diff. $\Delta$NDCG'], axis=1)
            plot_df['Model'] = sens_df['Model']
            plot_df['Paper'] = ['Burke et al.'] + ['Li et al.'] * 3 + ['Ekstrand et al.'] * 3 + ['Ours'] * 3

            # style_order = dict(zip(sens_df['Model'].values, ['o', 'v', '^', '<', '+', 'x', '*', 's', 'p', 'D']))
            # style_order = dict(zip(sens_df['Model'].values, ['$\mathbf{' + str(x) + '}$' for x in range(10)]))
            style_order = dict(zip(sens_df['Model'].values, ['o', 's', 'd', 'X', 's', 'd', 'X', 's', 'd', 'X']))
            s = 250
            marker = '*'

            sns.scatterplot(
                x='Rel. Diff. NDCG', y=r'Rel. Diff. $\Delta$NDCG', data=plot_df,
                hue='Paper', s=s, ax=axs[i],
                marker=marker,
                # style='Model', markers=style_order
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

            axs[i].xaxis.set_major_formatter(lambda x, pos: f'{x*100:.1f}%')
            axs[i].yaxis.set_major_formatter(lambda x, pos: f'{x*100:.1f}%')

            axs[i].set_title(f"{dset}: {sens_attr}", pad=12)

            if i == 0:
                inset_pos = [0.7, 0.5, 0.2, 0.4]
                x1, x2, y1, y2 = -0.001, 0.001, -0.29, -0.14
            elif i == 1:
                inset_pos = [0.02, 0.5, 0.4, 0.4]
                x1, x2, y1, y2 = -0.03, 0.01, -0.26, 0.07
            elif i == 2:
                i += 1
                continue
                # inset_pos = [0.02, 0.5, 0.2, 0.4]
                # x1, x2, y1, y2 = -0.02, 0.02, -0.65, 0.05
            else:
                inset_pos = [0.02, 0.5, 0.4, 0.4]
                x1, x2, y1, y2 = -0.03, 0.02, 0.11, 0.45

            axins = axs[i].inset_axes(inset_pos)
            sns.scatterplot(
                x='Rel. Diff. NDCG', y=r'Rel. Diff. $\Delta$NDCG', data=plot_df,
                hue='Paper', s=s, ax=axins, legend=False,
                marker=marker,
                # style='Model', markers=style_order
            )

            twin_xaxins = axins.twinx()
            twin_xaxins.set_xticklabels([])
            twin_xaxins.set_yticklabels([])
            twin_xaxins.tick_params(axis='both', length=0)

            if i == 1 or i == 2:
                axins.spines[['left', 'bottom']].set_position('zero')
            else:
                axins.spines[['left']].set_position('zero')

            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.set_xticklabels([])
            axins.tick_params(axis='y', labelsize=8)
            axins.yaxis.set_major_locator(mpl_tick.MaxNLocator(4))
            axins.tick_params(axis='x', length=0)
            axins.set_xlabel('')
            axins.set_ylabel('')

            axins.yaxis.set_major_formatter(lambda x, pos: f'{x*100:.1f}%')

            axs[i].indicate_inset_zoom(axins, edgecolor="black")

            if i in [0, 3]:
                if i == 0:
                    inset_pos = [0.05, 0.7, 0.15, 0.15]
                    spine_pos = ('axes', 0.0)
                    x1, x2, y1, y2 = -0.046, -0.041, -0.24, -0.19
                    labelsize = 7
                elif i == 3:
                    inset_pos = [0.35, 0.1, 0.2, 0.2]
                    spine_pos = ('axes', 0.0)
                    x1, x2, y1, y2 = 0.035, 0.045, -0.58, -0.501
                    labelsize = 8

                axins_second = axs[i].inset_axes(inset_pos)
                sns.scatterplot(
                    x='Rel. Diff. NDCG', y=r'Rel. Diff. $\Delta$NDCG', data=plot_df,
                    hue='Paper', s=s, ax=axins_second, legend=False,
                    marker=marker,
                    # style='Model', markers=style_order
                )

                twin_xaxins = axins_second.twinx()
                twin_xaxins.set_xticklabels([])
                twin_xaxins.set_yticklabels([])
                twin_xaxins.tick_params(axis='both', length=0)

                axins_second.spines[['left']].set_position(spine_pos)

                axins_second.set_xlim(x1, x2)
                axins_second.set_ylim(y1, y2)
                axins_second.set_xticklabels([])
                axins_second.tick_params(axis='y', labelsize=labelsize)
                axins_second.yaxis.set_major_locator(mpl_tick.MaxNLocator(1))
                axins_second.tick_params(axis='x', length=0)
                axins_second.set_xlabel('')
                axins_second.set_ylabel('')

                axins_second.yaxis.set_major_formatter(lambda x, pos: f'{x*100:.1f}%')

                axs[i].indicate_inset_zoom(axins_second, edgecolor="black")

            i += 1

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
    for dset in ["ML-1M", "LFM-1K"]:
        for sens_attr in ["Gender", "Age"]:
            extent = axs[i].get_window_extent().transformed(fig.dpi_scale_trans.inverted())

            # bbox_args = [1.7, 1.1] if i == 0 else [1.1, 1.1]
            width = extent.width
            height = extent.height
            deltaw = (1.04 * width - width) / 2.0
            deltah = (0.93 * height - height) / 2.0
            offsetw = 0.6 if i == 3 else -0.2
            offseth = -1.4
            a = np.array([[-deltaw - deltaw * offsetw, -deltah], [deltaw, deltah + deltah * offseth]])
            new_bbox = extent._points + a
            fig.savefig(os.path.join('scripts', f'ax_{dset}_{sens_attr}_expanded.png'), bbox_inches=mpl_trans.Bbox(new_bbox), dpi=300)
            i += 1
