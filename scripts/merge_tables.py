import os
import re
import pickle
import argparse

import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '--d', required=True)
    parser.add_argument('--model', '--m', nargs='+', default=None)
    parser.add_argument('--sensitive_attribute', '--sa', required=True)
    parser.add_argument('--cid99', help='Considers explanations with cid 99-100-101', action='store_true')
    parser.add_argument('--exclude', '--ex', nargs='+', help='Exclude certaing config ids', default=None)

    args = parser.parse_args()
    args.model = args.model or ["GCMC", "LightGCN", "NGCF"]
    args.dataset = args.dataset.lower()
    args.sensitive_attribute = args.sensitive_attribute.lower()
    args.exclude = args.exclude or []
    if not args.cid99:
        args.exclude.extend(['99', '100', '101'])
    print(args)

    out_path = os.path.join(os.getcwd(), 'scripts', 'merged_plots', args.dataset, args.sensitive_attribute)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    def pval_symbol(pval):
        if pval < 0.01:
            return '^'
        elif pval < 0.05:
            return '*'

        return ''

    def hl(val):
        return "\hl{" + val + "}"

    delta_col = "$\Delta$NDCG"
    final_df = []
    orig_pert_pval_dicts = {}
    for mod in args.model:
        plots_path = os.path.join(
            os.getcwd(), 'scripts', 'plots', args.dataset, mod, args.sensitive_attribute
        )
        for fold in os.scandir(plots_path):
            if args.exclude is not None and any(c in fold.name for c in args.exclude):
                continue

            df = pd.read_csv(os.path.join(fold, 'DP_barplot.md'), sep='|', skiprows=[1], usecols=list(range(1, 8)))
            df.columns = df.columns.str.strip()
            df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            df['Model'] = mod

            final_df.append(df)

            with open(os.path.join(fold, 'orig_pert_pval_dict.pkl'), 'rb') as f:
                orig_pert_pval_dicts[fold.name] = pickle.load(f)

    final_df = pd.concat(final_df, ignore_index=True).drop_duplicates(subset=['Split', 'Policy', 'Model'])

    pol_order = ['Orig'] + sorted(filter(lambda x: x != 'Orig', final_df.Policy.unique().tolist()))

    metric_order = ['NDCG', delta_col] + (['M', 'F'] if args.sensitive_attribute == 'gender' else ['Y', 'O'])
    models_order = ['GCMC', 'LightGCN', 'NGCF']

    melt_df = final_df.melt(['Model', 'Split', 'Policy', 'pvalue'], var_name='Metric', value_name='Value')

    valid_df = melt_df[melt_df.Split == 'Valid']
    test_df = melt_df[melt_df.Split == 'Test']

    for split_df, split in zip([valid_df, test_df], ['Valid', 'Test']):
        pval_dict = {}
        for _, row in split_df[split_df['Metric'] == delta_col].iterrows():
            pval_dict[(row['Model'], row['Policy'])] = row['pvalue']

        pivot_df = split_df[split_df.columns[split_df.columns != 'Split']].pivot(
            index=['Model', 'Policy'], columns='Metric', values='Value'
        )
        pivot_df = pivot_df.reindex(models_order, level=0).reindex(pol_order, level=1).reindex(metric_order, axis=1)
        pivot_df = pivot_df.applymap('{:.4f}'.format)
        for key, pval in pval_dict.items():
            pivot_df.loc[key, delta_col] = pivot_df.loc[key, delta_col] + pval_symbol(pval)
        pivot_df.to_markdown(os.path.join(out_path, f'merge_plot_{split}_{"_".join(args.model)}.md'))

        for m in pivot_df.index.get_level_values(0).unique():
            sorted_pols = pivot_df.loc[m, delta_col].str.replace('[*^]', '', regex=True).astype(float).sort_values()
            fairest_pols = sorted_pols[sorted_pols == sorted_pols.iloc[0]].index
            for f_pol in fairest_pols:
                pivot_df.loc[(m, f_pol), delta_col] = "\textbf{" + pivot_df.loc[(m, f_pol), delta_col] + "}"

        with open(os.path.join(out_path, f'merge_plot_{split}_{"_".join(args.model)}.tex'), 'w') as f:
            f.write(pivot_df.to_latex(
                column_format='llrrr',
                multirow=True,
                escape=False
            ))

    pivot_df = melt_df[melt_df['Metric'] == delta_col].pivot(
        index=['Model', 'Policy'], columns=['Metric', 'Split'], values='Value'
    )
    pivot_df = pivot_df.reindex(
        models_order, level=0
    ).reindex(
        pol_order, level=1
    ).reindex(
        metric_order, axis=1, level=0
    ).reindex(
        ['Valid', 'Test'], axis=1, level=1
    )
    pivot_df = pivot_df.applymap('{:.4f}'.format)
    pivot_df.to_markdown(os.path.join(out_path, f'delta_{"_".join(args.model)}.md'))

    rq3_df = pivot_df.astype(float).copy(deep=True)

    for m in pivot_df.index.get_level_values(0).unique():
        m_df = pivot_df.loc[m]

        highlight_pols = m_df.loc[
            (m_df[(delta_col, 'Valid')] < m_df.loc['Orig', (delta_col, 'Valid')]) &
            (m_df[(delta_col, 'Test')] < m_df.loc['Orig', (delta_col, 'Test')])
        ].index
        for spl in ['Valid', 'Test']:
            sorted_pols = pivot_df.loc[m, (delta_col, spl)].astype(float).sort_values()
            fairest_pols = sorted_pols[sorted_pols == sorted_pols.iloc[0]].index

            for f_pol in fairest_pols:
                pivot_df.loc[
                    (m, f_pol), (delta_col, spl)
                ] = "\textbf{" + pivot_df.loc[(m, f_pol), (delta_col, spl)] + "}"

            for hl_pol in highlight_pols:
                pivot_df.loc[
                    (m, hl_pol), (delta_col, spl)
                ] = hl(pivot_df.loc[(m, hl_pol), (delta_col, spl)] )

    with open(os.path.join(out_path, f'delta_{"_".join(args.model)}.tex'), 'w') as f:
        f.write(pivot_df.to_latex(
            column_format='llrr',
            multicolumn_format='c',
            multirow=True,
            escape=False
        ))

    rq3_table = []
    for m in rq3_df.index.get_level_values(0).unique():
        rq3_mdf = rq3_df.loc[m]
        for pol in rq3_mdf.index.get_level_values(0).unique():
            if pol == 'Orig':
                continue
            perc_change = ((rq3_mdf.loc[pol, (delta_col, 'Valid')] / rq3_mdf.loc['Orig', (delta_col, 'Valid')]) - 1)
            _pol = pol.replace('CN+', '') if pol != 'CN' else 'B'
            rq3_table.append([m, _pol, f"{perc_change * 100:.1f}\%"])

    rq3_df = pd.DataFrame(rq3_table, columns=["Model", "Policy", "Change"])
    rq3_df["SA"] = args.sensitive_attribute
    rq3_df["Dataset"] = args.dataset
    rq3_df = rq3_df.pivot(index="Policy", columns=["Dataset", "Model", "SA"], values="Change")
    rq3_df.to_csv(os.path.join(out_path, f'rq3_perc_change_{"_".join(args.model)}.csv'))
    with open(os.path.join(out_path, f'rq3_perc_change_{"_".join(args.model)}.tex'), 'w') as f:
        f.write(rq3_df.to_latex(
            column_format='lrrr',
            multicolumn_format='c',
            multirow=True,
            escape=False
        ))

    def delta2sign(row, sa):
        dg = ['M', 'F'] if sa == 'gender' else ['Y', 'O']
        return row[delta_col] * (-1 if row[dg[0]] < row[dg[1]] else 1)

    rq1_table = []
    ecir_plot_data = []
    ecir_df = melt_df.copy()
    ecir_df = ecir_df.pivot(
        index=['Model', 'Split', 'Policy', 'pvalue'], columns='Metric', values='Value'
    ).reset_index()
    ecir_df[delta_col] = ecir_df.apply(lambda row: delta2sign(row, args.sensitive_attribute), axis=1)
    for mod, mod_df in ecir_df.groupby('Model'):
        best_pol_df = mod_df[(mod_df.Split == 'Valid') & (mod_df.Policy != 'Orig')]
        best_pol = best_pol_df.set_index('Policy')[delta_col].abs().sort_values().index[0]
        mod_spl_df = mod_df.set_index(['Split', 'Policy'])

        before_delta = pval_symbol(mod_spl_df.loc[('Test', 'Orig'), 'pvalue']) + \
                       f"{mod_spl_df.loc[('Test', 'Orig'), delta_col]:.3f}"
        after_delta = pval_symbol(mod_spl_df.loc[('Test', best_pol), 'pvalue']) + \
                      f"{mod_spl_df.loc[('Test', best_pol), delta_col]:.3f}"

        ecir_plot_data.append([mod, 'Before', f"{mod_spl_df.loc[('Test', 'Orig'), 'NDCG']:.3f}", 'NDCG'])
        ecir_plot_data.append([mod, 'Before', before_delta, delta_col])
        ecir_plot_data.append([mod, 'After', f"{mod_spl_df.loc[('Test', best_pol), 'NDCG']:.3f}", 'NDCG'])
        ecir_plot_data.append([mod, 'After', after_delta, delta_col])

        rq1_sorted_pols = best_pol_df.set_index('Policy')[delta_col].abs().sort_values()
        rq1_fairest_pols = rq1_sorted_pols[rq1_sorted_pols == rq1_sorted_pols.iloc[0]].index
        rq1_fpols = list(rq1_fairest_pols[rq1_fairest_pols != 'Orig'])
        rq1_mdf = mod_df.pivot(index=['Model', 'Policy'], columns=['Split'], values=delta_col)

        hl_pols_mdf = rq1_mdf.abs()
        highlight_pols = hl_pols_mdf.loc[
            (hl_pols_mdf['Valid'] < hl_pols_mdf.loc[(mod, 'Orig'), 'Valid']) &
            (hl_pols_mdf['Test'] < hl_pols_mdf.loc[(mod, 'Orig'), 'Test'])
        ].index
        # for idx in highlight_pols:
        #     rq1_mdf.loc[idx, 'Valid'] = hl(f"{rq1_mdf.loc[idx, 'Valid']:.4f}")
        #     rq1_mdf.loc[idx, 'Test'] = hl(f"{rq1_mdf.loc[idx, 'Test']:.4f}")

        def upd_rq1_vals(row_idx, col_idx):
            value = pval_symbol(mod_spl_df.loc[(col_idx.name, row_idx.name[1]), 'pvalue']) + f"{abs(row_idx.item()):.4f}"
            return hl(value) if row_idx.name in highlight_pols else value

        rq1_table.append(
            rq1_mdf.reindex(['Valid', 'Test'], axis=1).loc[[(mod, p) for p in ['Orig'] + rq1_fpols]].apply(
                lambda col_s: col_s.to_frame().apply(
                    lambda row_s: upd_rq1_vals(row_s, col_s),
                    axis=1
                ), axis=0
            )
        )

    with open(os.path.join(out_path, f'delta_{"_".join(args.model)}.tex'), 'w') as f:
        f.write(pivot_df.to_latex(
            column_format='llrr',
            multicolumn_format='c',
            multirow=True,
            escape=False
        ))

    rq1_df = pd.concat(rq1_table)
    rq1_df_idx = rq1_df.index.to_frame()
    rq1_df_idx.insert(1, "SA", "G" if args.sensitive_attribute.lower() == "gender" else "A")
    rq1_df.index = pd.MultiIndex.from_frame(rq1_df_idx)
    rq1_df.index = rq1_df.index.map(
        lambda x: (
            f"\rotatebox[origin=c]{{90}}{{{x[0]}}}",
            f"\rotatebox[origin=c]{{90}}{{{x[1]}}}",
            x[2].replace('CN+', '') if x[2] != 'CN' else 'B'
        )
    )

    with open(os.path.join(out_path, f'rq1_delta_{"_".join(args.model)}.tex'), 'w') as f:
        f.write(rq1_df.to_latex(
            column_format='lllrr',
            multicolumn_format='c',
            multirow=True,
            escape=False
        ).replace(' *', ' {\scriptsize *}').replace('^', '{\scriptsize \^{}}'))

    ecir_plot_df = pd.DataFrame(ecir_plot_data, columns=['Model', 'Status', 'Value', 'Metric'])
    ecir_plot_df = ecir_plot_df.pivot(index='Model', columns=['Metric', 'Status'], values='Value')

    ecir_plot_df = ecir_plot_df.reindex(
        models_order
    ).reindex(
        ['NDCG', delta_col], axis=1, level=0
    ).reindex(
        ['Before', 'After'], axis=1, level=1
    )
    print(ecir_plot_df)

    with open(os.path.join(out_path, f'ecir_table_{"_".join(args.model)}.tex'), 'w') as f:
        f.write(ecir_plot_df.to_latex(
            column_format='lrrrr',
            multicolumn_format='c',
            escape=False
        ).replace('*', '{\scriptsize *}').replace('^', '{\scriptsize \^{}}'))

    merged_plots_path = os.path.join(os.getcwd(), 'scripts', 'merged_plots')
    rq3_csvs = []
    for d, subd, files in os.walk(merged_plots_path):
        if 'rq3_perc_change_GCMC_LightGCN_NGCF.csv' in files:
            rq3_csvs.append(pd.read_csv(os.path.join(d, 'rq3_perc_change_GCMC_LightGCN_NGCF.csv'), index_col=[0], header=[0,1,2]))

    if len(rq3_csvs) == 4:
        dsets_order = ['ml-1m', 'lastfm-1k']
        sa_order = ['gender', 'age']
        pol_order = ['B', 'ZN', 'LD', 'S', 'F', 'IP', 'ZN+IP', 'LD+IP', 'S+IP', 'F+IP']
        rq3_final_df = pd.concat(rq3_csvs, axis=1)
        rq3_final_df = rq3_final_df.reindex(
            dsets_order, axis=1, level=0
        ).reindex(
            models_order, axis=1, level=1
        ).reindex(
            sa_order, axis=1, level=2
        ).reindex(
            pol_order, axis=0
        )

        rq3_final_df = rq3_final_df.applymap(lambda x: hl(x) if '-' in x else x)

        with open(os.path.join(merged_plots_path, f'rq3_final_perc_change_{"_".join(models_order)}.tex'), 'w') as f:
            f.write(rq3_final_df.to_latex(
                column_format='lrrrrrrrrrrrr',
                multicolumn_format='c',
                escape=False
            ))
