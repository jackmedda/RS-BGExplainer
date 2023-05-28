import os
import pickle
import argparse

import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '--d', required=True)
    parser.add_argument('--model', '--m', nargs='+', required=True)
    parser.add_argument('--sensitive_attribute', '--sa', required=True)
    parser.add_argument('--cid99', help='Considers explanations with cid 99-100-101', action='store_true')

    args = parser.parse_args()
    args.dataset = args.dataset.lower()
    args.sensitive_attribute = args.sensitive_attribute.lower()
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

    delta_col = "$\Delta$NDCG"
    final_df = []
    orig_pert_pval_dicts = {}
    for mod in args.model:
        plots_path = os.path.join(
            os.getcwd(), 'scripts', 'plots', args.dataset, mod, args.sensitive_attribute
        )
        for fold in os.scandir(plots_path):
            if not args.cid99 and any(c in fold.name for c in ['99', '100', '101']):
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
    pol_map = {x: x.replace('_GNNUERS', '') for x in pol_order}
    pol_order = list(pol_map.values())

    metric_order = ['NDCG', delta_col] + (['M', 'F'] if args.sensitive_attribute == 'gender' else ['Y', 'O'])
    models_order = ['GCMC', 'LightGCN', 'NGCF']

    final_df['Policy'] = final_df['Policy'].map(pol_map)

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
                ] = "\hl{" + pivot_df.loc[(m, hl_pol), (delta_col, spl)] + "}"

    with open(os.path.join(out_path, f'delta_{"_".join(args.model)}.tex'), 'w') as f:
        f.write(pivot_df.to_latex(
            column_format='llrr',
            multicolumn_format='c',
            multirow=True,
            escape=False
        ))

    def delta2sign(row, sa):
        dg = ['M', 'F'] if sa == 'gender' else ['Y', 'O']
        return row[delta_col] * (-1 if row[dg[0]] < row[dg[1]] else 1)

    ecir_plot_data = []
    ecir_df = melt_df.copy()
    ecir_df = ecir_df.pivot(
        index=['Model', 'Split', 'Policy', 'pvalue'], columns='Metric', values='Value'
    ).reset_index()
    ecir_df[delta_col] = ecir_df.apply(lambda row: delta2sign(row, args.sensitive_attribute), axis=1)
    for mod, mod_df in ecir_df.groupby('Model'):
        best_pol = mod_df[mod_df.Split == 'Valid'].sort_values(delta_col).iloc[0]['Policy']
        mod_spl_df = mod_df.set_index(['Split', 'Policy'])

        before_delta = pval_symbol(mod_spl_df.loc[('Test', 'Orig'), 'pvalue']) + \
                       f"{mod_spl_df.loc[('Test', 'Orig'), delta_col]:.3f}"
        after_delta = pval_symbol(mod_spl_df.loc[('Test', best_pol), 'pvalue']) + \
                      f"{mod_spl_df.loc[('Test', best_pol), delta_col]:.3f}"

        ecir_plot_data.append([mod, 'Before', f"{mod_spl_df.loc[('Test', 'Orig'), 'NDCG']:.3f}", 'NDCG'])
        ecir_plot_data.append([mod, 'Before', before_delta, delta_col])
        ecir_plot_data.append([mod, 'After', f"{mod_spl_df.loc[('Test', best_pol), 'NDCG']:.3f}", 'NDCG'])
        ecir_plot_data.append([mod, 'After', after_delta, delta_col])

    ecir_plot_df = pd.DataFrame(ecir_plot_data, columns=['Model', 'Status', 'Value', 'Metric'])
    ecir_plot_df = ecir_plot_df.pivot(index='Model', columns=['Metric', 'Status'], values='Value')
    ecir_plot_df = ecir_plot_df.reindex(
        models_order, level=0
    ).reindex(
        ['NDCG', delta_col], axis=1, level=0
    ).reindex(
        ['Before', 'After'], axis=1, level=1
    )
    print(ecir_plot_df)
