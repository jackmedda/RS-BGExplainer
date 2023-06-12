# %%
import os
import argparse
import inspect

import pandas as pd

import biga.utils as utils


# %%
def get_plots_path():
    plots_path = os.path.join(
        script_path,
        f'new_dp_plots'
    )

    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    return plots_path


# %%
parser = argparse.ArgumentParser()
parser.add_argument('--saved_path', default='saved')

args = parser.parse_args()

script_path = os.path.abspath(os.path.dirname(inspect.getsourcefile(lambda: 0)))
if os.sep not in args.saved_path:
    args.saved_path = os.path.join(script_path, os.pardir, args.saved_path)

config_path = os.path.join(args.saved_path, os.pardir, 'config')

print(args)

check_datasets = []
out_data = []
out_columns = [
    '# Users',
    '# Items',
    '# Interactions',
    'Gender Repr.',
    'Age Repr.',
    'Min. Degree per user',
]

stats_path = os.path.join(get_plots_path(), 'datasets_stats.csv')
if not os.path.exists(stats_path):
    for model_path in os.listdir(args.saved_path):
        if 'perturbed' not in model_path and os.path.isfile(os.path.join(args.saved_path, model_path)):
            dset_name = model_path.replace('GCMC-', '').replace('LightGCN-', '').replace('NGCF-', '')  # removes model name
            dset_name = dset_name[::-1].split('_', maxsplit=1)[1].split('-', maxsplit=3)[-1][::-1]  # removes date
            if dset_name.lower() not in check_datasets:
                config_files = os.listdir(config_path)
                try:
                    conf_idx = config_files.index(f"{dset_name.lower()}_explainer.yaml")
                    conf_file = os.path.join(config_path, config_files[conf_idx])
                except ValueError:
                    conf_file = os.path.join(config_path, 'explainer.yaml')

                check_datasets.append(dset_name.lower())
                config, model, dataset, train_data, valid_data, test_data = utils.load_data_and_model(
                    os.path.join(args.saved_path, model_path),
                    conf_file
                )

                user_feat = pd.DataFrame(dataset.user_feat.numpy()).iloc[1:]
                sens_cols = user_feat.columns[user_feat.columns.isin(['gender', 'age'])]
                sens_info = dict.fromkeys(['Gender', 'Age'])
                for col in sens_cols:
                    if col in dataset.field2id_token and col in ['age', 'gender']:
                        user_feat[col] = user_feat[col].map(dataset.field2id_token[col].__getitem__)
                        if col == 'age':
                            user_feat[col] = user_feat[col].map({'M': 'Y', 'F': 'O'})
                        col_info = (user_feat[[col]].value_counts() / len(user_feat) * 100).map(lambda x: f"{x:.1f}%")
                        sens_info[col.title()] = '; '.join(
                            col_info.to_frame().reset_index().apply(
                                lambda x: f"{x[col]} : {x[0]}".replace('%', '\%'), axis=1
                            ).values
                        )
                    else:
                        sens_info[col] = 'NA'

                out_data.append([
                    dataset.user_num - 1,
                    dataset.item_num - 1,
                    dataset.inter_num,
                    sens_info['Gender'],
                    sens_info['Age'],
                    dataset.history_item_matrix()[2][1:].min().item()
                ])

    df = pd.DataFrame(out_data, columns=out_columns, index=check_datasets)
    df.index.name = "Dataset"
    df = df.T
    print(df)
    df.to_csv(stats_path)
else:
    df = pd.read_csv(stats_path, index_col=0)


def dict_mapper(x):
    if isinstance(x, str):
        if 'Gender' in x or 'Age' in x:
            x = x.replace('Age ', '')
            return x.split(' Gender ') if 'Gender' in x else [x, '']
    return x

df = df[["ml-1m", "tafeng", "lastfm-1k", "insurance"]]
df.columns = ["ML-1M", "FENG", "LFM-1K", "INS"]
df.loc["Age Repr."] = df.loc["Age Repr."].str.replace('M :', 'Y :').str.replace('F :', 'O :')
df.loc["Domain"] = ["Movie", "Grocery", "Music", "Insurance"]
df.rename(index={
    "bar{Degree}": "$\overline{\text{User DEG}}$",
    "bar{Density}": "$\overline{\text{User DY}}$",
    "bar{Reachability}": "$\overline{\text{User IDG}}$",
    "Gini Degree": "Gini User DEG",
    "Gini Density": "Gini User DY",
    "Gini Reachability": "Gini User IDG",
    "Min. Degree per user": "Min. User DEG"
}, inplace=True)

df = df.applymap(dict_mapper).explode(df.columns.tolist())
df['Group'] = df["ML-1M"].map(
    lambda x: None if not isinstance(x, str) else ("G" if "F :" in x else ("A" if "O :" in x else None))
)
df = df.reset_index().set_index(['index', 'Group'])

with pd.option_context('max_colwidth', None):
    tex_table = df.to_latex(column_format="rr|r|r|r|r", escape=False, multirow=True)

with open(os.path.join(get_plots_path(), 'datasets_stats.tex'), 'w') as tex_f:
    tex_f.write(
        tex_table.replace(
            '# Users & NaN ', r'\multicolumn{2}{r|}{\# Users}'
        ).replace(
            '# Items & NaN ', r'\multicolumn{2}{r|}{\# Items}'
        ).replace(
            '# Interactions & NaN ', r'\multicolumn{2}{r|}{\# Interactions}'
        ).replace(
            'Min. User DEG & NaN ', r'\multicolumn{2}{r|}{Min. User DEG}'
        ).replace(
            'Domain & NaN ', r'\multicolumn{2}{r|}{Domain}'
        ).replace(
            'ML-1M', r'ML-1M \cite{DBLP:journals/tiis/HarperK16}'
        ).replace(
            'FENG', r'FENG\footnote{\url{https://www.kaggle.com/datasets/chiranjivdas09/ta-feng-grocery-dataset}}'
        ).replace(
            'LFM-1K', r'LFM-1K \cite{DBLP:books/daglib/0025137}'
        ).replace(
            'INS', r'INS\footnote{\url{https://www.kaggle.com/datasets/mrmorj/insurance-recommendation}}'
        ).replace(
            r'\cline{1-6}', r'\cline{2-6}'
        ).replace(
            'Gender Repr.', r'\multirow{2}{*}{Repr.}'
        ).replace(
            'Age Repr.', ''
        ).replace('NaN', 'NA')
    )
