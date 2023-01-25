# %%
import os
import argparse
import inspect

import pandas as pd

import src.utils.utils as utils


# %%
def get_plots_path():
    plots_path = os.path.join(
        script_path,
        os.pardir,
        f'dp_plots'
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
out_columns = ['# Interactions', '# Users', '# Items', 'Avg. Users Degree', 'Avg. Items Degree', 'Sparsity', 'Sens Attr Distrib']
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
            sens_cols = user_feat.columns[~user_feat.columns.str.contains('_id')]
            sens_info = ""
            for col in sens_cols:
                if col in dataset.field2id_token:
                    user_feat[col] = user_feat[col].map(dataset.field2id_token[col].__getitem__)
                    col_info = (user_feat[[col]].value_counts() / len(user_feat) * 100).map(lambda x: f"{x:.1f}%")
                    col_info = col_info.reset_index().to_dict(orient="list")
                    sens_info += str(col_info) + "\n"

            out_data.append([
                dataset.inter_num,
                dataset.user_num - 1,
                dataset.item_num - 1,
                round(dataset.avg_actions_of_users, 1),
                round(dataset.avg_actions_of_items, 1),
                f"{dataset.sparsity * 100:.1f}%",
                sens_info
            ])

df = pd.DataFrame(out_data, columns=out_columns, index=check_datasets)
print(df)
df.to_csv(os.path.join(get_plots_path(), 'datasets_stats.csv'))
with pd.option_context('max_colwidth', None):
    df.index.name = "Dataset"
    df.reset_index().replace('%', '\%', regex=True).to_latex(
        os.path.join(get_plots_path(), 'datasets_stats.tex'), index=None, escape=False
    )
