import os
import pprint
import argparse

import pandas as pd

from biga.utils import copytree


def map_multi2binary(df, field, label):
    mask = df[field] == label
    df.loc[mask, field] = "M"
    df.loc[~mask, field] = "F"
    return df


if __name__ == "__main__":
    r"""
    python -m src.data.map_multi2binary --user_filepath dataset.user --map_attributes age
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '--d', required=True)
    parser.add_argument('--map_attributes', '--ma', nargs='+', required=True)
    parser.add_argument('--user_filepath', '--uf', default='')

    args = parser.parse_args()

    datasets_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'dataset'
    )

    uf = args.user_filepath
    uf = os.path.join(datasets_path, args.dataset, f"{args.dataset}.user") if not uf else uf
    user_df = pd.read_csv(uf, sep='\t')

    pprint.pprint(vars(args))

    recbole_extended_data_types = ['.inter', '.item', '.test', '.train', '.user', '.validation']

    for attr in args.map_attributes:
        unique_labels = user_df[attr].unique()
        if len(unique_labels) == 2:
            raise ValueError(f"The attribute `{attr}` is already binary")

        for label in unique_labels:
            df = user_df.copy(deep=True)
            df = map_multi2binary(df, attr, label)

            dset_new_name = f"{args.dataset}_{attr.split(':')[0] if ':' in attr else attr}_{label}"
            copy_dataset(datasets_path, args.dataset, dset_new_name)

            df.to_csv(os.path.join(dst, f"{args.dataset}.user"), index=None, sep='\t')

            print('-' * 50)
            print(f"Attr: {attr}    Label: {label}")
            print(df[attr].value_counts().map(lambda x: f"{x / len(df) * 100:.2f}%"))
