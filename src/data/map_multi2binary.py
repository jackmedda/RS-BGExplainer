import os
import pprint
import argparse

import pandas as pd


def map_multi2binary(df, field, label):
    mask = df[field] == label
    df[mask] = "M"
    df[~mask] = "F"
    return df


if __name__ == "__main__":
    r"""
    python -m src.data.preprocess_dataset --train_split 0.7 --test_split 0.2 --validation_split 0.1 --split_type timestamp --user_field user_id:token --item_field course_id:token --time_field timestamp:float --in_filepath C:\Users\Giacomo\PycharmProjects\BDExplainer\dataset\coco_5_America\coco_5_America.inter --user_filepath C:\Users\Giacomo\PycharmProjects\BDExplainer\dataset\coco_5_America\coco_5_America.user --out_folderpath C:\Users\Giacomo\PycharmProjects\BDExplainer\dataset\coco_7_America\ --dataset_name coco_7_America --min_interactions 7
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_filepath', required=True)
    parser.add_argument('--out_folderpath', required=True)
    parser.add_argument('--dataset_name', required=True)
    parser.add_argument('--map_attribues', nargs='+', required=True)
    
    args = parser.parse_args()

    user_df = pd.read_csv(args.user_filepath, sep='\t')

    pprint.pprint(vars(args))

    os.makedirs(args.out_folderpath, exist_ok=True)

    for attr in args.map_atttibutes:
        for label in user_df[attr].unique():
            df = user_df.copy(deep=True)
            df = map_multi2binary(df, attr, label)
            df.to_csv(os.path.join(args.out_folderpath, f"{args.dataset_name}_{label}.user"), index=None, sep='\t')
