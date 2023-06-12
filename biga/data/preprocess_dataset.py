import os
import pickle
import pprint
import argparse

import tqdm
import pandas as pd

import biga.data.utils as data_utils


if __name__ == "__main__":
    """
    # Windows
    python -m biga.data.preprocess_dataset --train_split 0.7 --test_split 0.2 --validation_split 0.1 --split_type per_user --user_field user_id:token --item_field item_id:token --time_field timestamp:float --in_filepath dataset\lastfm-1k\lastfm-1k.inter --user_filepath dataset\lastfm-1k\lastfm-1k.user --dataset_name lastfm-1k_n-1 --min_interactions 20 --out_folderpath lastfm-1k_n-1\
    # Linux
    python3.9 -m biga.data.preprocess_dataset --train_split 0.7 --test_split 0.2 --validation_split 0.1 --split_type per_user --user_field user_id:token --item_field item_id:token --time_field timestamp:float --in_filepath dataset/lastfm-1k/lastfm-1k.inter --user_filepath dataset/lastfm-1k/lastfm-1k.user --min_interactions 20 --dataset_name lastfm-1k_n-1 --out_folderpath lastfm-1k_n-1/

    # Train to Test (Linux)
    python3.9 -m biga.data.preprocess_dataset --train_split 0.7 --test_split 0.2 --validation_split 0.1 --split_type per_user --user_field user_id:token --item_field item_id:token --time_field timestamp:float --in_filepath dataset/lastfm-1k/lastfm-1k.inter --user_filepath dataset/lastfm-1k/lastfm-1k.user --min_interactions 20 --dataset_name lastfm-1k_n-1 --out_folderpath lastfm-1k_n-1/ --train_to_test 1
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_filepath', required=True)
    parser.add_argument('--user_filepath', required=True)
    parser.add_argument('--out_folderpath', required=True)
    parser.add_argument('--dataset_name', required=True)
    parser.add_argument('--split_type', choices=['random', 'per_user'], default='per_user')
    parser.add_argument('--train_split', default=0.7, type=float)
    parser.add_argument('--train_to_test', default=0, type=int,
                        help="it moves the last N train interactions to test")
    parser.add_argument('--test_split', default=0.2, type=float)
    parser.add_argument('--validation_split', default=0.1, type=float)
    parser.add_argument('--user_field', default='user_id')
    parser.add_argument('--item_field', default='item_id')
    parser.add_argument('--time_field', default='timestamp')
    parser.add_argument('--random_state', type=int, default=120)
    parser.add_argument('--min_interactions', type=int, default=0)
    parser.add_argument('--add_token', action='store_true', help='add `token` or `float` to header')

    args = parser.parse_args()
    if args.train_to_test > 0:
        args.train_split = (args.train_split, args.train_to_test)

    df = pd.read_csv(args.in_filepath, sep='\t')
    user_df = pd.read_csv(args.user_filepath, sep='\t')

    pprint.pprint(vars(args))

    os.makedirs(args.out_folderpath, exist_ok=True)

    if args.min_interactions > 0:
        df = data_utils.filter_min_interactions(df, by=args.user_field, min_interactions=args.min_interactions)
        user_df = user_df[user_df[args.user_field].isin(df[args.user_field])]
        print(df.describe())
        print(df.apply(pd.unique, axis=0).apply(len))

    if args.add_token:
        if not 'token' in args.user_field:
            data_utils.add_token(df, user_df, args)
        else:
            print("`token` already present in headers")

    df.to_csv(os.path.join(args.out_folderpath, f"{args.dataset_name}.inter"), index=None, sep='\t')
    user_df.to_csv(os.path.join(args.out_folderpath, f"{args.dataset_name}.user"), index=None, sep='\t')

    if args.split_type == "per_user":
        print("> Splitting per user")
        train, val, test = data_utils.split_data_per_user(df,
                                                          train_split=args.train_split,
                                                          test_split=args.test_split,
                                                          validation_split=args.validation_split,
                                                          user_field=args.user_field,
                                                          time_field=args.time_field)
    elif args.split_type == "random":
        print("> Splitting randomly")
        train, val, test = data_utils.random_split(df,
                                                   train_split=args.train_split,
                                                   test_split=args.test_split,
                                                   validation_split=args.validation_split,
                                                   random_state=args.random_state)
    else:
        raise NotImplementedError(f"`split_type` = `{args.split_type}` not supported")

    remove_token = 'token' in args.user_field

    for data, data_name in zip([train, val, test], ["train", "validation", "test"]):
        with open(os.path.join(args.out_folderpath, f'{args.dataset_name}.{data_name}'), 'wb') as f:
            cols = [args.user_field, args.item_field]
            out_data = dict(zip(map(lambda x: x.split(':')[0] if remove_token else x, cols), data[cols].values.T))
            pickle.dump(out_data, f)
