import os
import pickle
import pprint
import argparse

import tqdm
import pandas as pd


def split_data_by_timestamp(interactions,
                            train_split=0.80,
                            test_split=None,
                            validation_split=None,
                            user_field='user_id',
                            time_field='timestamp'):
    train_set = []
    test_set = []
    val_set = []
    groups = interactions.groupby([user_field])
    for i, (_, group) in enumerate(tqdm.tqdm(groups, desc=f"Splitting data per_user")):
        if time_field:
            sorted_group = group.sort_values(time_field)
        else:
            sorted_group = group

        if isinstance(train_split, float) or isinstance(test_split, float):
            n_rating_train = int(len(sorted_group.index) * train_split) if train_split is not None else 0
            n_rating_test = int(len(sorted_group.index) * test_split) if test_split is not None else 0
            n_rating_val = int(len(sorted_group.index) * validation_split) if validation_split is not None else 0

            if len(sorted_group.index) > (n_rating_train + n_rating_test + n_rating_val):
                n_rating_train += len(sorted_group.index) - (n_rating_train + n_rating_test + n_rating_val)
        else:
            raise ValueError(f"split type not accepted")

        if n_rating_train == 0:
            start_index = len(sorted_group) - n_rating_test
            start_index = start_index - n_rating_val if n_rating_val is not None else start_index
            train_set.append(sorted_group.iloc[:start_index])
        else:
            train_set.append(sorted_group.iloc[:n_rating_train])
            start_index = n_rating_train

        if n_rating_val > 0:
            val_set.append(sorted_group.iloc[start_index:(start_index + n_rating_val)])
            start_index += n_rating_val

        if n_rating_test > 0:
            test_set.append(sorted_group.iloc[start_index:(start_index + n_rating_test)])
        else:
            test_set.append(sorted_group.iloc[start_index:])

    train, test = pd.concat(train_set), pd.concat(test_set)
    validation = pd.concat(val_set) if val_set else None

    return train, validation, test


def random_split(interactions: pd.DataFrame,
                 train_split=0.80,
                 test_split=None,
                 validation_split=None,
                 random_state=None):
    train_set = []
    test_set = []
    val_set = []

    interactions = interactions.sample(frac=1, random_state=random_state)

    if isinstance(train_split, float) or isinstance(test_split, float):
        n_rating_train = int(len(interactions) * train_split) if train_split is not None else 0
        n_rating_test = int(len(interactions) * test_split) if test_split is not None else 0
        n_rating_val = int(len(interactions) * validation_split) if validation_split is not None else 0

        if len(interactions) > (n_rating_train + n_rating_test + n_rating_val):
            n_rating_train += len(interactions) - (n_rating_train + n_rating_test + n_rating_val)
    else:
        raise ValueError(f"split type not accepted")

    if n_rating_train == 0:
        start_index = len(interactions) - n_rating_test
        start_index = start_index - n_rating_val if n_rating_val is not None else start_index
        train_set.append(interactions.iloc[:start_index])
    else:
        train_set.append(interactions.iloc[:n_rating_train])
        start_index = n_rating_train

    if n_rating_val > 0:
        val_set.append(interactions.iloc[start_index:(start_index + n_rating_val)])
        start_index += n_rating_val

    if n_rating_test > 0:
        test_set.append(interactions.iloc[start_index:(start_index + n_rating_test)])
    else:
        test_set.append(interactions.iloc[start_index:])

    train, test = pd.concat(train_set), pd.concat(test_set)
    validation = pd.concat(val_set) if val_set else None

    return train, validation, test


def filter_min_interactions(interactions, by='user_id', min_interactions=20):
    n_inters = interactions.groupby(by).apply(len)
    mask = n_inters >= min_interactions

    return interactions[interactions[by].isin(n_inters[mask].index)]


if __name__ == "__main__":
    r"""
    python -m src.data.preprocess_dataset --train_split 0.7 --test_split 0.2 --validation_split 0.1 --split_type timestamp --user_field user_id:token --item_field item_id:token --time_field timestamp:float --in_filepath dataset\lastfm_1k\lastfm_1k.inter --user_filepath dataset\lastfm_1k\lastfm_1k.user --out_folderpath lastfm_1k\ --dataset_name lastfm_1k --min_interactions 20
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_filepath', required=True)
    parser.add_argument('--user_filepath', required=True)
    parser.add_argument('--out_folderpath', required=True)
    parser.add_argument('--dataset_name', required=True)
    parser.add_argument('--split_type', choices=['random', 'timestamp'], default='timestamp')
    parser.add_argument('--train_split', default=0.7, type=float)
    parser.add_argument('--test_split', default=0.2, type=float)
    parser.add_argument('--validation_split', default=0.1, type=float)
    parser.add_argument('--user_field', default='user_id')
    parser.add_argument('--item_field', default='item_id')
    parser.add_argument('--time_field', default='timestamp')
    parser.add_argument('--random_state', type=int, default=120)
    parser.add_argument('--min_interactions', type=int, default=0)
    
    args = parser.parse_args()
    
    df = pd.read_csv(args.in_filepath, sep='\t')
    user_df = pd.read_csv(args.user_filepath, sep='\t')

    pprint.pprint(vars(args))

    os.makedirs(args.out_folderpath, exist_ok=True)

    if args.min_interactions > 0:
        df = filter_min_interactions(df, by=args.user_field, min_interactions=args.min_interactions)
        user_df = user_df[user_df[args.user_field].isin(df[args.user_field])]
        print(df.describe())
        print(df.apply(pd.unique, axis=0).apply(len))
        df.to_csv(os.path.join(args.out_folderpath, f"{args.dataset_name}.inter"), index=None, sep='\t')
        user_df.to_csv(os.path.join(args.out_folderpath, f"{args.dataset_name}.user"), index=None, sep='\t')

    if args.split_type == "timestamp":
        print("> Splitting by timestamp")
        train, val, test = split_data_by_timestamp(df,
                                                   train_split=args.train_split,
                                                   test_split=args.test_split,
                                                   validation_split=args.validation_split,
                                                   user_field=args.user_field,
                                                   time_field=args.time_field)
    elif args.split_type == "random":
        print("> Splitting randomly")
        train, val, test = random_split(df,
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
