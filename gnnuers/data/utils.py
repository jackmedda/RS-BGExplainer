import os
import shutil
import decimal

import tqdm
import pandas as pd
from pandas.api.types import is_numeric_dtype


# def split_data_temporally(interactions,
#                           start_train_split=0.80,
#                           test_split=None,
#                           validation_split=None,
#                           user_field='user_id',
#                           time_field='timestamp'):
#     # TODO: each temporal split could be yielded instead of returning an entire structure with all the splits
#     # TODO: funzione che trova il set di giorni più vicini al N% (e.g. 10%, 20%), in modo che i giorni siano un numero
#     #       tondo, tipo 90 giorni (3 mesi) o 100
#     # TODO: trovare un modo efficiente per controllare se un timestamp sta in un intervallo di tempo
#     train_set = []
#     test_set = []
#     val_set = []
#     groups = interactions.groupby([user_field])
#     for i, (_, group) in enumerate(tqdm.tqdm(groups, desc=f"Splitting data per_user")):
#         if time_field:
#             sorted_group = group.sort_values(time_field)
#         else:
#             sorted_group = group
#
#         if isinstance(train_split, float) or isinstance(test_split, float):
#             n_rating_train = int(len(sorted_group.index) * train_split) if train_split is not None else 0
#             n_rating_test = int(len(sorted_group.index) * test_split) if test_split is not None else 0
#             n_rating_val = int(len(sorted_group.index) * validation_split) if validation_split is not None else 0
#
#             if len(sorted_group.index) > (n_rating_train + n_rating_test + n_rating_val):
#                 n_rating_train += len(sorted_group.index) - (n_rating_train + n_rating_test + n_rating_val)
#         else:
#             raise ValueError(f"split type not accepted")
#
#         if n_rating_train == 0:
#             start_index = len(sorted_group) - n_rating_test
#             start_index = start_index - n_rating_val if n_rating_val is not None else start_index
#             train_set.append(sorted_group.iloc[:start_index])
#         else:
#             train_set.append(sorted_group.iloc[:n_rating_train])
#             start_index = n_rating_train
#
#         if n_rating_val > 0:
#             val_set.append(sorted_group.iloc[start_index:(start_index + n_rating_val)])
#             start_index += n_rating_val
#
#         if n_rating_test > 0:
#             test_set.append(sorted_group.iloc[start_index:(start_index + n_rating_test)])
#         else:
#             test_set.append(sorted_group.iloc[start_index:])
#
#     train, test = pd.concat(train_set), pd.concat(test_set)
#     validation = pd.concat(val_set) if val_set else None
#
#     return train, validation, test


def split_data_per_user(interactions,
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
        if time_field in group.columns:
            sorted_group = group.sort_values(time_field)
        else:
            sorted_group = group

        # if train_split is a tuple the last train_split[1] interactions in the train are moved to the test
        if isinstance(train_split, tuple):
            tr_split, tr_to_test = train_split
            if not isinstance(tr_to_test, int):
                raise ValueError("the number of interactions to move from train to test must be an integer")
            n_train, n_val, n_test = _compute_set_sizes(len(sorted_group), (tr_split, validation_split, test_split))
            tr_split = n_train - tr_to_test
            te_split = n_test + tr_to_test
            val_split = n_val
        else:
            tr_split, val_split, te_split = train_split, validation_split, test_split

        _add_split_interactions(
            sorted_group,
            (tr_split, val_split, te_split),
            (train_set, val_set, test_set)
        )

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

    _add_split_interactions(
        interactions,
        (train_split, validation_split, test_split),
        (train_set, val_set, test_set)
    )

    train, test = pd.concat(train_set), pd.concat(test_set)
    validation = pd.concat(val_set) if val_set else None

    return train, validation, test


def _add_split_interactions(dataset, splits, sets):
    train_set, val_set, test_set = sets
    if isinstance(splits[0], float):
        n_rating_train, n_rating_val, n_rating_test = _compute_set_sizes(len(dataset), splits)
    else:
        n_rating_train, n_rating_val, n_rating_test = splits

    if n_rating_train == 0:
        start_index = len(dataset) - n_rating_test
        start_index = start_index - n_rating_val if n_rating_val is not None else start_index
        train_set.append(dataset.iloc[:start_index])
    else:
        train_set.append(dataset.iloc[:n_rating_train])
        start_index = n_rating_train

    if n_rating_val > 0:
        val_set.append(dataset.iloc[start_index:(start_index + n_rating_val)])
        start_index += n_rating_val

    if n_rating_test > 0:
        test_set.append(dataset.iloc[start_index:(start_index + n_rating_test)])
    else:
        test_set.append(dataset.iloc[start_index:])


def _compute_set_sizes(n_data, splits):
    train_split, validation_split, test_split = splits

    if isinstance(train_split, float) or isinstance(test_split, float):
        n_rating_train = int(n_data * train_split) if train_split is not None else 0
        n_rating_val = int(n_data * validation_split) if validation_split is not None else 0
        n_rating_test = int(n_data * test_split) if test_split is not None else 0

        if n_data > (n_rating_train + n_rating_test + n_rating_val):
            n_rating_train += n_data - (n_rating_train + n_rating_test + n_rating_val)
    else:
        raise ValueError(f"split type not accepted")

    return n_rating_train, n_rating_val, n_rating_test


def filter_min_interactions(interactions, by='user_id', min_interactions=20):
    return interactions.groupby(by).filter(lambda x: len(x) >= min_interactions)


def copy_dataset(datasets_path, orig_dset_name, dset_new_name):
    dst = os.path.join(datasets_path, dset_new_name)
    shutil.copytree(os.path.join(datasets_path, orig_dset_name), dst, ignore=lambda x, y: ['splits_backup'])

    for dirname, _, filename in os.walk(dst, topdown=True):
        for data_type in recbole_extended_data_types:
            if f"{args.dataset}.{data_type}" == filename:
                os.rename(
                    os.path.join(dirname, filename),
                    os.path.join(dirname, f"{dset_new_name}.{data_type}")
                )


def add_token(df, user_df, args):
    for _df in [df, user_df]:
        new_cols = []
        for col in _df.columns:
            suffix = ':token'
            if is_numeric_dtype(_df[col]) and col not in [args.user_field, args.item_field]:
                suffix = ':float'
            new_cols.append(col + suffix)
        _df.columns = new_cols

    args.user_field += ':token'
    args.item_field += ':token'
    args.time_field += ':float'
