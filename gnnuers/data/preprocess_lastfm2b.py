import os
import argparse
import pickle as pkl
from datetime import datetime

import tqdm
import pandas as pd  # better use pandas >= 2.0.0
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folderpath', required=True)
    parser.add_argument('--chunksize', default=50_000_000)
    parser.add_argument('--min_interactions', default=1_000)
    parser.add_argument('--overwrite', action="store_true")

    args = parser.parse_args()

    le_size = 2_014_164_872
    lc_size = 519_293_333

    pp_users_filepath = os.path.join(args.folderpath, 'preprocessed_lastfm2b_users.tsv')
    if args.overwrite or not os.path.exists(pp_users_filepath):
        users_df = pd.read_csv(os.path.join(args.folderpath, 'users.tsv'), sep='\t', on_bad_lines='warn')

        age_mask = (users_df.age > 0) & (users_df.age < 120)
        gender_mask = (users_df.gender == 'm') | (users_df.gender == 'f')
        users_df = users_df[age_mask & gender_mask]
        users_df['gender'] = users_df.gender.str.upper()
        users_df.to_csv(pp_users_filepath, sep='\t', index=None)
    else:
        users_df = pd.read_csv(pp_users_filepath, sep='\t')

    tracks_albums_filepath = os.path.join(args.folderpath, 'tracks_albums_df.tsv')
    if args.overwrite or not os.path.exists(tracks_albums_filepath):
        tracks_df = pd.read_csv(
            os.path.join(args.folderpath, 'tracks.tsv'),
            usecols=['track_id', 'artist_name'],
            sep='\t', on_bad_lines='warn'
        )
        albums_df = pd.read_csv(
            os.path.join(args.folderpath, 'albums.tsv'),
            usecols=['album_id', 'artist_name'],
            sep='\t', on_bad_lines='warn'
        )

        tracks_albums_ids_filepath = os.path.join(args.folderpath, 'tracks_albums_ids.npy')
        if args.overwrite or not os.path.exists(tracks_albums_ids_filepath):
            tracks_albums_ids = None
            le_kwargs = dict(
                usecols=['track_id', 'album_id'],
                sep='\t',
                on_bad_lines='warn',
                chunksize=args.chunksize
            )
            with pd.read_csv(os.path.join(args.folderpath, 'listening-events.tsv'), **le_kwargs) as reader:
                for chunk in tqdm.tqdm(
                    reader, total=int(np.ceil(le_size / args.chunksize)),
                    desc='Extracting the tracks and albums ids to identify unique artists'
                ):
                    if tracks_albums_ids is None:
                        tracks_albums_ids = np.unique(chunk.values, axis=0)
                    else:
                        tracks_albums_ids = np.unique(
                            np.concatenate((tracks_albums_ids, np.unique(chunk.values, axis=0)), axis=0),
                            axis=0
                        )
            if tracks_albums_ids is not None:
                np.save(tracks_albums_ids_filepath, tracks_albums_ids)
        else:
            tracks_albums_ids = np.load(tracks_albums_ids_filepath)

        # import pdb; pdb.set_trace()
        tr_alb_df = pd.DataFrame(tracks_albums_ids, columns=['track_id', 'album_id'])
        tr_alb_df = tr_alb_df.join(tracks_df.set_index('track_id'), on='track_id')
        tr_alb_df = tr_alb_df.join(albums_df.set_index('album_id'), on='album_id', rsuffix='_alb')

        an_na = tr_alb_df['artist_name'].isna() & ~tr_alb_df['artist_name_alb'].isna()
        an_alb_na = ~tr_alb_df['artist_name'].isna() & tr_alb_df['artist_name_alb'].isna()

        tr_alb_df.loc[an_na, 'artist_name'] = tr_alb_df.loc[an_na, 'artist_name_alb']
        tr_alb_df.loc[an_alb_na, 'artist_name_alb'] = tr_alb_df.loc[an_alb_na, 'artist_name']

        tr_alb_df = tr_alb_df[tr_alb_df.artist_name == tr_alb_df.artist_name_alb]  # removes also NaN
        del tr_alb_df['artist_name_alb']

        an_unique = tr_alb_df['artist_name'].unique()
        an_aid_map = dict(zip(an_unique, np.arange(an_unique.shape[0])))
        tr_alb_df['artist_id'] = tr_alb_df['artist_name'].map(an_aid_map)

        # some tracks can be in multiple albums
        # now we delete some tracks with equal ids but different artist_names
        tr_aid = tr_alb_df[['track_id', 'artist_id']].values
        tr_aid_unq = np.unique(tr_aid, axis=0)
        tr_unq, tr_cnt = np.unique(tr_aid_unq[:, 0], return_counts=True)
        bad_tracks = tr_unq[tr_cnt > 1]

        del tr_alb_df['artist_id']

        tr_alb_df = tr_alb_df[~tr_alb_df.track_id.isin(bad_tracks)]

        tr_alb_df.to_csv(tracks_albums_filepath, sep='\t', index=None)
    else:
        tr_alb_df = pd.read_csv(tracks_albums_filepath, sep='\t')

    track_artist_map = dict(tr_alb_df[['track_id', 'artist_name']].values)
    del tr_alb_df

    ua_df_filepath = os.path.join(args.folderpath, 'user_artist_df.tsv')
    if args.overwrite or not os.path.exists(ua_df_filepath):
        user_artist_count = []
        lc_kwargs = dict(
            sep='\t',
            on_bad_lines='warn',
            chunksize=args.chunksize
        )
        with pd.read_csv(os.path.join(args.folderpath, 'listening-counts.tsv'), **lc_kwargs) as reader:
            for chunk in tqdm.tqdm(
                reader, total=int(np.ceil(lc_size / args.chunksize)),
                desc='Extracting the listening counts for each track and taking the sum grouping by artist name'
            ):
                chunk['artist_name'] = chunk['track_id'].map(lambda x: track_artist_map.get(x, None))
                chunk.dropna(inplace=True)
                del chunk['track_id']
                user_artist_count.append(
                    chunk.groupby(['user_id', 'artist_name']).sum().reset_index()
                )

        # import pdb; pdb.set_trace()
        ua_df = pd.concat(user_artist_count, ignore_index=True)
        ua_df = ua_df.groupby(['user_id', 'artist_name']).sum().reset_index()

        ua_df.to_csv(ua_df_filepath, sep='\t', index=None)
    else:
        ua_df = pd.read_csv(ua_df_filepath, sep='\t')

    # only users with gender and age
    ua_df = ua_df[ua_df.user_id.isin(users_df.user_id)]
    ua_df_idx = ua_df.set_index(['user_id', 'artist_name'])

    ua_last_timestamp_path = os.path.join(args.folderpath, 'user_artist_last_timestamp.npy')
    if args.overwrite or not os.path.exists(ua_last_timestamp_df_path):
        unique_ua_pos = dict(zip(ua_df_idx.index.tolist(), np.arange(len(ua_df_idx))))
        ua_last_timestamp = np.full((len(ua_df_idx),), -1, dtype=int)
        le_kwargs = dict(
            usecols=['user_id', 'track_id', 'timestamp'],
            sep='\t',
            on_bad_lines='warn',
            chunksize=args.chunksize
        )
        with pd.read_csv(os.path.join(args.folderpath, 'listening-events.tsv'), **le_kwargs) as reader:
            for chunk in tqdm.tqdm(
                reader, total=int(np.ceil(le_size / args.chunksize)),
                desc='Retrieving the last timestamp each artist (the track of the artist) was listened to'
            ):
                chunk['timestamp'] = chunk['timestamp'].map(lambda x: int(datetime.fromisoformat(x).timestamp()))
                chunk['artist_name'] = chunk['track_id'].map(lambda x: track_artist_map.get(x, None))
                chunk.dropna(inplace=True)
                del chunk['track_id']
                chunk_time = chunk.groupby(['user_id', 'artist_name']).max().to_dict()['timestamp']
                for k, v in chunk_time.items():
                    if k in unique_ua_pos:
                        ua_last_timestamp[unique_ua_pos[k]] = max(unique_ua_pos[k], v)

        del unique_ua_pos
        np.save(ua_last_timestamp_path, ua_last_timestamp)
    else:
        ua_last_timestamp = np.load(ua_last_timestamp_path)

    ua_df_idx['timestamp'] = ua_last_timestamp
    ua_df_idx.to_csv(os.path.join(args.folderpath, 'preprocessed_lastfm2b_inter.tsv'), sep='\t')

    ua_df_1000 = ua_df_idx.reset_index().groupby('user_id').filter(lambda x: len(x) >= args.min_interactions)
    ua_df_idx.to_csv(os.path.join(args.folderpath, 'preprocessed_lastfm2b_inter_1000.tsv'), sep='\t', index=None)

    ua_df_idx.to_csv(os.path.join(args.folderpath, 'preprocessed_lastfm2b_users_1000.tsv'), sep='\t', index=None)
    users_df = users_df[users_df.user_id.isin(ua_df_idx.user_id.unique())]
    users_df.to_csv(os.path.join(args.folderpath, 'preprocessed_lastfm2b_users_1000.tsv'), sep='\t', index=None)
