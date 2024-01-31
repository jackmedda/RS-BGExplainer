import os
import pickle
import logging

import torch
import numpy as np
import pandas as pd
from recbole.utils import FeatureType, FeatureSource, set_color

import gnnuers.utils as utils
from gnnuers.data import Dataset
from gnnuers.data.interaction import np_unique_cat_recbole_interaction


class PerturbedDataset(Dataset):
    FEATS = [FeatureSource.INTERACTION, FeatureSource.ITEM]  # our method never deletes users

    def __init__(self, config, explanations_path, best_exp):
        self.explanations_path = explanations_path
        self.__base_dataset = Dataset(config)  # must be instantiated to get the recbole mappings of user and item ids

        self.best_exp = best_exp
        self.best_explanation = None
        self.exps_config = None
        self.perturbed_edges = self.load_perturbed_edges()
        self.mapped_perturbed_edges = utils.remap_edges_recbole_ids(
            self.__base_dataset, self.perturbed_edges, field2id_token=True
        )

        super(PerturbedDataset, self).__init__(config)

    def get_best_explanation(self, exps, exps_config):
        if self.best_exp[0] == "fairest":
            best_exp = utils.get_best_exp_early_stopping(exps, exps_config)
        elif self.best_exp[0] == "fixed_exp":
            best_exp = utils.get_exp_by_epoch(exps, self.best_exp[1])
        else:
            raise ValueError("`best_exp` must be set to select the explanation of a specific epoch")

        return best_exp

    def load_perturbed_edges(self):
        logger = logging.getLogger()

        with open(os.path.join(self.explanations_path, 'cf_data.pkl'), 'rb') as exp_file:
            exps = pickle.load(exp_file)
        logger.info(f"Original Fair Loss: {exps[0][-1]}")

        self.exps_config = utils.read_recbole_config_skip_errors(
            os.path.join(self.explanations_path, 'config.yaml'),
            self.__base_dataset.config
        )
        logger.info(self.exps_config)

        if self.exps_config['exp_rec_data'] != 'valid':
            logger.warning('Performing Graph Augmentation on Explanations NOT produced on Validation Data.')

        self.best_explanation = self.get_best_explanation(exps, self.exps_config)

        return self.best_explanation[utils.exp_col_index('del_edges')]

    def perturb_split(self, split):
        if split in self.SPLITS:
            spl_file = os.path.join(self.config['data_path'], f"{self.config['dataset']}.{split}")
            if not os.path.isfile(spl_file):
                return None

            with open(spl_file, 'rb') as split_data_file:
                split_data = pickle.load(split_data_file)

            if isinstance(split_data[self.uid_field], torch.Tensor):
                split_data = {k: v.numpy() for k, v in split_data.items()}

            # Recbole tokens are always treated as strings (e.g., user\item ids)
            split_data = {k: v.astype(str) for k, v in split_data.items()}

            mapped_split_data = utils.remap_edges_recbole_ids(
                self.__base_dataset,
                np.stack((split_data[self.uid_field], split_data[self.iid_field])),
                field2id_token=False
            )

            new_split_data, unique, counts = np_unique_cat_recbole_interaction(
                mapped_split_data, self.perturbed_edges,
                uid_field=self.uid_field, iid_field=self.iid_field, return_unique_counts=True
            )

            if split != "train":
                # validation and test set should not be affected by any deleted edge in the training set
                # but if an edge is added to the training set, it must be removed from the validation or test set
                common_interactions = unique[:, counts > 1]
                new_split_data = np_unique_cat_recbole_interaction(
                    mapped_split_data, common_interactions, uid_field=self.uid_field, iid_field=self.iid_field
                )

            new_split_data = utils.remap_edges_recbole_ids(
                self.__base_dataset,
                np.stack((new_split_data[self.uid_field], new_split_data[self.iid_field])),
                field2id_token=True
            )

            return dict(zip([self.uid_field, self.iid_field], new_split_data))
        else:
            raise ValueError(f"split `{split}` is not supported for LRS configuration")

    def perturb_feat(self, source=FeatureSource.INTERACTION):
        if source in self.FEATS:
            feat_file = os.path.join(self.config['data_path'], f"{self.config['dataset']}.{source.value}")
            if not os.path.isfile(feat_file):
                return None  # it could be triggered if the dataset does not have item features (i.e. ".item" file)

            # new dataset is the merge of splits => it takes care of particular cases, like an edge in the validation
            # set that was added to the training set (it should then not be considered as deleted, and it should be in
            # the final dataset
            pert_splits = [self.perturb_split(split) for split in self.SPLITS]
            merged_splits = pert_splits[0]
            for p_split in pert_splits[1:]:
                for key in merged_splits:
                    merged_splits[key] = np.concatenate([merged_splits[key], p_split[key]])

            # Recbole tokens are always treated as strings (e.g., user\item ids)
            pert_df = pd.DataFrame(merged_splits).astype(str)

            feat_df_cols = [self.uid_field, self.iid_field]
            if source == FeatureSource.INTERACTION:
                feat_df = self.__base_dataset.inter_feat
            elif source == FeatureSource.ITEM and self.__base_dataset.item_feat is not None:
                feat_df = self.__base_dataset.item_feat
                pert_df = pert_df[[self.iid_field]]
            else:
                raise NotImplementedError(
                    f"feat {source} not supported for modification when re-training with perturbed data"
                )

            for field, token_map in self.__base_dataset.field2id_token.items():
                if field in feat_df.columns:
                    # Recbole tokens are always treated as strings (e.g., user\item ids)
                    feat_df[field] = feat_df[field].map(token_map.__getitem__).astype(str)
            new_feat_df = pert_df.join(feat_df.set_index(feat_df_cols), on=feat_df_cols, how="left")

            return new_feat_df
        else:
            return None

    def _remap(self, remap_list):
        tokens, split_point = self._concat_remaped_tokens(remap_list)

        common_field = remap_list[0][1]
        new_ids_list = [self.__base_dataset.field2token_id[common_field][orig_id] for orig_id in tokens]

        new_ids_list = np.split(new_ids_list, split_point)

        mp = self.__base_dataset.field2id_token[common_field]
        token_id = self.__base_dataset.field2token_id[common_field]

        for (feat, field, ftype), new_ids in zip(remap_list, new_ids_list):
            if field not in self.field2id_token:
                self.field2id_token[field] = mp
                self.field2token_id[field] = token_id
            if ftype == FeatureType.TOKEN:
                feat[field] = new_ids
            elif ftype == FeatureType.TOKEN_SEQ:
                split_point = np.cumsum(feat[field].agg(len))[:-1]
                feat[field] = np.split(new_ids, split_point)

    def _load_data_split(self, split):
        filename = os.path.join(self.dataset_path, f'{self.dataset_name}.{split}')
        if not os.path.isfile(filename):
            if split in ['train', 'test']:
                raise NotImplementedError(f'The splitting method "LRS" needs at least train and test.')

        split_data = self.perturb_split(split)

        return split_data

    def _load_feat(self, filepath, source):
        """Load features according to source into :class:`pandas.DataFrame`.

        Set features' properties, e.g. type, source and length.

        Args:
            filepath (str): path of input file.
            source (FeatureSource or str): source of input file.

        Returns:
            pandas.DataFrame: Loaded feature

        Note:
            For sequence features, ``seqlen`` will be loaded, but data in DataFrame will not be cut off.
            Their length is limited only after calling :meth:`~_dict_to_interaction` or
            :meth:`~_dataframe_to_interaction`
        """
        self.logger.debug(set_color(f'Loading feature from [{filepath}] (source: [{source}]).', 'green'))

        load_col, unload_col = self._get_load_and_unload_col(source)
        if load_col == set():
            return None

        field_separator = self.config['field_separator']
        columns = []
        usecols = []
        dtype = {}
        encoding = self.config['encoding']
        with open(filepath, 'r', encoding=encoding) as f:
            head = f.readline()[:-1]
        for field_type in head.split(field_separator):
            field, ftype = field_type.split(':')
            try:
                ftype = FeatureType(ftype)
            except ValueError:
                raise ValueError(f'Type {ftype} from field {field} is not supported.')
            if load_col is not None and field not in load_col:
                continue
            if unload_col is not None and field in unload_col:
                continue
            if isinstance(source, FeatureSource) or source != 'link':
                self.field2source[field] = source
                self.field2type[field] = ftype
                if not ftype.value.endswith('seq'):
                    self.field2seqlen[field] = 1
            columns.append(field)
            usecols.append(field_type)
            dtype[field_type] = np.float64 if ftype == FeatureType.FLOAT else str

        if len(columns) == 0:
            self.logger.warning(f'No columns has been loaded from [{source}]')
            return None

        df: pd.DataFrame = self.perturb_feat(source)
        if df is None or source not in self.FEATS:
            df = pd.read_csv(
                filepath, delimiter=field_separator, usecols=usecols, dtype=dtype, encoding=encoding, engine='python'
            )
            df.columns = columns

        seq_separator = self.config['seq_separator']
        for field in columns:
            ftype = self.field2type[field]
            if not ftype.value.endswith('seq'):
                continue
            df[field].fillna(value='', inplace=True)
            if ftype == FeatureType.TOKEN_SEQ:
                df[field] = [np.array(list(filter(None, _.split(seq_separator)))) for _ in df[field].values]
            elif ftype == FeatureType.FLOAT_SEQ:
                df[field] = [np.array(list(map(float, filter(None, _.split(seq_separator))))) for _ in df[field].values]
            self.field2seqlen[field] = max(map(len, df[field].values))
        return df
