import os
import pickle
import logging

import torch
import numpy as np
import pandas as pd
from recbole.data.interaction import Interaction
from recbole.data.dataset import Dataset as RecboleDataset


class Dataset(RecboleDataset):
    SPLITS = ['train', 'validation', 'test']

    def __init__(self, config):
        if 'LRS' not in config['eval_args']['split']:
            raise ValueError("Perturbed graph can be used only when splits are loaded.")

        super(Dataset, self).__init__(config)

    def split_by_loaded_splits(self):
        next_df = []
        for split in self.SPLITS:
            filename = os.path.join(self.dataset_path, f'{self.dataset_name}.{split}')
            if not os.path.isfile(filename):
                if split in ['train', 'test']:
                    raise NotImplementedError(f'The splitting method "LRS" needs at least train and test.')
            else:
                split_data = self.perturb_split(split)

                split_keys = list(split_data.keys())
                if self.uid_field not in split_keys or self.iid_field not in split_keys or len(split_keys) != 2:
                    raise ValueError(f'The splitting grouping method "LRS" should contain only the fields '
                                     f'`{self.uid_field}` and `{self.iid_field}`.')

                for field in [self.uid_field, self.iid_field]:
                    data = split_data[field]
                    data = data.numpy() if isinstance(data, torch.Tensor) else data
                    split_data[field] = torch.LongTensor([self.field2token_id[field][val] for val in data.astype(str)])

                next_df.append(Interaction(split_data))

        self._drop_unused_col()
        next_ds = [self.copy(_) for _ in next_df]
        return next_ds

    def build(self):
        """Processing dataset according to evaluation setting, including Group, Order and Split.
        See :class:`~recbole.config.eval_setting.EvalSetting` for details.

        Returns:
            list: List of built :class:`Dataset`.
        """
        self._change_feat_format()

        if self.benchmark_filename_list is not None:
            self._drop_unused_col()
            cumsum = list(np.cumsum(self.file_size_list))
            datasets = [
                self.copy(self.inter_feat[start:end])
                for start, end in zip([0] + cumsum[:-1], cumsum)
            ]
            return datasets

        # ordering
        ordering_args = self.config["eval_args"]["order"]
        if ordering_args == "RO":
            self.shuffle()
        elif ordering_args == "TO":
            self.sort(by=self.time_field)
        else:
            raise NotImplementedError(
                f"The ordering_method [{ordering_args}] has not been implemented."
            )

        # splitting & grouping
        split_args = self.config["eval_args"]["split"]
        if split_args is None:
            raise ValueError("The split_args in eval_args should not be None.")
        if not isinstance(split_args, dict):
            raise ValueError(f"The split_args [{split_args}] should be a dict.")

        split_mode = list(split_args.keys())[0]
        assert len(split_args.keys()) == 1
        group_by = self.config["eval_args"]["group_by"]
        if split_mode == "RS":
            if not isinstance(split_args["RS"], list):
                raise ValueError(f'The value of "RS" [{split_args}] should be a list.')
            if group_by is None or group_by.lower() == "none":
                datasets = self.split_by_ratio(split_args["RS"], group_by=None)
            elif group_by == "user":
                datasets = self.split_by_ratio(
                    split_args["RS"], group_by=self.uid_field
                )
            else:
                raise NotImplementedError(
                    f"The grouping method [{group_by}] has not been implemented."
                )
        elif split_mode == "LS":
            datasets = self.leave_one_out(
                group_by=self.uid_field, leave_one_mode=split_args["LS"]
            )
        elif split_mode == "LRS":  # Load Ready Splits
            datasets = self.split_by_loaded_splits()
        else:
            raise NotImplementedError(
                f"The splitting_method [{split_mode}] has not been implemented."
            )

        return datasets
