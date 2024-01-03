import torch
import numpy as np
from recbole.data import Interaction as RecboleInteraction


class Interaction(RecboleInteraction):

    def _setop_by_user_item_ids(self,
                                other,
                                op,
                                uid_field="user_id",
                                iid_field="item_id",
                                return_unique_counts=False):
        _, unique, counts = unique_cat_recbole_interaction(
            self.interaction,
            other.interaction if isinstance(other, Interaction) else other,
            uid_field=uid_field,
            iid_field=iid_field,
            return_unique_counts=True
        )

        if op == "diff":
            ret = self[counts == 1]
        elif op == "intersection":
            ret = self[counts > 1]
        else:
            raise ValueError(f"Interaction operation [set{op}] is not supported.")

        if return_unique_counts:
            return ret, unique, counts
        else:
            return ret

    def setdiff_by_user_item_ids(self,
                                 other,
                                 uid_field="user_id",
                                 iid_field="item_id",
                                 return_unique_counts=False):
        return self._setop_by_user_item_ids(
            other,
            "diff",
            uid_field=uid_field,
            iid_field=iid_field,
            return_unique_counts=return_unique_counts
        )

    def setintersection_by_user_item_ids(self,
                                         other,
                                         uid_field="user_id",
                                         iid_field="item_id",
                                         return_unique_counts=False):
        return self._setop_by_user_item_ids(
            other,
            "intersection",
            uid_field=uid_field,
            iid_field=iid_field,
            return_unique_counts=return_unique_counts
        )


def unique_cat_recbole_interaction(inter, other, uid_field='user_id', iid_field='item_id', return_unique_counts=False):
    if isinstance(inter, dict):
        _inter = torch.stack((torch.as_tensor(inter[uid_field]), torch.as_tensor(inter[iid_field])))
    else:
        _inter = torch.as_tensor(inter)

    if isinstance(other, dict):
        _other = torch.stack((torch.as_tensor(other[uid_field]), torch.as_tensor(other[iid_field])))
    else:
        _other = torch.as_tensor(other)
    unique, counts = torch.cat((_inter, _other), dim=1).unique(dim=1, return_counts=True)
    new_inter = unique[:, counts == 1]

    if not return_unique_counts:
        return dict(zip([uid_field, iid_field], new_inter))
    else:
        return dict(zip([uid_field, iid_field], new_inter)), unique, counts


def np_unique_cat_recbole_interaction(inter, other, uid_field='user_id', iid_field='item_id', return_unique_counts=False):
    _inter = np.stack((inter[uid_field], inter[iid_field])) if isinstance(inter, dict) else inter
    _other = np.stack((other[uid_field], other[iid_field])) if isinstance(other, dict) else other

    unique, counts = np.unique(np.concatenate((_inter, _other), axis=1), axis=1, return_counts=True)
    new_inter = unique[:, counts == 1]

    if not return_unique_counts:
        return dict(zip([uid_field, iid_field], new_inter))
    else:
        return dict(zip([uid_field, iid_field], new_inter)), unique, counts

