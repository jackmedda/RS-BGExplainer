import abc
from typing import Tuple, Dict, Type

import gmpy2
import torch
import numpy as np


class RankingLoss(torch.nn.modules.loss._Loss, metaclass=abc.ABCMeta):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', temperature=0.01, **kwargs) -> None:
        super(RankingLoss, self).__init__(size_average, reduce, reduction)
        self.temperature = temperature

    @abc.abstractmethod
    def forward(self, _input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("subclasses must implement this method")


class TopKLoss(RankingLoss):
    def __init__(self, size_average=None, reduce=None, topk=None, reduction: str = 'mean', temperature=0.01) -> None:
        super(TopKLoss, self).__init__(
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            temperature=temperature
        )
        self.topk = topk

    def forward(self, _input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("subclasses must implement this method")


class NDCGApproxLoss(TopKLoss):
    __MAX_TOPK_ITEMS__ = 10000

    def __init__(self, size_average=None, reduce=None, topk=None, reduction: str = 'mean', temperature=0.01) -> None:
        """
        Lower values of `temperature` makes the loss more accurate in approximating NDCG
        """

        super(NDCGApproxLoss, self).__init__(
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            topk=topk,
            temperature=temperature
        )

    def forward(self, _input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if _input.shape[1] > self.__MAX_TOPK_ITEMS__:
            topk = self.topk or target.shape[1]
            _, _input_topk = torch.topk(_input, dim=1, k=topk)
            _input = _input[torch.arange(_input.shape[0])[:, None], _input_topk]
            target = target[torch.arange(target.shape[0])[:, None], _input_topk]

        _input_temp = torch.nn.ReLU()(_input) / self.temperature

        def approx_ranks(inp):
            shape = inp.shape[1]

            a = torch.tile(torch.unsqueeze(inp, 2), [1, 1, shape])
            a = torch.transpose(a, 1, 2) - a
            return torch.sum(torch.sigmoid(a), dim=-1) + .5

        def inverse_max_dcg(_target,
                            gain_fn=lambda _target: torch.pow(2.0, _target) - 1.,
                            rank_discount_fn=lambda rank: 1. / rank.log1p()):
            topk = self.topk or _target.shape[1]
            ideal_sorted_target = torch.topk(_target, topk).values
            rank = (torch.arange(ideal_sorted_target.shape[1]) + 1).to(_target.device)
            discounted_gain = gain_fn(ideal_sorted_target).to(_target.device) * rank_discount_fn(rank)
            discounted_gain = torch.sum(discounted_gain, dim=1, keepdim=True)
            return torch.where(discounted_gain > 0., 1. / discounted_gain, torch.zeros_like(discounted_gain))

        def ndcg(_target, _ranks):
            topk = self.topk or _target.shape[1]
            sorted_target, sorted_idxs = torch.topk(_target, topk)
            discounts = 1. / _ranks[torch.arange(_ranks.shape[0])[:, None], sorted_idxs].log1p()
            gains = torch.pow(2., sorted_target).to(_target.device) - 1.
            dcg = torch.sum(gains * discounts, dim=-1, keepdim=True)
            return dcg * inverse_max_dcg(_target)

        ranks = approx_ranks(_input_temp)

        return -ndcg(target, ranks)


class SoftmaxLoss(RankingLoss):
    """"
    Based on TensorFlow Ranking SoftmaxLoss
    """

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', temperature=0.1, **kwargs) -> None:
        super(SoftmaxLoss, self).__init__(
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            temperature=temperature,
            **kwargs
        )

    def forward(self, _input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _input_temp = _input / self.temperature

        target_sum = target.sum(dim=1, keepdim=True)
        padded_target = torch.where(target_sum > 0, target, torch.ones_like(target) * 1e-7)
        padded_target_sum = padded_target.sum(dim=1, keepdim=True)
        target = torch.nan_to_num(padded_target / padded_target_sum, nan=0)

        return torch.nn.BCEWithLogitsLoss(reduction='none')(_input_temp, target).mean(dim=1, keepdim=True)


class BeyondAccLoss(torch.nn.modules.loss._Loss):
    __constants__ = ['reduction']

    def __init__(self,
                 size_average=None, reduce=None, reduction: str = 'mean', temperature=0.01, topk=10, **kwargs) -> None:
        super(BeyondAccLoss, self).__init__(size_average, reduce, reduction)
        self.temperature = temperature
        self.topk = topk

    def loss_type(self):
        raise NotImplementedError("subclasses must implement this method")

    def forward(self, _input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("subclasses must implement this method")

    def is_user_feat_needed(self):
        raise NotImplementedError("subclasses must implement this method")


class FairLoss(BeyondAccLoss):
    def __init__(self,
                 sensitive_attribute: str,
                 loss: Type[RankingLoss] = NDCGApproxLoss,
                 size_average=None, reduce=None, reduction: str = 'mean', temperature=0.01, topk=10, **kwargs) -> None:
        super(FairLoss, self).__init__(size_average, reduce, reduction, temperature, topk, **kwargs)

        self.ranking_loss_function: RankingLoss = loss(
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            temperature=temperature,
            topk=self.topk,
            **kwargs
        )
        self.sensitive_attribute = sensitive_attribute
        self.user_feat = None

    def loss_type(self):
        return FairLoss.__name__

    def update_user_feat(self, user_feat):
        self.user_feat = user_feat

    def forward(self, _input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("subclasses must implement this method")

    def is_user_feat_needed(self):
        return True


class DPLoss(FairLoss):
    def __init__(self,
                 sensitive_attribute: str,
                 loss: Type[RankingLoss] = NDCGApproxLoss,
                 adv_group_data: Tuple[str, int, float] = None,
                 size_average=None, reduce=None, reduction: str = 'mean', temperature=0.01, **kwargs) -> None:
        super(DPLoss, self).__init__(
            sensitive_attribute,
            loss=loss,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            temperature=temperature,
            **kwargs
        )

        self.adv_group_data = adv_group_data
        self.previous_loss_value = None

    def update_previous_loss_value(self, prev_loss_value):
        self.previous_loss_value = prev_loss_value

    def forward(self, _input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.user_feat is None:
            raise AttributeError('each forward call should be preceded by a call of `update_user_feat`')

        groups = self.user_feat[self.sensitive_attribute].unique().numpy()
        masks = []
        for gr in groups:
            masks.append((self.user_feat[self.sensitive_attribute] == gr).numpy())
        masks = np.stack(masks)

        loss_values = self.ranking_loss_function(_input, target)

        masked_loss = []
        for gr, mask in zip(groups, masks):
            if self.previous_loss_value is not None:
                gr_loss = (loss_values[mask].sum() + self.previous_loss_value[gr].sum())
                gr_loss = gr_loss / (loss_values[mask].shape[0] + self.previous_loss_value[gr].shape[0])
                masked_loss.append(gr_loss.unsqueeze(dim=-1))
                if self.previous_loss_value[gr].shape[0] > 0:
                    self.previous_loss_value[gr] = np.concatenate([
                        loss_values[mask].detach().cpu().numpy(),
                        self.previous_loss_value[gr]
                    ], axis=0)
                else:
                    self.previous_loss_value[gr] = loss_values[mask].detach().cpu().numpy()
            else:
                masked_loss.append(loss_values[mask].mean(dim=0))
        masked_loss = torch.stack(masked_loss)

        fair_loss = None
        total_loss = None
        for gr_i_idx in range(len(groups)):
            if self.adv_group_data[0] == "global":
                if groups[gr_i_idx] != self.adv_group_data[1]:
                    # the loss optimizes towards -1, but the global loss is positive
                    fair_loss = (masked_loss[gr_i_idx] - (-self.adv_group_data[2])).abs()
                    total_loss = fair_loss if total_loss is None else total_loss + fair_loss
            else:
                gr_i = groups[gr_i_idx]
                for gr_j_idx in range(gr_i_idx + 1, len(groups)):
                    l_val = masked_loss[gr_i_idx]
                    r_val = masked_loss[gr_j_idx]

                    if self.adv_group_data[0] == "local":
                        if self.adv_group_data[1] == gr_i:
                            l_val = l_val.detach()
                        else:
                            r_val = r_val.detach()

                    fair_loss = (l_val - r_val).abs()
                    total_loss = fair_loss if total_loss is None else total_loss + fair_loss

        self.update_user_feat(None)

        return fair_loss / max(int(gmpy2.comb(len(groups), 2)), 1)


class UserCoverageLoss(BeyondAccLoss):

    def __init__(self,
                 min_relevant_items=0,
                 size_average=None, reduce=None, reduction: str = 'mean', temperature=0.1, **kwargs) -> None:
        super(UserCoverageLoss, self).__init__(
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            temperature=temperature,
            **kwargs
        )
        self.min_relevant_items = min(min_relevant_items, round(0.5 * self.topk))

    def loss_type(self):
        return BeyondAccLoss.__name__

    def is_user_feat_needed(self):
        return False

    def forward(self, _input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _input_temp = _input / self.temperature

        _, _input_topk_idxs = torch.topk(_input, k=self.topk, dim=1)
        relevant_idxs = target.gather(dim=1, index=_input_topk_idxs)
        relevant_recs = relevant_idxs.sum(dim=1)

        # take only users with #relevant_items < min_relevant_items
        rel_mask = relevant_recs <= self.min_relevant_items

        # exclude negative relevance
        _input_temp = torch.relu(_input_temp)[rel_mask]
        target = target[rel_mask]

        loss = torch.where(target == 1, _input_temp * -1 - 1e-7, torch.relu(_input_temp * -1)).mean()

        return torch.nan_to_num(loss, nan=0)  # if all users have enough relevant items `loss` is NaN


def get_ranking_loss(loss="ndcg"):
    return {
        "ndcg": NDCGApproxLoss,
        "softmax": SoftmaxLoss
    }[loss.lower()]


def get_beyondacc_loss(loss="dp"):
    return {
        "dp": DPLoss,
        "uc": UserCoverageLoss
    }[loss.lower()]