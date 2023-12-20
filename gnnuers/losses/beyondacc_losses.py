from typing import Tuple, Dict, Type

import gmpy2
import torch
import numpy as np

from gnnuers.losses.ranking_losses import RankingLoss, NDCGApproxLoss


class BeyondAccLoss(torch.nn.modules.loss._Loss):
    __constants__ = ['reduction']

    def __init__(self,
                 size_average=None, reduce=None, reduction: str = 'mean', temperature=0.01, topk=10, **kwargs) -> None:
        super(BeyondAccLoss, self).__init__(size_average, reduce, reduction)
        self.temperature = temperature
        self.topk = topk

    def loss_type(self):
        return self.__class__.__name__

    def forward(self, _input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("subclasses must implement this method")

    def is_data_feat_needed(self):
        raise NotImplementedError("subclasses must implement this method")


class FairLoss(BeyondAccLoss):
    def __init__(self,
                 discriminative_attribute: str,
                 size_average=None, reduce=None, reduction: str = 'mean', temperature=0.01, **kwargs) -> None:
        super(FairLoss, self).__init__(size_average, reduce, reduction, temperature, **kwargs)

        self.discriminative_attribute = discriminative_attribute
        self.data_feat = None

    def loss_type(self):
        if 'consumer' in self.__class__.__name__.lower():
            return 'Consumer'
        elif 'provider' in self.__class__.__name__.lower():
            return 'Provider'
        else:
            raise ValueError('A `FairLoss` subclass should contain "Consumer" or "Provider" in its name')

    def update_data_feat(self, data_feat):
        self.data_feat = data_feat

    def forward(self, _input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("subclasses must implement this method")

    def is_data_feat_needed(self):
        return True


class ConsumerDPLoss(FairLoss):
    def __init__(self,
                 sensitive_attribute: str,
                 loss: Type[RankingLoss] = NDCGApproxLoss,
                 adv_group_data: Tuple[str, int, float] = None,
                 size_average=None, reduce=None, reduction: str = 'mean', temperature=0.01, **kwargs) -> None:
        super(ConsumerDPLoss, self).__init__(
            sensitive_attribute,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            temperature=temperature,
            **kwargs
        )

        self.ranking_loss_function: RankingLoss = loss(
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            temperature=temperature,
            topk=self.topk
        )

        self.adv_group_data = adv_group_data
        self.previous_loss_value = None
        self.deactivate_gradient = kwargs.get("deactivate_gradient", True)

    def update_previous_loss_value(self, prev_loss_value):
        self.previous_loss_value = prev_loss_value

    def forward(self, _input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.data_feat is None:
            raise AttributeError('each forward call should be preceded by a call of `update_data_feat`')

        groups = self.data_feat[self.discriminative_attribute].unique().numpy()
        masks = []
        for gr in groups:
            masks.append((self.data_feat[self.discriminative_attribute] == gr).numpy())
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
                    fair_loss = (masked_loss[gr_i_idx] - (-self.adv_group_data[2])).abs() * -1
                    total_loss = fair_loss if total_loss is None else total_loss + fair_loss
            else:
                gr_i = groups[gr_i_idx]
                for gr_j_idx in range(gr_i_idx + 1, len(groups)):
                    l_val = masked_loss[gr_i_idx]
                    r_val = masked_loss[gr_j_idx]

                    if self.adv_group_data[0] == "local" and self.deactivate_gradient:
                        if self.adv_group_data[1] == gr_i:
                            l_val = l_val.detach()
                        else:
                            r_val = r_val.detach()

                    fair_loss = (l_val - r_val).abs() * -1
                    total_loss = fair_loss if total_loss is None else total_loss + fair_loss

        self.update_data_feat(None)

        return fair_loss / max(int(gmpy2.comb(len(groups), 2)), 1)


class ProviderDPLoss(FairLoss):
    __TOPK_OFFSET__ = 0.1

    def __init__(self,
                 discriminative_attribute: str,
                 adv_group_data: Tuple[str, int, float] = None,
                 groups_distrib: Dict[int, float] = None,
                 size_average=None, reduce=None, reduction: str = 'mean', temperature=0.01, **kwargs) -> None:
        super(ProviderDPLoss, self).__init__(
            discriminative_attribute,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            temperature=temperature,
            **kwargs
        )

        self.adv_group_data = adv_group_data
        self.groups_distrib = groups_distrib

    def forward(self, _input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Instead of estimating the visibility on the top-k, we take the top 10% of items, because the top-k could not
        include items of both groups (short-head, long-tail) due to recommendation of only short-head items or because
        the top-k during training include also interactions in the training set.
        """
        if self.data_feat is None:
            raise AttributeError('each forward call should be preceded by a call of `update_data_feat`')

        _input = _input[:, 1:]  # data_feat does not have padding item 0

        input_topk_vals, input_topk_idxs = torch.topk(_input, k=round(_input.shape[1] * self.__TOPK_OFFSET__), dim=1)

        groups = self.data_feat[self.discriminative_attribute].unique().numpy()
        groups_recs_distrib = []
        targets = []
        for i, gr in enumerate(groups):
            mask = (self.data_feat[self.discriminative_attribute].to(_input.device)[input_topk_idxs] == gr).to(_input.device)

            if self.discriminative_attribute.lower() == 'visibility':
                input_topk_vals = torch.where(mask, input_topk_vals, 0)  # reduce the relevance of items not in the group

                mask = mask.float()
                mask_sum = mask.sum(dim=1, keepdim=True)
                padded_mask = torch.where(mask_sum > 0, mask, torch.ones_like(mask) * 1e-7)
                padded_mask_sum = padded_mask.sum(dim=1, keepdim=True)
                mask = torch.nan_to_num(padded_mask / padded_mask_sum, nan=0)

                visibility = torch.nn.BCEWithLogitsLoss(reduction='mean')(input_topk_vals, mask)

                groups_recs_distrib.append(visibility)
            elif self.discriminative_attribute.lower() == 'exposure':
                exposure = torch.where(mask, input_topk_vals, 0).sum(dim=1)

                groups_recs_distrib.append(exposure)
            else:
                raise NotImplementedError(f'Provider fairness loss with discriminative attribute `{self.discriminative_attribute}` is not supported')

        disparity = groups_recs_distrib[0] / self.groups_distrib[0] - groups_recs_distrib[1] / self.groups_distrib[1]
        if self.discriminative_attribute.lower() == 'visibility':
            fair_loss = disparity.abs() * -1
        elif self.discriminative_attribute.lower() == 'exposure':
            fair_loss = (disparity.sum() / input_topk_vals.sum()).abs() * -1

        return fair_loss


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
        self.only_relevant = kwargs.get('only_relevant', True)

    def is_data_feat_needed(self):
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

        # the 1st only increases the relevance of relevant items, the 2nd also decreases the relevance of non-relevant items
        if self.only_relevant:
            loss = torch.where(target == 1, _input_temp * -1 - 1e-7, torch.relu(_input_temp * -1)).mean()
        else:
            loss = torch.where(target == 1, _input_temp * -1 - 1e-7, _input_temp).mean()

        return torch.nan_to_num(loss, nan=0)  # if all users have enough relevant items `loss` is NaN
