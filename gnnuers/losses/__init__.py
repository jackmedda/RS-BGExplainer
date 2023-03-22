from typing import Callable, Tuple, Dict

import gmpy2
import torch
import numpy as np


class NDCGApproxLoss(torch.nn.modules.loss._Loss):
    __constants__ = ['reduction']

    __MAX_TOPK_ITEMS__ = 10000

    def __init__(self, size_average=None, reduce=None, topk=None, reduction: str = 'mean', temperature=0.1) -> None:
        super(NDCGApproxLoss, self).__init__(size_average, reduce, reduction)
        self.topk = topk
        self.temperature = temperature

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


# class NDCGApproxLoss(torch.nn.modules.loss._Loss):
#     __constants__ = ['reduction']
#
#     __MAX_TOPK_ITEMS__ = 10000
#
#     def __init__(self, size_average=None, reduce=None, topk=None, reduction: str = 'mean', temperature=0.1) -> None:
#         super(NDCGApproxLoss, self).__init__(size_average, reduce, reduction)
#         self.topk = topk
#         self.temperature = temperature
#
#     def forward(self, _input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         if _input.shape[1] > self.__MAX_TOPK_ITEMS__:
#             topk = self.topk or target.shape[1]
#             _, _input_topk = torch.topk(_input, dim=1, k=topk)
#             _input = _input[torch.arange(_input.shape[0])[:, None], _input_topk]
#             target = target[torch.arange(target.shape[0])[:, None], _input_topk]
#
#         _input_temp = torch.nn.ReLU()(_input) / self.temperature
#
#         def approx_ranks(inp):
#             shape = inp.shape[1]
#
#             a = torch.tile(torch.unsqueeze(inp, 2), [1, 1, shape])
#             a = torch.transpose(a, 1, 2) - a
#             return torch.sum(torch.sigmoid(a), dim=-1) + .5
#
#         def inverse_max_dcg(_target,
#                             gain_fn=lambda _target: torch.pow(2.0, _target) - 1.,
#                             rank_discount_fn=lambda rank: 1. / rank.log1p()):
#             topk = self.topk or _target.shape[1]
#             ideal_sorted_target = torch.topk(_target, topk).values
#             rank = (torch.arange(ideal_sorted_target.shape[1]) + 1).to(_target.device)
#             discounted_gain = gain_fn(ideal_sorted_target).to(_target.device) * rank_discount_fn(rank)
#             discounted_gain = torch.sum(discounted_gain, dim=1, keepdim=True)
#             return torch.where(discounted_gain > 0., 1. / discounted_gain, torch.zeros_like(discounted_gain))
#
#         def ndcg(_target, _ranks):
#             topk = self.topk or _target.shape[1]
#             sorted_target, sorted_idxs = torch.topk(_target, topk)
#             discounts = 1. / _ranks[torch.arange(_ranks.shape[0])[:, None], sorted_idxs].log1p()
#             gains = torch.pow(2., sorted_target).to(_target.device) - 1.
#             dcg = torch.sum(gains * discounts, dim=-1, keepdim=True)
#             return dcg * inverse_max_dcg(_target)
#
#         ranks = approx_ranks(_input_temp)
#
#         return -ndcg(target, ranks)

class DPLoss(torch.nn.modules.loss._Loss):
    __constants__ = ['reduction']

    def __init__(self,
                 sensitive_attribute: str,
                 user_feat,
                 topk=10,
                 loss: Callable = NDCGApproxLoss,
                 adv_group_data: Tuple[str, int, float] = None,
                 previous_loss_value: Dict[str, np.ndarray] = None,
                 size_average=None, reduce=None, reduction: str = 'mean', temperature=0.1) -> None:
        super(DPLoss, self).__init__(size_average, reduce, reduction)

        self.loss: Callable = loss(
            size_average=size_average,
            topk=topk,
            reduce=reduce,
            reduction=reduction,
            temperature=temperature
        )
        self.sensitive_attribute = sensitive_attribute
        self.user_feat = user_feat
        self.adv_group_data = adv_group_data
        self.previous_loss_value = previous_loss_value

    def forward(self, _input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        groups = self.user_feat[self.sensitive_attribute].unique().numpy()
        masks = []
        for gr in groups:
            masks.append((self.user_feat[self.sensitive_attribute] == gr).numpy())
        masks = np.stack(masks)

        loss_values = self.loss(_input, target)

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

        loss = None
        for gr_i_idx in range(len(groups)):
            gr_i = groups[gr_i_idx]
            if self.adv_group_data[0] == "global":
                # the loss works to optimize loss towards -1, the global loss is however positive
                loss = (masked_loss[gr_i_idx] - (-self.adv_group_data[2])).abs()
            else:
                for gr_j_idx in range(gr_i_idx + 1, len(groups)):
                    l_val = masked_loss[gr_i_idx]
                    r_val = masked_loss[gr_j_idx]

                    if self.adv_group_data[0] == "local":
                        if self.adv_group_data[1] == gr_i:
                            l_val = l_val.detach()
                        else:
                            r_val = r_val.detach()

                    if loss is None:
                        loss = (l_val - r_val).abs()
                    else:
                        loss += (l_val - r_val).abs()

        return loss / max(int(gmpy2.comb(len(groups), 2)), 1)
