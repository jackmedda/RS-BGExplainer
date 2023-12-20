import abc

import torch


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


class SigmoidBCELoss(RankingLoss):
    """"
    Based on TensorFlow Ranking SigmoidBCELoss
    """

    def __init__(self, size_average=None, reduce=None, topk=None, reduction: str = 'mean', temperature=0.1) -> None:
        super(SigmoidBCELoss, self).__init__(
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            topk=topk,
            temperature=temperature
        )

    def forward(self, _input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _input_temp = _input / self.temperature

        target_sum = target.sum(dim=1, keepdim=True)
        padded_target = torch.where(target_sum > 0, target, torch.ones_like(target) * 1e-7)
        padded_target_sum = padded_target.sum(dim=1, keepdim=True)
        target = torch.nan_to_num(padded_target / padded_target_sum, nan=0)

        return torch.nn.BCEWithLogitsLoss(reduction='none')(_input_temp, target).mean(dim=1, keepdim=True)
