from .ranking_losses import NDCGApproxLoss, SigmoidBCELoss
from .beyondacc_losses import ConsumerDPLoss, ProviderDPLoss, UserCoverageLoss


def get_ranking_loss(loss="ndcg"):
    return {
        "ndcg": NDCGApproxLoss,
        "sigmoid_bce": SigmoidBCELoss
    }[loss.lower()]


def get_beyondacc_loss(loss="consumer_dp"):
    return {
        "consumer_dp": ConsumerDPLoss,
        "provider_dp": ProviderDPLoss,
        "uc": UserCoverageLoss
    }[loss.lower()]


__exp_metrics_to_losses_map__ = {
    "consumer_DP": ConsumerDPLoss,
    "consumer_DP_across_random_samples": ConsumerDPLoss,
    "UC": UserCoverageLoss,
    "provider_DP": ProviderDPLoss
}


def get_loss_from_exp_metric(exp_metric):
    if exp_metric in __exp_metrics_to_losses_map__:
        return __exp_metrics_to_losses_map__[exp_metric]
    else:
        raise NotImplementedError(f'The exp_metric `{exp_metric}` is not supported')
