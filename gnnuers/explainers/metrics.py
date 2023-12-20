import torch

import gnnuers.evaluation as eval_utils


def get_exp_metric_value(exp_metric,
                         *args,
                         **kwargs):
    if exp_metric not in __exp_metrics_map__:
        raise NotImplementedError(f'exp_metric `{exp_metric}` is not implemented.')

    return __exp_metrics_map__[exp_metric](*args, **kwargs)


def _compute_explainer_consumer_DP_across_random_samples(pref_data,
                                                         eval_metric,
                                                         dset_name,
                                                         sens_attr,
                                                         batch_size,
                                                         iterations=100):
    # minus 1 encodes the demographic groups to {0, 1} instead of {1, 2}
    pref_data['Demo Group'] -= 1

    # it prevents from using memoization
    if hasattr(eval_utils.compute_DP_across_random_samples, "generated_groups"):
        if (dset_name, sens_attr) in eval_utils.compute_DP_across_random_samples.generated_groups:
            del eval_utils.compute_DP_across_random_samples.generated_groups[(dset_name, sens_attr)]

    exp_metric_value, _ = eval_utils.compute_DP_across_random_samples(
        pref_data, sens_attr, 'Demo Group', dset_name, eval_metric, iterations=iterations, batch_size=batch_size
    )

    return exp_metric_value[:, -1].mean()


def _compute_explainer_consumer_DP(pref_data,
                                   eval_metric,
                                   *args,
                                   **kwargs):
    gr_results = []
    for gr_mask in pref_data.groupby('Demo Group').groups.values():
        gr_results.append(pref_data.loc[gr_mask, eval_metric].mean())

    return eval_utils.compute_DP(*gr_results)


def _compute_explainer_provider_DP(pref_data,
                                   eval_metric,
                                   *args,
                                   **kwargs):
    dataset = kwargs.pop('dataset')
    discrim_attr = kwargs.pop('discriminative_attribute')
    groups_distrib = kwargs.pop('groups_distrib')

    return eval_utils.compute_provider_DP(pref_data, dataset, discrim_attr, groups_distrib, topk_column='topk_pred')


def _compute_explainer_UC(pref_data,
                          eval_metric,
                          *args,
                          **kwargs):
    min_rel_items = kwargs.get('coverage_min_relevant_items', 0)
    if min_rel_items > 0:
        raise NotImplementedError('`coverage_min_relevant_items` cannot be > 0. Not implemented')

    return (pref_data.loc[:, eval_metric] > min_rel_items).astype(int).sum() / pref_data.shape[0]


__exp_metrics_map__ = {
    "consumer_DP": _compute_explainer_consumer_DP,
    "consumer_DP_across_random_samples": _compute_explainer_consumer_DP_across_random_samples,
    "provider_DP": _compute_explainer_provider_DP,
    "UC": _compute_explainer_UC
}
