import numpy as np

import cpfair_robust.evaluation as eval_utils


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

    groups_distrib = [groups_distrib[0] / groups_distrib[1], 1 - groups_distrib[0] / groups_distrib[1]]

    topk_recs = np.stack(pref_data['topk_pred'].values)
    k = topk_recs.shape[1]

    mask_sh = dataset.item_feat[discrim_attr] == 1
    mask_lt = dataset.item_feat[discrim_attr] == 2

    if discrim_attr == 'visibility':
        metric = np.bincount(topk_recs.flatten(), minlength=dataset.item_num)

        metric_sh = metric[mask_sh].sum() / np.multiply(*topk_recs.shape)
        metric_lt = metric[mask_lt].sum() / np.multiply(*topk_recs.shape)
    elif discrim_attr == 'exposure':
        metric_sh = mask_sh[topk_recs].long()
        metric_lt = mask_lt[topk_recs].long()

        exposure_discount = np.log2(np.arange(1, k + 1) + 1)

        metric_sh = ((metric_sh / exposure_discount).sum(dim=1) / (1 / exposure_discount).sum()).mean()
        metric_lt = ((metric_lt / exposure_discount).sum(dim=1) / (1 / exposure_discount).sum()).mean()
    else:
        raise NotImplementedError(f'The fairness level `{fairness_level}` of the provider explanation metric is not supported')

    return abs(metric_sh / groups_distrib[0] - metric_lt / groups_distrib[1])


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
