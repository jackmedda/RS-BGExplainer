import numpy as np

import biga.utils as utils
import biga.evaluation as eval_utils


class LRScaler(object):

    def __int__(self, optimizer, n_batches):
        self.lr_scaling = np.linspace(0.1, 1, n_batches)
        self.optimizer = optimizer

        self.orig_lr = [pg['lr'] for pg in self.optimizer.param_groups]

    def update(self, batch_idx):
        for pg_idx, pg in enumerate(self.optimizer.param_groups):
            pg['lr'] = orig_lr[pg_idx] * self.lr_scaling[batch_idx]

    def restore(self):
        for pg_idx, pg in enumerate(self.optimizer.param_groups):
            pg['lr'] = self.orig_lr[pg_idx]


def get_scores(model, batched_data, tot_item_num, test_batch_size, item_tensor, pred=False):
    kws = {"pred": pred} if pred is not None else {}
    return utils.get_scores(model, batched_data, tot_item_num, test_batch_size, item_tensor, **kws)


def get_fair_metric_value(fair_metric,
                          *args,
                          **kwargs):
    func_map = {
        'DP_across_random_samples': _compute_explainer_DP_across_random_samples,
        'DP': _compute_explainer_DP
    }
    if fair_metric not in func_map:
        raise NotImplementedError(f'fair_metric `{fair_metric}` is not implemented.')

    return func_map[fair_metric](*args, **kwargs)


def _compute_explainer_DP_across_random_samples(pref_data,
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

    fair_metric_value, _ = eval_utils.compute_DP_across_random_samples(
        pref_data, sens_attr, 'Demo Group', dset_name, eval_metric, iterations=iterations, batch_size=batch_size
    )

    return fair_metric_value[:, -1].mean()


def _compute_explainer_DP(pref_data,
                          eval_metric,
                          *args,
                          **kwargs):
    gr_results = []
    for gr_mask in pref_data.groupby('Demo Group').groups.values():
        gr_results.append(pref_data.loc[gr_mask, eval_metric].mean())

    return eval_utils.compute_DP(*gr_results)
