import os
import re
import shutil
import pickle
import logging
import inspect
import argparse

import wandb
import torch
import pandas as pd

from recbole.data.dataloader import FullSortEvalDataLoader

import src.utils as utils
from src.explainers.explainer_dp_ndcg import DPBGExplainer


script_path = os.path.abspath(os.path.dirname(inspect.getsourcefile(lambda: 0)))


def load_already_done_exps_user_id(base_exps_file):
    """
    Only used for `individual` or 'group' explanations. It prevents the code from re-explaining already explained users.
    :param base_exps_file:
    :return:
    """
    files = [f for f in os.listdir(base_exps_file) if re.match(r'user_\d+', f) is not None]

    return [int(_id) for f in files for _id in f.split('_')[1].split('.')[0].split('#')]


def get_base_exps_filepath(config, config_id=-1, model_name=None, model_file=""):
    """
    return the filepath where explanations are saved
    :param config:
    :param config_id:
    :param model_name:
    :param model_file:
    :return:
    """
    epochs = config['cf_epochs']
    model_name = model_name or config['model']
    base_exps_file = os.path.join(script_path, 'dp_ndcg_explanations', config['dataset'], model_name)
    if config['explain_fairness']:
        fair_metadata = config["sensitive_attribute"]
        fair_loss = 'FairDP'
        base_exps_file = os.path.join(base_exps_file, fair_loss, fair_metadata, f"epochs_{epochs}")
    else:
        base_exps_file = os.path.join(base_exps_file, 'pred_explain', f"epochs_{epochs}")

    if os.path.exists(base_exps_file):
        if config_id == -1:
            paths_c_ids = sorted(os.listdir(base_exps_file), key=int)
            for path_c in paths_c_ids:
                config_path = os.path.join(base_exps_file, path_c, 'config.pkl')
                if os.path.exists(config_path):
                    with open(config_path, 'rb') as f:
                        _c = pickle.load(f)
                    if config.final_config_dict == _c.final_config_dict:
                        if model_file != "" and 'perturbed' in model_file:
                            check_perturb = input("The explanations of the perturbed graph could overwrite the "
                                                  "explanations from which the perturbed graph was generated. Type "
                                                  "y/yes to confirm this outcome. Other inputs will assign a new id: ")
                            if check_perturb.lower() != "y" and check_perturb.lower() != "yes":
                                continue
                        return os.path.join(base_exps_file, str(path_c))

            config_id = 1 if len(paths_c_ids) == 0 else str(int(max(paths_c_ids, key=int)) + 1)

        base_exps_file = os.path.join(base_exps_file, str(config_id))
    else:
        base_exps_file = os.path.join(base_exps_file, "1")

    return base_exps_file


def save_exps_df(base_exps_file, exps):
    """
    Saves the pandas dataframe representation of explanations
    :param base_exps_file:
    :param exps:
    :return:
    """
    data = [exp[:-3] + exp[-2:] for exp_list in exps.values() for exp in exp_list]
    data = list(map(lambda x: [x[0], *x[1:]], data))
    df = pd.DataFrame(
        data,
        columns=utils.EXPS_COLUMNS[:-3] + utils.EXPS_COLUMNS[-2:]
    )

    out_path = base_exps_file.split(os.sep)
    df.to_csv(
        f'{"_".join(out_path[out_path.index("explanations"):])}.csv',
        index=False
    )


def explain(config, model, _train_dataset, _rec_data, _test_data, base_exps_file, **kwargs):
    """
    Function that explains, that is generates perturbed graphs.
    :param config:
    :param model:
    :param _train_dataset:
    :param _rec_data:
    :param _test_data:
    :param base_exps_file:
    :param kwargs:
    :return:
    """
    epochs = config['cf_epochs']
    topk = config['cf_topk']
    explainer_config_file = kwargs.get("explainer_config_file", None)

    if not os.path.exists(base_exps_file):
        os.makedirs(base_exps_file)

    user_data = _rec_data.user_df[_rec_data.uid_field][torch.randperm(_rec_data.user_df.length)]

    with open(os.path.join(base_exps_file, "config.pkl"), 'wb') as config_file:
        pickle.dump(config, config_file)

    if explainer_config_file is not None:
        try:
            shutil.copy(explainer_config_file, os.path.join(base_exps_file, "config.yaml"))
        except shutil.SameFileError as e:
            print(f"Overwriting config {os.path.basename(base_exps_file)}")

    utils.wandb_init(
        config,
        name="Explanation",
        job_type="train",
        group=f"{model.__class__.__name__}_{config['dataset']}_{config['sensitive_attribute'].title()}_epochs{config['cf_epochs']}_exp={os.path.basename(base_exps_file)}",
        mode="disabled"
    )
    wandb.config.update({"exp": os.path.basename(base_exps_file)})

    bge = DPBGExplainer(config, _train_dataset, _rec_data, model, dist=config['cf_dist'], **kwargs)
    exp = bge.explain(user_data, _test_data, epochs, topk=topk)
    del bge

    exps_file_user = os.path.join(base_exps_file, f"all_users.pkl")

    with open(exps_file_user, 'wb') as f:
        pickle.dump(exp, f)
    logging.getLogger().info(f"Saved explanations at path {exps_file_user}")


def execute_explanation(model_file,
                        explainer_config_file=os.path.join("config", "explainer.yaml"),
                        load=False,
                        config_id=-1,
                        verbose=False):
    # load trained model, config, dataset
    config, model, dataset, train_data, valid_data, test_data = utils.load_data_and_model(model_file,
                                                                                          explainer_config_file)

    if config['exp_rec_data'] is not None:
        if config['exp_rec_data'] != 'train+valid':
            rec_data = locals()[f"{config['exp_rec_data']}_data"]
            if config['exp_rec_data'] == 'train':
                rec_data = FullSortEvalDataLoader(config, train_data.dataset, train_data.sampler)
        else:
            rec_data = valid_data
    else:
        rec_data = valid_data

    base_exps_filepath = get_base_exps_filepath(config, config_id=config_id, model_name=model.__class__.__name__, model_file=model_file)

    if not load:
        kwargs = dict(
            verbose=verbose,
            explainer_config_file=explainer_config_file
        )

        explain(
            config,
            model,
            train_data.dataset,
            rec_data,
            test_data,
            base_exps_filepath,
            **kwargs
        )
    else:
        with open(os.path.join(base_exps_filepath, "config.pkl"), 'rb') as config_file:
            config = pickle.load(config_file)

    # exps_data = utils.load_exps_file(base_exps_filepath)
    # save_exps_df(base_exps_filepath, exps_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', required=True)
    parser.add_argument('--explainer_config_file', default=os.path.join("config", "explainer.yaml"))
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--config_id', default=-1)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    print(args)

    execute_explanation(args.model_file,
                        explainer_config_file=args.explainer_config_file,
                        load=args.load,
                        config_id=args.config_id,
                        verbose=args.verbose)
