import os
import re
import argparse
import inspect
import pickle
import json

import torch
import pandas as pd

import src.utils as utils
from src.explainers.explainer_dp_ndcg import DPBGExplainer


def load_already_done_exps_user_id(base_exps_file):
    """
    Only used for `individual` or 'group' explanations. It prevents the code from re-explaining already explained users.
    :param base_exps_file:
    :return:
    """
    files = [f for f in os.listdir(base_exps_file) if re.match(r'user_\d+', f) is not None]

    return [int(_id) for f in files for _id in f.split('_')[1].split('.')[0].split('#')]


def get_base_exps_filepath(config, config_id=-1):
    """
    return the filepath where explanations are saved
    :param config:
    :param config_id:
    :return:
    """
    epochs = config['cf_epochs']
    base_exps_file = os.path.join(script_path, 'dp_ndcg_explanations', config['dataset'])
    if config['explain_fairness']:
        fair_metadata = "_".join(config["sensitive_attributes"])
        fair_loss = 'FairDP'
        base_exps_file = os.path.join(base_exps_file, fair_loss, fair_metadata, f"epochs_{epochs}")
    else:
        base_exps_file = os.path.join(base_exps_file, 'pred_explain', f"epochs_{epochs}")

    if os.path.exists(base_exps_file):
        if config_id == -1:
            i = 1
            for path_c in sorted(os.listdir(base_exps_file), key=int):
                config_path = os.path.join(base_exps_file, path_c, 'config.pkl')
                if os.path.exists(config_path):
                    with open(config_path, 'rb') as f:
                        _c = pickle.load(f)
                    if config.final_config_dict == _c.final_config_dict:
                        break
                i += 1

            base_exps_file = os.path.join(base_exps_file, str(i))
        else:
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


def explain(config, model, test_data, base_exps_file, **kwargs):
    """
    Function that explains, that is generates perturbed graphs.
    :param config:
    :param model:
    :param test_data:
    :param base_exps_file:
    :param kwargs:
    :return:
    """
    epochs = config['cf_epochs']
    topk = config['cf_topk']

    if not os.path.exists(base_exps_file):
        os.makedirs(base_exps_file)

    user_data = test_data.user_df[config['USER_ID_FIELD']][torch.randperm(test_data.user_df[config['USER_ID_FIELD']].shape[0])]

    with open(os.path.join(base_exps_file, "config.pkl"), 'wb') as config_file:
        pickle.dump(config, config_file)

    with open(os.path.join(base_exps_filepath, "config.json"), 'w') as config_json:
        json.dump(config.final_config_dict, config_json, indent=4, default=lambda x: str(x))

    bge = DPBGExplainer(config, train_data.dataset, rec_data, model, user_data, dist=config['cf_dist'], **kwargs)
    exp, _ = bge.explain((user_data, test_data), epochs, topk=topk)
    del bge

    exps_file_user = os.path.join(base_exps_file, f"all_users.pkl")

    with open(exps_file_user, 'wb') as f:
        pickle.dump(exp, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', required=True)
    parser.add_argument('--explainer_config_file', default=os.path.join("config", "gcmc_explainer.yaml"))
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--load_config_id', default=-1)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    script_path = os.path.abspath(os.path.dirname(inspect.getsourcefile(lambda: 0)))

    if args.model_file is None:
        raise FileNotFoundError("need to specify a saved file with `--model_file`")

    print(args)

    # load trained model, config, dataset
    config, model, dataset, train_data, valid_data, test_data = utils.load_data_and_model(args.model_file,
                                                                                          args.explainer_config_file)

    if config['exp_rec_data'] is not None:
        rec_data = locals()[f"{config['exp_rec_data']}_data"]
    else:
        rec_data = valid_data

    base_exps_filepath = get_base_exps_filepath(config, config_id=args.load_config_id)

    if not args.load:
        kwargs = dict(
            verbose=args.verbose
        )

        explain(
            config,
            model,
            rec_data,
            base_exps_filepath,
            **kwargs
        )
    else:
        with open(os.path.join(base_exps_filepath, "config.pkl"), 'rb') as config_file:
            config = pickle.load(config_file)

    # exps_data = utils.load_exps_file(base_exps_filepath)
    # save_exps_df(base_exps_filepath, exps_data)
