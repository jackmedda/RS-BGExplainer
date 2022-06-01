import argparse
import logging
import pickle
from logging import getLogger

import yaml
import torch
import tqdm
from recbole.config import Config
from recbole.data import create_dataset, data_preparation, save_split_dataloaders, load_split_dataloaders
from recbole.utils import init_logger, get_model, get_trainer, init_seed, set_color

from utils.utils import damerau_levenshtein_distance
from explainers.explainer import BGExplainer


def load_data_and_model(model_file):
    r"""Load filtered dataset, split dataloaders and saved model.
    Args:
        model_file (str): The path of saved model file.
    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    checkpoint = torch.load(model_file)
    config = checkpoint['config']
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    # specifying tot_item_num to the number of unique items makes the dataloader for evaluation to be batched
    # on interactions of one user at a time
    config['eval_batch_size'] = dataset.item_num
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))

    return config, model, dataset, train_data, valid_data, test_data


def explain(model, test_data, epochs, topk=10, dist_type="damerau_levenshtein"):
    iter_data = (
        tqdm.tqdm(
            test_data,
            total=len(test_data),
            ncols=100,
            desc=set_color(f"Explaining   ", 'pink'),
        )
    )

    for batch_idx, batched_data in enumerate(iter_data):
        user_id = batched_data[0].interaction[model.USER_ID][0]
        bge = BGExplainer(config, dataset, model, user_id, dist=dist_type)
        print(bge.explain(batched_data, epochs, topk=topk))
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', default=None)
    parser.add_argument('--explainer_config_file', default='../config/gcmc_explainer.yaml')

    args = parser.parse_args()

    if args.model_file is None:
        raise FileNotFoundError("need to specify a saved file with `--model_file`")

    print(args)

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(args.model_file)
    with open(args.explainer_config_file, 'r', encoding='utf-8') as f:
        explain_config_dict = yaml.load(f.read(), Loader=config.yaml_loader)
    config.final_config_dict.update(explain_config_dict)

    explain(model, test_data, 300)
