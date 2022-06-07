import argparse
import logging
import pickle
from logging import getLogger

import yaml
import torch
import tqdm
import scipy
import numpy as np
import pandas as pd
import networkx as nx
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


def explain(config, model, test_data, epochs, topk=10, dist_type="damerau_levenshtein"):
    iter_data = (
        tqdm.tqdm(
            test_data,
            total=len(test_data),
            ncols=100,
            desc=set_color(f"Explaining   ", 'pink'),
        )
    )

    exps = {}
    for batch_idx, batched_data in enumerate(iter_data):
        user_id = batched_data[0].interaction[model.USER_ID][0]
        bge = BGExplainer(config, train_data.dataset, model, user_id, dist=dist_type)
        exp = bge.explain(batched_data, epochs, topk=topk)
        exps[user_id.item()] = exp
        if batch_idx > 30:
            break

    data = [exp for exp_list in exps for exp in exp_list]
    data = list(map(lambda x: [x[0].item(), *x[1:]], data))
    df = pd.DataFrame(data, columns=['uid', 'topk', 'cf_topk', 'cf_topk_pred', 'loss_total', 'loss_pred', 'loss_dist'])

    df.to_csv('exps_test_1_epoch_hops_all_users.csv', index=False)


def explain2(config, model, test_data, epochs, topk=10, dist_type="damerau_levenshtein"):
    graph = get_adj_matrix(config, dataset)
    max_cc_graph = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()
    cc_graphs = [graph.subgraph(x).copy() for x in nx.connected_components(graph)]
    for x in cc_graphs:
        print(x.number_of_nodes(), x.number_of_edges(), nx.diameter(x))
    diam = nx.diameter(max_cc_graph)

    iter_data = (
        tqdm.tqdm(
            test_data,
            total=len(test_data),
            ncols=100,
            desc=set_color(f"Explaining   ", 'pink'),
        )
    )

    exps = []
    for batch_idx, batched_data in enumerate(iter_data):
        user_id = batched_data[0].interaction[model.USER_ID][0]
        for n_hops in range(1, diam + 1):
            config['n_hops'] = n_hops

            try:
                bge = BGExplainer(config, dataset, model, user_id, dist=dist_type)
                exp = bge.explain(batched_data, 1, topk=topk)
                exps.append([n_hops, user_id.item(), len(set(exp[0][1]) & set(exp[0][3]))])
            except:
                pass

    df = pd.DataFrame(exps, columns=['n_hops', 'uid', 'common_items'])

    df.to_csv('exps_test_1_epoch_hops_all_users.csv', index=False)


def get_adj_matrix(config, dataset):
    USER_ID = config['USER_ID_FIELD']
    ITEM_ID = config['ITEM_ID_FIELD']
    n_users = dataset.num(USER_ID)
    n_items = dataset.num(ITEM_ID)
    inter_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
    num_all = n_users + n_items

    A = scipy.sparse.dok_matrix((num_all, num_all), dtype=np.float32)
    inter_M = inter_matrix
    inter_M_t = inter_matrix.transpose()
    data_dict = dict(zip(zip(inter_M.row, inter_M.col + n_users), [1] * inter_M.nnz))
    data_dict.update(dict(zip(zip(inter_M_t.row + n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
    A._update(data_dict)
    A = A.tocoo()
    return nx.Graph(A)


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

    explain2(config, model, test_data, config['cf_epochs'])
