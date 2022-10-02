import os
import json
import shutil
import pickle
import inspect
import argparse
import logging

import numpy as np
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_trainer, init_seed, set_color

from src.models import GCMC, GCMCPerturbated
from src.utils import utils


def run_recbole(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset
    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    def runner(_model, _dataset, _config, _json_config=None):
        # dataset splitting
        train_data, valid_data, test_data = data_preparation(_config, _dataset)

        # model loading and initialization
        init_seed(_config['seed'], _config['reproducibility'])
        model_class = GCMC if _model == "GCMC" else GCMCPerturbated
        _model = model_class(_config, train_data.dataset).to(_config['device'])
        logger.info(_model)

        # trainer loading and initialization
        trainer = get_trainer(_config['MODEL_TYPE'], _config['model'])(_config, _model)

        split_saved_file = trainer.saved_model_file.split('-')
        trainer.saved_model_file = '-'.join(split_saved_file[:1] + [_dataset.dataset_name.upper()] + split_saved_file[1:])

        if args.use_perturbed_graph:
            sens_attr, c_id = args.load_config_attr_id
            perturb_str = f"perturbed({sens_attr}_epochs_{_json_config['cf_epochs']}_cid_{c_id})_"
            trainer.saved_model_file = perturb_str + trainer.saved_model_file

        # model training
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, saved=saved, show_progress=_config['show_progress']
        )

        # model evaluation
        test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=_config['show_progress'])

        logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
        logger.info(set_color('test result', 'yellow') + f': {test_result}')

        return {
            'best_valid_score': best_valid_score,
            'valid_score_bigger': _config['valid_metric_bigger'],
            'best_valid_result': best_valid_result,
            'test_result': test_result
        }

    # configurations initialization
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = logging.getLogger()
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)
    if args.use_perturbed_graph:
        if 'LRS' not in config['eval_args']['split']:
            raise ValueError("Perturbed graph can be used only when splits are already loaded.")

        splits = ['train', 'validation', 'test']
        data_path = config['data_path']
        try:
            script_path = os.path.abspath(os.path.dirname(inspect.getsourcefile(lambda: 0)))
            script_path = os.path.join(script_path, 'src') if 'src' not in script_path else script_path

            sens_attr, c_id = args.load_config_attr_id
            exp_path = os.path.join(script_path, 'dp_ndcg_explanations', dataset.dataset_name,
                                    'FairDP', sens_attr, f"epochs_{config['epochs']}", c_id)
            with open(os.path.join(exp_path, 'all_users.pkl'), 'rb') as exp_file:
                exps = pickle.load(exp_file)
            logger.info(f"Original Fair Loss: {exps[0][-1]}")

            with open(os.path.join(exp_path, 'config.json'), 'r') as json_config_file:
                json_config = json.load(json_config_file)

            if json_config['exp_rec_data'] != 'valid':
                logger.warning('Performing Graph Augmentation on Explanations NOT produced on Validation Data.')

            fairest_exp = exps[np.argmin([exp[7] for exp in exps])]
            edges = fairest_exp[8]

            check_new_edges = False  # if True must check if new edges are contained in test and/or validation
            for spl in splits:
                spl_file = os.path.join(data_path, f"{config['dataset']}.{spl}")
                if os.path.isfile(spl_file):
                    shutil.copyfile(spl_file, spl_file + '.temp')

                if spl == 'train':
                    with open(spl_file, 'rb') as split_data_file:
                        split_data = pickle.load(split_data_file)
                    new_split_data = utils.unique_cat_recbole_interaction(split_data, edges)
                    if len(new_split_data[dataset.uid_field]) > len(split_data[dataset.uid_field]):
                        check_new_edges = True
                    with open(spl_file, 'wb') as split_data_file:
                        pickle.dump(new_split_data, split_data_file)
                elif check_new_edges:
                    with open(spl_file, 'rb') as split_data_file:
                        split_data = pickle.load(split_data_file)
                    new_split_data, unique, counts = utils.unique_cat_recbole_interaction(split_data, edges, return_unique_counts=True)
                    if 2 in counts:
                        common_inter = unique[:, counts == 2]
                        new_split_data = utils.unique_cat_recbole_interaction(split_data, common_inter)
                        with open(spl_file, 'wb') as split_data_file:
                            pickle.dump(new_split_data, split_data_file)

            runner(model, dataset, config, json_config)
        finally:
            # Restore train validation test splits
            for spl in splits:
                spl_file = os.path.join(data_path, f"{config['dataset']}.{spl}")
                if os.path.isfile(spl_file + '.temp'):
                    if os.path.isfile(spl_file):
                        os.remove(spl_file)
                    shutil.move(spl_file + '.temp', spl_file)
    else:
        runner(model, dataset, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='GCMC')
    parser.add_argument('--dataset', default='ml-100k')
    parser.add_argument('--config_file_list', nargs='+', default=None)
    parser.add_argument('--config_dict', default=None)
    parser.add_argument('--saved', action='store_true')
    parser.add_argument('--use_perturbed_graph', action='store_true')
    parser.add_argument('--load_config_attr_id', type=str, nargs=2, default=None)  # ex. ("gender", "2")

    args = parser.parse_args()
    print(args)

    run_recbole(args.model, args.dataset, args.config_file_list, args.config_dict)
