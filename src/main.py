import os
import os.path as osp
import inspect
import argparse
import logging
import shutil

from recbole.config import Config
from recbole.data import create_dataset, data_preparation, save_split_dataloaders, load_split_dataloaders
from recbole.utils import init_logger, get_model, get_trainer, init_seed, set_color

from models import GCMC, GCMCPerturbated


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

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config['seed'], config['reproducibility'])
    model_class = GCMC if model == "GCMC" else GCMCPerturbated
    model = model_class(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='GCMC')
    parser.add_argument('--dataset', default='ml-100k')
    parser.add_argument('--config_file_list', nargs='+', default=None)
    parser.add_argument('--config_dict', default=None)
    parser.add_argument('--saved', action='store_true')

    args = parser.parse_args()
    print(args)

    # source_path = osp.abspath(osp.dirname(inspect.getsourcefile(lambda: 0)))
    # gcmc_perturb_path = osp.join(source_path, 'models', 'gcmcperturbated.py')
    # models_recbole_path = osp.join(
    #     source_path, os.pardir, 'venv', 'lib', 'python3.8', 'site-packages', 'recbole', 'model'
    # )
    # general_recs = osp.join(models_recbole_path, 'general_recommender')
    # shutil.copy(gcmc_perturb_path, osp.join(gcmc_perturb_path, general_recs))
    #
    # add_import = False
    # with open(osp.join(general_recs, '__init__.py'), 'r') as f:
    #     for line in f:
    #         pass
    #     last_line = line
    #     add_import = 'gcmcperturbated' not in last_line
    #
    # if add_import:
    #     with open(osp.join(general_recs, '__init__.py'), 'a') as f:
    #         f.write('from recbole.model.general_recommender.gcmcperturbated import GCMCPerturbated')
    #
    # yamls_path = osp.join(models_recbole_path, os.pardir, 'properties', 'model')
    # if not osp.exists(osp.join(yamls_path, 'GCMCPerturbated.yaml')):
    #     shutil.copyfile(osp.join(yamls_path, 'GCMC.yaml'), osp.join(yamls_path, 'GCMCPerturbated.yaml'))
    #
    # if args.config_file_list is None:
    #     args.config_file_list = [osp.abspath(
    #         osp.join(osp.dirname(inspect.getsourcefile(lambda: 0)), os.pardir, 'config', 'gcmc.yaml')
    #     )]

    run_recbole(args.model, args.dataset, args.config_file_list, args.config_dict)
