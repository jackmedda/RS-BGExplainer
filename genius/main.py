import os
import yaml
import shutil
import pickle
import inspect
import argparse
import logging

import numpy as np
import pandas as pd
from recbole.config import Config
from recbole.trainer import HyperTuning
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_model, get_trainer, init_seed, set_color, get_local_time

from genius.explain import execute_explanation
import genius.utils as utils


def training(_config, saved=True, model_file=None, hyper=False):
    logger = logging.getLogger() if hyper else None

    if model_file is not None:
        _config, _model, _dataset, train_data, valid_data, test_data = utils.load_data_and_model(model_file)
    else:
        # dataset filtering
        _dataset = create_dataset(_config)

        # dataset splitting
        train_data, valid_data, test_data = data_preparation(_config, _dataset)

        # model loading and initialization
        _model = get_model(_config['model'])(_config, train_data.dataset).to(_config['device'])

        if not hyper:
            logger.info(_config)
            logger.info(_dataset)
            logger.info(_model)

    # trainer loading and initialization
    trainer = get_trainer(_config['MODEL_TYPE'], _config['model'])(_config, _model)
    if model_file is not None:
        trainer.resume_checkpoint(model_file)
    else:
        split_saved_file = os.path.basename(trainer.saved_model_file).split('-')
        trainer.saved_model_file = os.path.join(
            os.path.dirname(trainer.saved_model_file),
            '-'.join(split_saved_file[:1] + [_dataset.dataset_name.upper()] + split_saved_file[1:])
        )

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        saved=saved,
        show_progress=_config['show_progress'] and not hyper,
        verbose=not hyper
    )

    # model evaluation
    test_result = trainer.evaluate(
        test_data,
        load_best_model=saved,
        show_progress=_config['show_progress'] and not hyper
    )

    if not hyper:
        logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
        logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': _config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def recbole_hyper(base_config, params_file, config_file_list, saved=True):
    def objective_function(c_dict, c_file_list):
        config = Config(
            model=base_config['model'],
            dataset=base_config['dataset'],
            config_file_list=c_file_list,
            config_dict=c_dict
        )
        config['data_path'] = os.path.join(base_config.file_config_dict['data_path'], base_config.dataset)
        init_seed(base_config['seed'], config['reproducibility'])
        logging.basicConfig(level=logging.ERROR)

        return training(config, saved=False, hyper=True)

    hp = HyperTuning(
        objective_function, algo='exhaustive', params_file=params_file, fixed_config_file_list=config_file_list
    )
    hp.run()

    output_path = os.path.join(base_config['checkpoint_dir'], 'hyper', base_config['dataset'], base_config['model'])
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"{get_local_time()}.txt")

    hp.export_result(output_file=output_file)
    print('best params: ', hp.best_params)
    print('best result: ')
    print(hp.params2result[hp.params2str(hp.best_params)])

    with open(output_file, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(
            'Best Params and Results\n' +
            str(hp.best_params).rstrip('\r\n') + '\n' +
            str(hp.params2result[hp.params2str(hp.best_params)]) + '\n\n' + content
        )


def main(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True, seed=None, hyper_params_file=None):
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
    config['data_path'] = os.path.join(config.file_config_dict['data_path'], config.dataset)
    seed = seed or config['seed']
    init_seed(seed, config['reproducibility'])
    # logger initialization
    init_logger(config)

    if args.run == 'train':
        runner(config, saved=saved, model_file=args.model_file)
    elif args.run == 'explain':
        runner(args.model_file, *explain_args)
    elif args.run == 'recbole_hyper':
        config['seed'] = seed
        runner(config, hyper_params_file, config_file_list, saved=saved)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # train_group = parser.add_argument_group(
    #     "train",
    #     "All the arguments related to training with classic and augmented data"
    # )
    explain_group = parser.add_argument_group(
        "explain",
        "All the arguments related to create explanations"
    )
    recbole_hyper_group = parser.add_argument_group(
        "recole_hyper",
        "All the arguments related to run the hyperparameter optimization on the recbole models for training"
    )

    parser.add_argument('--run', default='train', choices=['train', 'explain', 'recbole_hyper'], required=True)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--model', default='GCMC')
    parser.add_argument('--dataset', default='ml-100k')
    parser.add_argument('--config_file_list', nargs='+', default=None)
    parser.add_argument('--config_dict', default=None)
    parser.add_argument('--model_file', default=None)
    explain_group.add_argument('--base_explainer_config_file', default=os.path.join("config", "base_explainer.yaml"))
    explain_group.add_argument('--explainer_config_file', default=os.path.join("config", "explainer.yaml"))
    # explain_group.add_argument('--load', action='store_true')
    explain_group.add_argument('--explain_config_id', default=-1)
    explain_group.add_argument('--verbose', action='store_true')
    explain_group.add_argument('--wandb_online', action='store_true')
    explain_group.add_argument('--hyper_optimize', action='store_true')
    explain_group.add_argument('--overwrite', action='store_true')
    recbole_hyper_group.add_argument('--params_file', default=None)

    args, unk_args = parser.parse_known_args()
    print(args)

    unk_args[::2] = map(lambda s: s.replace('-', ''), unk_args[::2])
    unk_args = dict(zip(unk_args[::2], unk_args[1::2]))
    print("Unknown args", unk_args)

    if args.hyper_optimize and not args.verbose:
        from tqdm import tqdm
        from functools import partialmethod

        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    args.wandb_online = {False: "offline", True: "online"}[args.wandb_online]
    explain_args = [
        args.base_explainer_config_file,
        args.explainer_config_file,
        args.explain_config_id,
        args.verbose,
        args.wandb_online,
        unk_args,
        args.hyper_optimize,
        args.overwrite
    ]

    if args.run == 'train':
        runner = training
    elif args.run == 'explain':
        runner = execute_explanation
    elif args.run == 'recbole_hyper':
        runner = recbole_hyper
    else:
        raise NotImplementedError(f"The run `{args.run}` is not supported.")

    main(
        args.model,
        args.dataset,
        args.config_file_list,
        args.config_dict,
        seed=args.seed,
        hyper_params_file=args.params_file
    )