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
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_model, get_trainer, init_seed, set_color

from src.explain_dp_ndcg import execute_explanation
from src.plot_orig_perturb import prepare_data, graph_statistics
from src.utils import utils


def training(_model, _dataset, _config, saved=True, _yaml_config=None):
    logger = logging.getLogger()
    # dataset splitting
    train_data, valid_data, test_data = data_preparation(_config, _dataset)

    # model loading and initialization
    init_seed(_config['seed'], _config['reproducibility'])
    _model = get_model(_model)(_config, train_data.dataset).to(_config['device'])
    logger.info(_model)

    # trainer loading and initialization
    trainer = get_trainer(_config['MODEL_TYPE'], _config['model'])(_config, _model)

    split_saved_file = os.path.basename(trainer.saved_model_file).split('-')
    trainer.saved_model_file = os.path.join(
        os.path.dirname(trainer.saved_model_file),
        '-'.join(split_saved_file[:1] + [_dataset.dataset_name.upper()] + split_saved_file[1:])
    )

    if args.use_perturbed_graph:
        sens_attr, c_id = args.load_config_attr_id
        perturb_str = f"perturbed_({sens_attr}_epochs_{_yaml_config['cf_epochs']}_cid_{c_id}_best_exp_{args.best_exp})_"
        trainer.saved_model_file = os.path.join(
            os.path.dirname(trainer.saved_model_file),
            perturb_str + os.path.basename(trainer.saved_model_file)
        )

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


def main(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
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
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = logging.getLogger()
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    if args.run == 'evaluate_perturbed' or args.run == 'graph_stats':
        orig_config, orig_model, orig_dataset, orig_train_data, orig_valid_data, orig_test_data = \
            utils.load_data_and_model(args.original_model_file, args.explainer_config_file)

    if args.use_perturbed_graph:
        if 'LRS' not in config['eval_args']['split']:
            raise ValueError("Perturbed graph can be used only when splits are already loaded.")

        splits = ['train', 'validation', 'test']
        feats = ['inter', 'item']
        data_path = config['data_path']

        # Check outside of the try block
        for spl in splits:
            spl_file = os.path.join(data_path, f"{config['dataset']}.{spl}")
            if os.path.isfile(spl_file):
                if os.path.isfile(spl_file + '.temp'):
                    raise FileExistsError("Only one training with augmented graph per dataset can be performed")

        try:
            script_path = os.path.abspath(os.path.dirname(inspect.getsourcefile(lambda: 0)))
            script_path = os.path.join(script_path, 'src') if 'src' not in script_path else script_path

            sens_attr, c_id = args.load_config_attr_id
            exp_path = os.path.join(script_path, 'dp_ndcg_explanations', dataset.dataset_name, model,
                                    'FairDP', sens_attr, f"epochs_{config['epochs']}", c_id)
            with open(os.path.join(exp_path, 'all_users.pkl'), 'rb') as exp_file:
                exps = pickle.load(exp_file)
            logger.info(f"Original Fair Loss: {exps[0][-1]}")

            with open(os.path.join(exp_path, 'config.yaml'), 'r') as yaml_config_file:
                yaml_config = yaml.load(yaml_config_file.read(), Loader=config.yaml_loader)

            if yaml_config['exp_rec_data'] != 'valid':
                logger.warning('Performing Graph Augmentation on Explanations NOT produced on Validation Data.')

            best_exp = None
            if args.best_exp[0] == "fairest":
                best_exp = exps[np.argmin([exp[9] for exp in exps])]
            elif args.best_exp[0] == "fairest_before_exp":
                best_exp = exps[np.argmin([exp[9] for exp in exps[:int(args.best_exp[1])]])]
            elif args.best_exp[0] == "fixed_exp":
                try:
                    best_exp = [e for e in exps if e[11] == int(args.best_exp[1])][0]
                except IndexError as e:
                    logger.info(f"No explanation exists for epoch {args.best_exp[1]}")
            edges = best_exp[10]
            mapped_edges = []

            for i, field in enumerate([dataset.uid_field, dataset.iid_field]):
                mapped_edges.append([])
                for val in edges[i]:
                    idx_val = val
                    if field == dataset.iid_field:
                        idx_val = val - dataset.num(dataset.uid_field)

                    mapped_edges[-1].append(int(dataset.field2id_token[field][idx_val]))
            edges = np.stack(mapped_edges)

            check_new_edges = False  # if True must check if new edges are contained in test and/or validation
            for spl in splits:
                spl_file = os.path.join(data_path, f"{config['dataset']}.{spl}")
                if os.path.isfile(spl_file):
                    if os.path.isfile(spl_file + '.temp'):
                        raise FileExistsError("Only one training with augmented graph per dataset can be performed")
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

            if check_new_edges:
                uid_field = config['USER_ID_FIELD'] or 'user_id'
                iid_field = config['ITEM_ID_FIELD'] or 'item_id'
                for feat in feats:
                    feat_file = os.path.join(data_path, f"{config['dataset']}.{feat}")
                    if os.path.isfile(feat_file):
                        if os.path.isfile(feat_file + '.temp'):
                            raise FileExistsError("Only one training with augmented graph per dataset can be performed")
                        shutil.copyfile(feat_file, feat_file + '.temp')

                    feat_df = pd.read_csv(feat_file, sep='\t')
                    if feat == 'inter':
                        feat_df_cols = [uid_field + ':token', iid_field + ':token']

                        feat_data = set(map(tuple, feat_df[feat_df_cols].itertuples(index=False)))
                        edges_data = set(map(tuple, edges.T))
                        new_feat_edges = np.array(list(edges_data - feat_data))
                    else:
                        feat_df_cols = [iid_field + ':token']

                        new_feat_edges = np.setdiff1d(edges[1], feat_df[feat_df_cols[0]].values)[:, None]

                    if new_feat_edges.shape[0] > 0:
                        order_feat_cols = np.concatenate((feat_df_cols, np.setdiff1d(feat_df.columns, feat_df_cols)))
                        none_filling_data = np.full((new_feat_edges.shape[0], feat_df.columns.shape[0] - len(feat_df_cols)), None)

                        new_feat_rows = np.concatenate([new_feat_edges, none_filling_data], axis=1)
                        feat_df = pd.concat((feat_df, pd.DataFrame(new_feat_rows, columns=order_feat_cols)))

                        feat_df.to_csv(feat_file, index=None, sep='\t')

            dataset = create_dataset(config)
            logger.info(dataset)

            if args.run == 'train':
                runner(model, dataset, config, saved=saved, _yaml_config=yaml_config)
            elif args.run == 'explain':
                runner(*explain_args)
            elif args.run == 'evaluate_perturbed':
                logger.info("EVALUATE PERTURBED MODEL")
                _, pert_model, pert_dataset, _, _, _ = utils.load_data_and_model(args.model_file, args.explainer_config_file)
                runner(
                    orig_config,
                    orig_model,
                    pert_model,
                    orig_dataset,
                    pert_dataset,
                    orig_train_data,
                    orig_test_data,
                    topk=args.topk,
                    perturbed_model_file=os.path.splitext(os.path.basename(args.model_file))[0]
                )
            elif args.run == 'graph_stats':
                pert_config, _, _, pert_train_data, _, _ = utils.load_data_and_model(args.model_file, args.explainer_config_file)
                runner(
                    pert_config,
                    orig_train_data,
                    orig_valid_data,
                    orig_test_data,
                    pert_train_data,
                    args.original_model_file,
                    sens_attr,
                    c_id,
                    *args.best_exp
                )
        finally:
            # Restore train validation test splits
            for spl in splits:
                spl_file = os.path.join(data_path, f"{config['dataset']}.{spl}")
                if os.path.isfile(spl_file + '.temp'):
                    if os.path.isfile(spl_file):
                        os.remove(spl_file)
                    shutil.move(spl_file + '.temp', spl_file)
            for feat in feats:
                feat_file = os.path.join(data_path, f"{config['dataset']}.{feat}")
                if os.path.isfile(feat_file + '.temp'):
                    if os.path.isfile(feat_file):
                        os.remove(feat_file)
                    shutil.move(feat_file + '.temp', feat_file)
    else:
        dataset = create_dataset(config)
        logger.info(dataset)

        if args.run == 'train':
            runner(model, dataset, config, saved=saved)
        elif args.run == 'explain':
            runner(*explain_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    train_group = parser.add_argument_group(
        "train",
        "All the arguments related to training with classic and augmented data"
    )
    explain_group = parser.add_argument_group(
        "explain",
        "All the arguments related to create explanations"
    )
    evaluate_perturbed_group = parser.add_argument_group(
        "evaluate_perturbed",
        "All the arguments related to evaluate the original and the perturbed model"
    )

    train_group.add_argument('--run', default='train', choices=['train', 'explain', 'evaluate_perturbed', 'graph_stats'], required=True)
    train_group.add_argument('--model', default='GCMC')
    train_group.add_argument('--dataset', default='ml-100k')
    train_group.add_argument('--config_file_list', nargs='+', default=None)
    train_group.add_argument('--config_dict', default=None)
    train_group.add_argument('--saved', action='store_true')
    train_group.add_argument('--use_perturbed_graph', action='store_true')
    train_group.add_argument('--best_exp', nargs=2, help="one of ['fairest', 'fairest_before_exp', 'fixed_exp'] with"
                                                         " the chosen exp number for the last two types")
    train_group.add_argument('--load_config_attr_id', type=str, nargs=2, default=None)  # ex. ("gender", "2")
    explain_group.add_argument('--model_file')
    explain_group.add_argument('--explainer_config_file', default=os.path.join("config", "explainer.yaml"))
    explain_group.add_argument('--load', action='store_true')
    explain_group.add_argument('--explain_config_id', default=-1)
    explain_group.add_argument('--verbose', action='store_true')
    evaluate_perturbed_group.add_argument('--original_model_file')
    evaluate_perturbed_group.add_argument('--topk', default=10)

    args = parser.parse_args()
    print(args)

    explain_args = [args.model_file, args.explainer_config_file, args.load, args.explain_config_id, args.verbose]

    if args.run == 'train':
        runner = training
    elif args.run == 'explain':
        runner = execute_explanation
    elif args.run == 'evaluate_perturbed':
        runner = prepare_data
    elif args.run == 'graph_stats':  # only works with --use_perturbed_graph
        runner = graph_statistics
    else:
        raise NotImplementedError(f"The run `{args.run}` is not supported.")

    main(args.model, args.dataset, args.config_file_list, args.config_dict)
