#!/usr/bin/env python
# coding:utf-8

# CODE BY: https://github.com/Alibaba-NLP/HiAGM
# Rights reserved to original authors. Only minor adaptations have been made.

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.models.HiAGM.data_modules.data_loader import data_loaders
from src.models.HiAGM.data_modules.vocab import Vocab
from src.models.HiAGM.helper import logger as logger
from src.models.HiAGM.helper.configure import Configure
from src.models.HiAGM.helper.utils import load_checkpoint, save_checkpoint
from src.models.HiAGM.models.model import HiAGM
from src.models.HiAGM.prepare_our_data import prepare_dataset, write_split
from src.models.HiAGM.train_modules.criterions import ClassificationLoss
from src.models.HiAGM.train_modules.trainer import Trainer
from src.utils.metrics import compute_metrics, compute_hierarchical_metrics

os.linesep = '\n'


def _predict(model: HiAGM, loader):
    y_pred, y_true = list(), list()
    model.eval()
    infer_time_batch = .0
    for batch_i, batch in enumerate(tqdm(loader)):
        infer_time_start: int = time.perf_counter_ns()
        logits = model(batch)
        infer_time_end: int = time.perf_counter_ns()
        infer_time_batch += infer_time_end - infer_time_start

        pred = torch.sigmoid(logits).detach().cpu().numpy()

        # print(tuple(pred.size()))
        true = batch['label'].detach().cpu().numpy()
        # pred = pred.topk(k=2, largest=True, dim=1)[1].detach().cpu().numpy().tolist()
        # pred = ["_".join([str(i) for i in sorted(ls)]) for ls in pred]
        #
        # true = batch["label_list"]
        # true = ["_".join([str(i) for i in sorted(ls)]) for ls in true]

        y_pred.extend(pred)
        y_true.extend(true)
        if batch_i == 10:
            break
    return np.array(y_pred), np.array(y_true), infer_time_batch / len(loader)


def set_optimizer(config, model):
    """
    :param config: helper.configure, Configure Object
    :param model: computational graph
    :return: torch.optim
    """
    params = model.optimize_params_dict()
    if config.train.optimizer.type == 'Adam':
        return torch.optim.Adam(lr=config.train.optimizer.learning_rate,
                                params=params)
    else:
        raise TypeError("Recommend the Adam optimizer")


def train(config, config_numer: int):
    """
    :param config: helper.configure, Configure Object
    :param config_numer: number of the configuration (for validation)
    """
    results = list()
    validation: bool = config["data"]["val_file"] is not None
    split_num: int = 0
    for train_split, test_split, val_split in prepare_dataset(config):
        split_num += 1
        fold_tot = config['stratifiedCV'] * config['n_repeats']
        print(f"Fold {split_num}/{fold_tot} ({config['stratifiedCV']} folds * {config['n_repeats']} repeats)")
        # loading corpus and generate vocabulary
        corpus_vocab: Vocab = write_split(config, train_split, test_split, val_split)
        # corpus_vocab = Vocab(config,
        #                      min_freq=5,
        #                      max_size=50000)

        # get data
        train_loader, dev_loader, test_loader = data_loaders(config, corpus_vocab)

        # build up model
        hiagm = HiAGM(config, corpus_vocab, model_type=config.model.type, model_mode='TRAIN')
        hiagm.to(config.train.device_setting.device)
        # define training objective & optimizer
        criterion = ClassificationLoss(os.path.join(config.data.data_dir, config.data.hierarchy),
                                       corpus_vocab.v2i['label'],
                                       recursive_penalty=config.train.loss.recursive_regularization.penalty,
                                       recursive_constraint=config.train.loss.recursive_regularization.flag)
        optimize = set_optimizer(config, hiagm)

        # get epoch trainer
        trainer = Trainer(model=hiagm,
                          criterion=criterion,
                          optimizer=optimize,
                          vocab=corpus_vocab,
                          config=config)

        # set origin log
        best_epoch = [-1, -1]
        best_performance = [0.0, 0.0]
        model_checkpoint = f"{config.train.checkpoint.dir}_split_{split_num}_config_{config_numer}"
        model_name = config.model.type
        wait = 0
        if not os.path.isdir(model_checkpoint):
            os.mkdir(model_checkpoint)
            config.train.start_epoch = 0
        else:
            # loading previous checkpoint
            dir_list = os.listdir(model_checkpoint)
            dir_list.sort(key=lambda fn: os.path.getatime(os.path.join(model_checkpoint, fn)))
            latest_model_file = ''
            for model_file in dir_list[::-1]:
                if model_file.startswith('best'):
                    continue
                else:
                    latest_model_file = model_file
                    break
            if os.path.isfile(os.path.join(model_checkpoint, latest_model_file)):
                logger.info('Loading Previous Checkpoint...')
                logger.info('Loading from {}'.format(os.path.join(model_checkpoint, latest_model_file)))
                best_performance, config = load_checkpoint(model_file=os.path.join(model_checkpoint, latest_model_file),
                                                           model=hiagm,
                                                           config=config,
                                                           optimizer=optimize)
                logger.info('Previous Best Performance---- Micro-F1: {}%, Macro-F1: {}%'.format(
                    best_performance[0], best_performance[1]))

        # train
        for epoch in range(config.train.start_epoch, config.train.end_epoch):
            start_time = time.time()
            trainer.train(train_loader,
                          epoch)
            trainer.eval(train_loader, epoch, 'TRAIN')
            performance = trainer.eval(dev_loader, epoch, 'DEV')
            # saving best model and check model
            if not (performance['micro_f1'] >= best_performance[0] or performance['macro_f1'] >= best_performance[1]):
                wait += 1
                if wait % config.train.optimizer.lr_patience == 0:
                    logger.warning(
                        "Performance has not been improved for {} epochs, updating learning rate".format(wait))
                    trainer.update_lr()
                if wait == config.train.optimizer.early_stopping:
                    logger.warning("Performance has not been improved for {} epochs, stopping train with early stopping"
                                   .format(wait))
                    break

            if performance['micro_f1'] > best_performance[0]:
                wait = 0
                logger.info('Improve Micro-F1 {}% --> {}%'.format(best_performance[0], performance['micro_f1']))
                best_performance[0] = performance['micro_f1']
                best_epoch[0] = epoch
                save_checkpoint({
                    'epoch': epoch,
                    'model_type': config.model.type,
                    'state_dict': hiagm.state_dict(),
                    'best_performance': best_performance,
                    'optimizer': optimize.state_dict()
                }, os.path.join(model_checkpoint, 'best_micro_' + model_name))
            if performance['macro_f1'] > best_performance[1]:
                wait = 0
                logger.info('Improve Macro-F1 {}% --> {}%'.format(best_performance[1], performance['macro_f1']))
                best_performance[1] = performance['macro_f1']
                best_epoch[1] = epoch
                save_checkpoint({
                    'epoch': epoch,
                    'model_type': config.model.type,
                    'state_dict': hiagm.state_dict(),
                    'best_performance': best_performance,
                    'optimizer': optimize.state_dict()
                }, os.path.join(model_checkpoint, 'best_macro_' + model_name))

            if epoch % 10 == 1:
                save_checkpoint({
                    'epoch': epoch,
                    'model_type': config.model.type,
                    'state_dict': hiagm.state_dict(),
                    'best_performance': best_performance,
                    'optimizer': optimize.state_dict()
                }, os.path.join(model_checkpoint, model_name + '_epoch_' + str(epoch)))

            logger.info('Epoch {} Time Cost {} secs.'.format(epoch, time.time() - start_time))

        best_epoch_model_file = os.path.join(model_checkpoint, 'best_micro_' + model_name)
        if os.path.isfile(best_epoch_model_file):
            load_checkpoint(best_epoch_model_file, model=hiagm,
                            config=config,
                            optimizer=optimize)
            trainer.eval(test_loader, best_epoch[0], 'TEST')

        best_epoch_model_file = os.path.join(model_checkpoint, 'best_macro_' + model_name)
        if os.path.isfile(best_epoch_model_file):
            load_checkpoint(best_epoch_model_file, model=hiagm,
                            config=config,
                            optimizer=optimize)
            trainer.eval(test_loader, best_epoch[1], 'TEST')

        # Use the model to predict test/validation samples
        y_pred, y_true = _predict(trainer.model, dev_loader)  # (samples, num_classes)

        # Compute metrics with sklearn
        metrics = compute_metrics(y_true, y_pred, argmax_flag=False)
        # hiagm.label_map

        h_metrics = compute_hierarchical_metrics(y_true, y_pred, encoder_dump_or_mapping=hiagm.vocab,
                                                 taxonomy_path=Path(config['data']["taxonomy_path"]))
        # Save metric for current fold
        # requires python 3.9
        all_metrics = metrics | h_metrics  # join
        results.append(all_metrics)

    df = pd.DataFrame(results)
    df.loc["avg", :] = df.mean(axis=0)
    save_name = f"results_{'val' if validation else 'test'}_{config_numer}"
    save_path = Path(config["results_dir"])
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(save_path / (save_name + ".csv"))
    # dump_yaml(config, save_path / (save_name + ".yml"))


if __name__ == "__main__":
    base_path: Path = Path("config/HiAGM")
    # configs = [base_path / f"bugs_{x+1}.json" for x in (range(4, 12))]  # remember the +1
    configs = [base_path / 'hiagm_config.json']
    for number, c in enumerate(configs):
        config = Configure(config_json_file=c)

        if config.train.device_setting.device == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(config.train.device_setting.visible_device_list)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ''
        torch.manual_seed(2019)
        torch.cuda.manual_seed(2019)
        logger.Logger(config)

        # model_checkpoint = f"{configs.train.checkpoint.dir}_split_{split_num}"
        #
        # if not os.path.isdir(configs.train.checkpoint.dir):
        #     os.mkdir(configs.train.checkpoint.dir)

        train(config, number)


def load_previous_checkpoint(performance_logger, model_checkpoint, latest_model_file, hiagm, config, optimizer):
    logger.info('Loading Previous Checkpoint...')
    logger.info('Loading from {}'.format(os.path.join(model_checkpoint, latest_model_file)))
    performance_logger["best_performance"], config = load_checkpoint(
        model_file=os.path.join(model_checkpoint, latest_model_file),
        model=hiagm,
        config=config,
        optimizer=optimizer)
    logger.info('Previous Best Performance---- Micro-F1: {}%, Macro-F1: {}%'.format(
        performance_logger["best_performance"][0], performance_logger["best_performance"][1]))


def save_metric_checkpoint(performance, epoch, performance_logger, config, hiagm, optimizer):
    model_checkpoint = f"{config.train.checkpoint.dir}"
    model_name = config.model.type
    wait = None
    if performance['micro_f1'] > performance_logger["best_performance"][0]:
        wait = 0
        logger.info(
            'Improve Micro-F1 {}% --> {}%'.format(performance_logger["best_performance"][0], performance['micro_f1']))
        performance_logger["best_performance"][0] = performance['micro_f1']
        performance_logger["best_epoch"][0] = epoch
        save_checkpoint({
            'epoch': epoch,
            'model_type': config.model.type,
            'state_dict': hiagm.state_dict(),
            'best_performance': performance_logger["best_performance"],
            'optimizer': optimizer.state_dict()
        }, os.path.join(model_checkpoint, 'best_micro_' + model_name))
    if performance['macro_f1'] > performance_logger["best_performance"][1]:
        wait = 0
        logger.info(
            'Improve Macro-F1 {}% --> {}%'.format(performance_logger["best_performance"][1], performance['macro_f1']))
        performance_logger["best_performance"][1] = performance['macro_f1']
        performance_logger["best_epoch"][1] = epoch
        save_checkpoint({
            'epoch': epoch,
            'model_type': config.model.type,
            'state_dict': hiagm.state_dict(),
            'best_performance': performance_logger["best_performance"],
            'optimizer': optimizer.state_dict()
        }, os.path.join(model_checkpoint, 'best_macro_' + model_name))

    if epoch % 10 == 1:
        save_checkpoint({
            'epoch': epoch,
            'model_type': config.model.type,
            'state_dict': hiagm.state_dict(),
            'best_performance': performance_logger["best_performance"],
            'optimizer': optimizer.state_dict()
        }, os.path.join(model_checkpoint, model_name + '_epoch_' + str(epoch)))

    return wait


def load_best_epoch(trainer, test_loader, performance_logger, hiagm, config, optimizer):
    model_checkpoint = f"{config.train.checkpoint.dir}"
    model_name = config.model.type

    best_epoch_model_file = os.path.join(model_checkpoint, 'best_micro_' + model_name)
    if os.path.isfile(best_epoch_model_file):
        load_checkpoint(best_epoch_model_file, model=hiagm,
                        config=config,
                        optimizer=optimizer)
        trainer.eval(test_loader, performance_logger["best_epoch"][0], 'TEST')

    best_epoch_model_file = os.path.join(model_checkpoint, 'best_macro_' + model_name)
    if os.path.isfile(best_epoch_model_file):
        load_checkpoint(best_epoch_model_file, model=hiagm,
                        config=config,
                        optimizer=optimizer)
        trainer.eval(test_loader, performance_logger["best_epoch"][1], 'TEST')


def make_or_load_checkpoint(config, performance_logger, hiagm, optimizer):
    model_checkpoint = f"{config['train'].checkpoint.dir}"
    if not os.path.isdir(model_checkpoint):
        os.mkdir(model_checkpoint)
        config['train'].start_epoch = 0
    else:
        # loading previous checkpoint
        dir_list = os.listdir(model_checkpoint)
        dir_list.sort(key=lambda fn: os.path.getatime(os.path.join(model_checkpoint, fn)))
        latest_model_file = ''
        for model_file in dir_list[::-1]:
            if model_file.startswith('best'):
                continue
            else:
                latest_model_file = model_file
                break
        if os.path.isfile(os.path.join(model_checkpoint, latest_model_file)):
            load_previous_checkpoint(performance_logger, model_checkpoint,
                                     latest_model_file, hiagm, config, optimizer)
