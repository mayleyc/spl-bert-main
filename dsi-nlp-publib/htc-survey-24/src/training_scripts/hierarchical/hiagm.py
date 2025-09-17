import os
import time
from pathlib import Path
from typing import List, Dict

from sklearn.model_selection import train_test_split

import src.models.HiAGM.helper.logger as logger
from src.dataset_tools.dataset_manager import DatasetManager
from src.models.HiAGM import Vocab, Configure, HiAGM, _predict, \
    set_optimizer, ClassificationLoss, Trainer, data_loaders
from src.models.HiAGM.prepare_our_data import write_jsonl_split
from src.models.HiAGM.train import save_metric_checkpoint, load_best_epoch, \
    make_or_load_checkpoint
from src.training_scripts.script_utils import save_results
from src.utils.metrics import compute_metrics, compute_hierarchical_metrics

os.linesep = '\n'


def run_training(config: Configure, dataset: str, out_folder: Path):
    ds_manager = DatasetManager(dataset_name=dataset, training_config=config.dict)
    os.makedirs(out_folder, exist_ok=True)
    containing_folder = out_folder
    results = list()
    # Train in splits
    fold_i: int = 0
    for (x_train, y_train), (x_test, y_test) in ds_manager.get_split():
        fold_i += 1
        out_folder = containing_folder / f"fold_{fold_i}"

        config['train'].checkpoint.dir = out_folder / "checkpoints"
        os.makedirs(out_folder, exist_ok=True)
        # WOS/RCV1 can't be stratified because of few classes
        # Get validation data
        stratify = y_train if dataset != "wos" and dataset != "rcv1" else None
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2,
                                                          random_state=ds_manager.seeds['validation_split_seed'],
                                                          stratify=stratify)
        # Loading corpus and generate vocabulary
        corpus_vocab: Vocab = write_jsonl_split(config, (x_train, ds_manager.binarizer.inverse_transform(y_train)),
                                                (x_test, ds_manager.binarizer.inverse_transform(y_test)),
                                                (x_val, ds_manager.binarizer.inverse_transform(y_val)))
        # Get data
        train_loader, val_loader, test_loader = data_loaders(config, corpus_vocab)

        print(f"Building model for fold {fold_i}.")
        # build up model
        hiagm = HiAGM(config, corpus_vocab, model_type=config['model'].type, model_mode='TRAIN')
        hiagm.to(config['train'].device_setting.device)
        # define training objective & optimizer
        criterion = ClassificationLoss(os.path.join(config['data'].data_dir, config['data'].hierarchy),
                                       corpus_vocab.v2i['label'],
                                       recursive_penalty=config['train'].loss.recursive_regularization.penalty,
                                       recursive_constraint=config['train'].loss.recursive_regularization.flag)
        optimizer = set_optimizer(config, hiagm)

        # get epoch trainer
        trainer = Trainer(model=hiagm,
                          criterion=criterion,
                          optimizer=optimizer,
                          vocab=corpus_vocab,
                          config=config)

        # set origin log
        performance_logger: Dict = {
            "best_epoch": [-1, -1],
            "best_performance": [0.0, 0.0]
        }
        wait = 0
        make_or_load_checkpoint(config, performance_logger, hiagm, optimizer)

        # train
        for epoch in range(config["train"].start_epoch, config["train"].end_epoch):
            start_time = time.time()
            trainer.train(train_loader,
                          epoch)
            trainer.eval(train_loader, epoch, 'TRAIN')
            performance = trainer.eval(val_loader, epoch, 'DEV')
            # saving best model and check model
            if not (performance['micro_f1'] >= performance_logger["best_performance"][0] or performance['macro_f1'] >=
                    performance_logger["best_performance"][1]):
                wait += 1
                if wait % config["train"].optimizer.lr_patience == 0:
                    logger.warning(
                        "Performance has not been improved for {} epochs, updating learning rate".format(wait))
                    trainer.update_lr()
                if wait == config["train"].optimizer.early_stopping:
                    logger.warning("Performance has not been improved for {} epochs, stopping train with early stopping"
                                   .format(wait))
                    break

            n_w = save_metric_checkpoint(performance, epoch, performance_logger, config, hiagm, optimizer)
            wait = n_w if n_w is not None else wait
            logger.info(f'Epoch {epoch} Time Cost {time.time() - start_time} secs.')

        load_best_epoch(trainer, test_loader, performance_logger, hiagm, config, optimizer)

        # Use the model to predict test/validation samples
        y_pred, y_true, inf_time = _predict(trainer.model, val_loader)  # (samples, num_classes)

        # Compute metrics with sklearn
        metrics = compute_metrics(y_true, y_pred, argmax_flag=False)
        # hiagm.label_map

        h_metrics = compute_hierarchical_metrics(y_true, y_pred, encoder_dump_or_mapping=hiagm.vocab,
                                                 taxonomy_path=Path(config['data']["taxonomy_path"]))
        # Save metric for current fold
        # requires python 3.9
        all_metrics = metrics | h_metrics | {"inf_time": inf_time / config["train"]["batch_size"]}  # join
        results.append(all_metrics)

        save_results(results, containing_folder, config.dict)


def run_configuration():
    # https://drive.google.com/file/d/1FvXtFwSvO2Nb4vn6F8at95hPyZ3yOyqk/view?usp=sharing
    model_name: str = "HiAGM"
    # Paths
    config_base_path: Path = Path("config") / model_name
    output_path: Path = Path("dumps") / model_name
    config_list: List = [
        # "hiagm_config_blurb.json",
        # "hiagm_config_wos.json",
        "hiagm_config_bugs.json",
        # "hiagm_config_amazon.json",
        # "hiagm_config_rcv1.json",
    ]

    for c in config_list:
        # Prepare configuration
        config_path: Path = (config_base_path / c)
        config = Configure(config_json_file=config_path)
        logger.Logger(config)

        if config["train"].device_setting.device == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(config["train"].device_setting.visible_device_list)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ''

        print(f"Specific model: {model_name}")
        print(f"Dataset: {config['data']['dataset']}")
        # Prepare output
        out_folder = output_path / config['data']['dataset']
        # Train
        run_training(config=config,
                     dataset=config["data"]["dataset"],
                     out_folder=out_folder)


if __name__ == "__main__":
    run_configuration()
