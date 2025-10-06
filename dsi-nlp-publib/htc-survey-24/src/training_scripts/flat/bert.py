import datetime as dt
import os
from pathlib import Path
from typing import Type, List, Dict, Callable, Optional

import torch
import pandas as pd
import numpy as np

from src.dataset_tools.dataset_manager import DatasetManager
from src.models.BERT_flat.bert.bert_classifier import BERTForClassification
from src.models.BERT_flat.utility_functions import _setup_training, _predict
from src.training_scripts.script_utils import save_results
from src.utils.generic_functions import load_yaml, get_model_dump_path
from src.utils.metrics import compute_metrics, compute_hierarchical_metrics

import time

# DEBUG PARAMETERS
workers = 0


def fn_sigmoid(p, device):
    return p[0].sigmoid().to(device), p[1].to(device)


def run_training(config: Dict, dataset: str, model_class, out_folder: Path, split_fun: Optional[Callable] = None):
    ds_manager = DatasetManager(dataset_name=dataset, training_config=config)
    _training_testing_loop(config, model_class, ds_manager, out_folder,
                           logits_fn=fn_sigmoid, split_fun=split_fun)


def _train_single_split(x_train, x_val, x_test, y_train, y_val, y_test,
                        config: Dict, model_class: Type,
                        logits_fn, enc_: Path):
    # Create and train a model
    trainer, train_load, val_load, test_load = _setup_training(train_config=config, model_class=model_class,
                                                     workers=workers,
                                                     data=x_train, labels=y_train,
                                                     data_val=x_val, labels_val=y_val, data_test=x_test, labels_test=y_test,
                                                     logits_fn=logits_fn,
                                                     enc_=enc_)
    trainer.train(train_load, val_load)

    # TEST the model
    # First reload last improving epoch
    trainer.load_previous(trainer.last_saved_checkpoint, model_only=True)
    model = trainer.model
    # Use the model to predict test/validation samples
    y_pred, y_true, inf_time = _predict(model, test_load, config, logits_fn)  # (samples, num_classes)
    # print(f" ---> Inference time x sample: {inf_time / config['TEST_BATCH_SIZE']} ns")

    # Compute metrics with sklearn
    metrics = compute_metrics(y_true, y_pred, argmax_flag=False)
    h_metrics = compute_hierarchical_metrics(y_true, y_pred, encoder_dump_or_mapping=enc_,
                                             taxonomy_path=Path(config["taxonomy_path"]))
    metrics.update(**h_metrics)
    metrics.update(**{"inf_time": inf_time / config["TEST_BATCH_SIZE"]})

    # Necessary for sequential run. Empty cache should be automatic, but best be sure.
    del trainer, model
    torch.cuda.empty_cache()

    # Return metric for current fold
    return metrics, y_pred, y_true

def _training_testing_loop_old(config: Dict,
                           model_class: Type,
                           dataset: DatasetManager,
                           out_folder: Path,
                           logits_fn,
                           split_fun: Optional[Callable] = None,
                           save_name: str = None):
    multilabel, champ_loss, match_loss = config["multilabel"], \
                                         config.get("CHAMP_LOSS", False), \
                                         config.get("MATCH_LOSS", False)
    assert champ_loss <= multilabel, "Champ loss is only for multilabel settings"
    assert match_loss <= multilabel, "MATCH loss is only for multilabel settings"

    # General parameters
    model_folder = out_folder
    os.makedirs(model_folder, exist_ok=True)
    # config["MODEL_FOLDER"] = str(model_folder)
    results = list()
    # Train in splits
    fold_i: int = 0
    for (x_train, y_train), (x_test, y_test) in dataset.get_split():
        fold_i += 1
        print(f"Building model for fold {fold_i}.")
        config["MODEL_FOLDER"] = str(model_folder / f"fold_{fold_i}")
        if split_fun is not None:
            config.update(split_fun(fold_i))
        metrics = _train_single_split(x_train, x_test, y_train, y_test,
                                      config, model_class, logits_fn, dataset.binarizer)
        results.append(metrics)
        # ---------------------------------------
        save_results(results, out_folder, config)



def _training_testing_loop_kfold(config: Dict,
                           model_class: Type,
                           dataset: DatasetManager,
                           out_folder: Path,
                           logits_fn,
                           split_fun: Optional[Callable] = None,
                           save_name: str = None):
    multilabel, champ_loss, match_loss = config["multilabel"], \
                                         config.get("CHAMP_LOSS", False), \
                                         config.get("MATCH_LOSS", False)
    assert champ_loss <= multilabel, "Champ loss is only for multilabel settings"
    assert match_loss <= multilabel, "MATCH loss is only for multilabel settings"

    # General parameters
    model_folder = out_folder
    os.makedirs(model_folder, exist_ok=True)
    # config["MODEL_FOLDER"] = str(model_folder)
    save_preds = config.get("RELOAD", False)  # <- check if this is a testing run
    results = list()
    pred_y_list, true_y_list = list(), list()
    # Train in splits
    fold_i: int = 0
    for (x_train, y_train), (x_test, y_test) in dataset.get_split():
        fold_i += 1
        print(f"Building model for fold {fold_i}.")
        config["MODEL_FOLDER"] = str(model_folder / f"fold_{fold_i}")
        if split_fun is not None:
            config.update(split_fun(fold_i))
        metrics, y_pred, y_true = _train_single_split(x_train, x_test, y_train, y_test,
                                      config, model_class, logits_fn, dataset.binarizer)
        results.append(metrics)
        
        # ---------------------------------------
        save_results(results, out_folder, config)
        if save_preds:
            pred_y_list.append(y_pred)
            true_y_list.append(y_true)

    if save_preds:
        all_preds = pd.DataFrame(np.vstack(pred_y_list))
        all_trues = pd.DataFrame(np.vstack(true_y_list))
        all_preds.to_csv(out_folder / f"all_folds_pred_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv", index=False)
        all_trues.to_csv(out_folder / f"all_folds_true_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv", index=False)
        
def _training_testing_loop(config: Dict,
                           model_class: Type,
                           dataset: DatasetManager,
                           out_folder: Path,
                           logits_fn,
                           split_fun: Optional[Callable] = None,
                           save_name: str = None):
    multilabel, champ_loss, match_loss = config["multilabel"], \
                                         config.get("CHAMP_LOSS", False), \
                                         config.get("MATCH_LOSS", False)
    assert champ_loss <= multilabel, "Champ loss is only for multilabel settings"
    assert match_loss <= multilabel, "MATCH loss is only for multilabel settings"

    # General parameters
    model_folder = out_folder
    os.makedirs(model_folder, exist_ok=True)
    # config["MODEL_FOLDER"] = str(model_folder)
    save_preds = config.get("RELOAD", False)  # <- check if this is a testing run
    n_repeats = config.get("n_repeats", 1)
    results = list()
    pred_y_list, true_y_list = list(), list()
    # Train in splits
    fold_i: int = 0
    for (x_train, y_train), (x_val, y_val), (x_test, y_test) in dataset.get_split():
        #fold_i += 1
        if config["RELOAD"]:
            for r in range(1, n_repeats + 1):
                config["MODEL_FOLDER"] = str(model_folder / f"repeat_{r}")
                metrics, y_pred, y_true = _train_single_split(x_train, x_val, x_test, y_train, y_val, y_test,
                                        config, model_class, logits_fn, dataset.binarizer)
                
                results.append(metrics)

        else:
            for r in range(1, n_repeats + 1):
                print(f"\n=== Repeat {r}/{n_repeats} ===")
                config["MODEL_FOLDER"] = str(model_folder / f"repeat_{r}")
                os.makedirs(str(model_folder / f"repeat_{r}"), exist_ok=True)
                metrics, y_pred, y_true = _train_single_split(x_train, x_val, x_test, y_train, y_val, y_test,
                                        config, model_class, logits_fn, dataset.binarizer)
                
                results.append(metrics)
        
        # ---------------------------------------
        save_results(results, out_folder, config)
    
        if save_preds:
            pred_y_list.append(y_pred)
            true_y_list.append(y_true)

    if save_preds:
        all_preds = pd.DataFrame(np.vstack(pred_y_list))
        all_trues = pd.DataFrame(np.vstack(true_y_list))
        all_preds.to_csv(out_folder / f"all_folds_pred_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv", index=False)
        all_trues.to_csv(out_folder / f"all_folds_true_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv", index=False)
        
def run_configuration():
    # Paths
    config_base_path: Path = Path("config") / "BERT"
    output_path: Path = Path("dumps") / "BERT"
    config_list: List = ["bert_bgc.yml"] #"bert_amz.yml",  #"bert_wos.yml", ] #, "bert_wos.yml", "bert_rcv1.yml", "bert_bugs.yml", 

    for c in config_list:
        # Prepare configuration
        config_path: Path = (config_base_path / c)
        config: Dict = load_yaml(config_path)
        specific_model = f"{config['name']}_{config['CLF_STRATEGY']}_{config['CLF_STRATEGY_NUM_LAYERS']}"
        print(f"Specific model: {specific_model}")
        print(f"Dataset: {config['dataset']}")
        # Prepare output
        out_folder = output_path / specific_model / f"run_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        kw = dict()
        if config["RELOAD"] is True:
            reload_path = Path(config["PATH_TO_RELOAD"])
            kw = dict(split_fun=lambda f: get_model_dump_path(reload_path, f, config.get("EPOCH_RELOAD", None)))
            out_folder = reload_path

        # Train
        run_training(config=config,
                     dataset=config["dataset"],
                     model_class=BERTForClassification,
                     out_folder=out_folder, **kw)


if __name__ == "__main__":
    start_time = time.time()
    run_configuration()
    print("--- %s seconds ---" % (time.time() - start_time))
