import os
import time
from operator import itemgetter
from pathlib import Path
from typing import Type, Callable, List, Dict, Any, Union

from types import SimpleNamespace


import joblib
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchmetrics as tm
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import compute_class_weight
from tqdm import tqdm
from transformers import AutoTokenizer

from src.dataset_tools.blurb.generate_hierarchy import get_bgc_split_txt
from src.dataset_tools.linux_bugs.prepare_linux_dataset import read_dataset as read_bugs
from src.dataset_tools.rcv1.generate_splits import get_rcv1_split
from src.dataset_tools.wos.generation import read_wos_dataset
from src.models.BERT_flat.losses.champ import BCEChampLoss
from src.models.BERT_flat.losses.match import BCEMatchLoss
from src.models.BERT_flat.losses.standard import ce_loss, bce_loss
from src.models.BERT_flat.torch_dataset import TransformerDatasetFlat
from src.utils.generic_functions import dump_yaml, load_yaml
from src.utils.metrics import compute_metrics, compute_hierarchical_metrics
from src.utils.text_utilities.multilabel import normalize_labels
from src.utils.torch_train_eval.early_stopper import EarlyStopping
from src.utils.torch_train_eval.evaluation import MetricSet
from src.utils.torch_train_eval.grad_accum_trainer import GradientAccumulatorTrainer
from src.utils.torch_train_eval.trainer import Trainer
from hierarchy_dict_gen import AmazonTaxonomyParser, BGCParser, WOSTaxonomyParser
from src.dataset_tools.dataset_manager import DatasetManager

import os
import datetime
import json
from time import perf_counter
import copy
import pickle
import glob
from itertools import combinations


import torch
import torch.nn as nn
import torch.nn.init as init


import json
from timeit import default_timer as timer


from sklearn import preprocessing


# misc
from common import *

import_cmpe = True
if import_cmpe:
    from src.models.BERT_flat.bert.cmpe import *


def log1mexp(x):
        assert(torch.all(x >= 0))
        return torch.where(x < 0.6931471805599453094, torch.log(-torch.expm1(-x)), torch.log1p(-torch.exp(-x)))

from torch.utils.data import Dataset
from PIL import Image

taxonomy_fp = ["data/Amazon/amazon_tax.txt", "data/BGC/bgc_tax.txt", "data/WebOfScience/wos_tax.txt"]

def get_loss_function(enc_: Path, train_config: Dict, weights=None):
    champ_loss, match_loss = (train_config.get("CHAMP_LOSS", False),
                              train_config.get("MATCH_LOSS", False))
    if isinstance(enc_, Path):
        mlb: MultiLabelBinarizer = joblib.load(enc_)
    elif isinstance(enc_, MultiLabelBinarizer):
        mlb: MultiLabelBinarizer = enc_
    loss_func = None
    if champ_loss:
        g: nx.Graph = nx.read_adjlist(train_config["taxonomy_path"], nodetype=str)
        ce_args = dict(g=g, enc=mlb, beta=.2, device=train_config["DEVICE"])
        loss_func = BCEChampLoss(**ce_args)
        print("Using BCE with CHAMP regularization")
    if match_loss:
        edges = set()
        #classes = mlb.classes_.tolist()
        # Load classes from taxonomy file instead of mlb
        if "amazon" in train_config["taxonomy_path"]:
            tax = AmazonTaxonomyParser(train_config["taxonomy_path"])
        elif "bgc" in train_config["taxonomy_path"]:   
            tax = BGCParser(train_config["taxonomy_path"])
        elif "wos" in train_config["taxonomy_path"]:
            tax = WOSTaxonomyParser(train_config["taxonomy_path"])

        tax.parse()  
        _, classes = tax._build_one_hot()
        with open(train_config["taxonomy_path"]) as fin:
            next(fin)  # skip first entry
            for line in fin:
                data = line.strip().split()
                p = data[0]
                if p not in classes:
                    continue
                p_id = classes.index(p)
                for c in data[1:]:
                    if c not in classes:
                        continue
                    c_id = classes.index(c)
                    edges.add((p_id, c_id))
        if champ_loss:
            ce_args = dict(hierarchy=edges, device=train_config["DEVICE"], base_loss=loss_func)
        else:
            ce_args = dict(hierarchy=edges, device=train_config["DEVICE"],
                           base_loss=F.binary_cross_entropy_with_logits)
        loss_func = BCEMatchLoss(**ce_args)
        print("Using BCE with MATCH regularization")
    if not champ_loss and not match_loss:
        if train_config.get("CLASS_BALANCED_WEIGHTED_LOSS", False) is True:
            loss_func = lambda *x, **k: bce_loss(*x, **k, weight=weights)
            print("Using weighted BCE loss")
        else:
            loss_func = bce_loss
            print("Using BCE loss")
    return loss_func


def collate_batch(tokenizer, batch, ml: bool = False):
    x = [t for t, *_ in batch]
    max_len = tokenizer.model_max_length if tokenizer.model_max_length <= 2048 else 512
    encoded_x = tokenizer(x, truncation=True, max_length=max_len, padding=True)
    '''print("encoded_x:", type(encoded_x)) #dict
    k, v = next(iter(encoded_x.items()))
    print("encoded_x item:", type(k), "\n", type(v)) #dict
    quit()'''

    item_x = {key: torch.tensor(val) for key, val in encoded_x.items()}
    y = [t for _, t in batch]
    return item_x, torch.LongTensor(y) if not ml else torch.stack(y, dim=0).long()


def compute_class_weights(ds) -> torch.FloatTensor:
    labs = ds.y.detach().cpu().numpy()
    labs = np.sort(labs)
    uniques, counts = np.unique(labs, return_counts=True)
    weights = compute_class_weight(class_weight="balanced", classes=uniques, y=labs)
    # tot = float(len(labs))
    # weights = counts / tot
    return torch.FloatTensor(weights)


def _setup_training(train_config, model_class: Type, workers: int, data, labels, data_val, labels_val, data_test, labels_test,
                    logits_fn: Callable, enc_: Path, **spl_args):
    # -------------------------------
    tokenizer = AutoTokenizer.from_pretrained(train_config["PRETRAINED_LM"])
    # dataset_class = TransformerDatasetFlat
    multilabel, champ_loss, match_loss = (train_config["multilabel"],
                                          train_config.get("CHAMP_LOSS", False),
                                          train_config.get("MATCH_LOSS", False))

    args_d = dict(remove_garbage=train_config["REMOVE_GARBAGE_TEXT"], multilabel=multilabel, encoder_path=enc_)
    train_data = TransformerDatasetFlat(data, labels, **args_d)
    val_data = TransformerDatasetFlat(data_val, labels_val, **args_d)
    test_data = TransformerDatasetFlat(data_test, labels_test, **args_d)
    train_config["n_class"] = train_data.n_y
    spl = train_config["spl"]
    # Initialize model
    if spl:
        base_model = model_class(**train_config)

        spl_ns = SimpleNamespace(**spl_args)
        
        device = getattr(spl_ns, "device", "cuda:0")  # default device
        dataset_name = getattr(spl_ns, "dataset_name", None)
        mat = getattr(spl_ns, "mat", None)
        num_st_nodes = getattr(spl_ns, "num_st_nodes", None)
        size = base_model.last_layer_size # get size of last layer from model

        cmpe, gate, R = get_circuit(device, dataset_name, mat, size, num_st_nodes, S = 2, gates=2, num_reps=1)
        model = SPLBERTModel(cmpe, gate, base_model=base_model)

    else:
        model = model_class(**train_config)

    # Initialize Optimizer and loss
    opt = torch.optim.AdamW(model.parameters(), lr=train_config["LEARNING_RATE"], weight_decay=train_config["L2_REG"])

    early_stopper = EarlyStopping(*train_config["EARLY_STOPPING"].values())
    # -------------------------------
    # Prepare dataset
    training_loader = torch.utils.data.DataLoader(train_data, batch_size=train_config["BATCH_SIZE"],
                                                  num_workers=workers, shuffle=True,
                                                  collate_fn=lambda x: collate_batch(tokenizer, x, ml=multilabel))
    validation_loader = torch.utils.data.DataLoader(val_data, batch_size=train_config["BATCH_SIZE"],
                                                    num_workers=workers, shuffle=True,
                                                    collate_fn=lambda x: collate_batch(tokenizer, x, ml=multilabel))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=train_config["TEST_BATCH_SIZE"],
                                                    num_workers=workers, shuffle=True,
                                                    collate_fn=lambda x: collate_batch(tokenizer, x, ml=multilabel))
    # -------------------------------
    w = None
    if train_config["CLASS_BALANCED_WEIGHTED_LOSS"] is True:
        w = compute_class_weights(train_data).to(train_config["DEVICE"])

    if not multilabel:
        if train_config["CLASS_BALANCED_WEIGHTED_LOSS"] is True:
            loss_func = lambda *x, **k: ce_loss(*x, **k, weight=w)
            print("Using weighted CE loss (multiclass)")
        else:
            loss_func = ce_loss
            print("Using CE loss (multiclass)")
    else:
        loss_func = get_loss_function(enc_, train_config, weights=w)
    
    if spl:
        loss_func = lambda pred, y, nll, l, device: nll #uses nll instead of standard loss

    # -------------------------------
    # Metrics
    metrics = MetricSet({
        "-": (
            tm.MetricCollection({
                "f1_macro": tm.F1Score(average="macro", num_classes=train_data.n_y),
                "acc_macro": tm.Accuracy(average="macro", num_classes=train_data.n_y, subset_accuracy=True)
            }),
            logits_fn
        )
    })
    # -------------------------------
    # Initiate training
    if train_config.get("gradient_accum_train"):
        print(f"Running with gradient accumulation: Simulated bs: {train_config['simulated_bs']}")
        trainer = GradientAccumulatorTrainer(model, train_config, loss_func, opt,
                                             early_stopper, metrics, unpack_flag=False)
    else:
        trainer = Trainer(model, train_config, loss_func, opt, early_stopper, metrics, unpack_flag=False,
                          add_start_time_folder=False)
    return trainer, training_loader, validation_loader, test_loader

def _setup_training_kfold(train_config, model_class: Type, workers: int, data, labels, data_val, labels_val,
                    logits_fn: Callable, enc_: Path):
    # -------------------------------
    tokenizer = AutoTokenizer.from_pretrained(train_config["PRETRAINED_LM"])
    # dataset_class = TransformerDatasetFlat
    multilabel, champ_loss, match_loss = (train_config["multilabel"],
                                          train_config.get("CHAMP_LOSS", False),
                                          train_config.get("MATCH_LOSS", False))

    args_d = dict(remove_garbage=train_config["REMOVE_GARBAGE_TEXT"], multilabel=multilabel, encoder_path=enc_)
    train_data = TransformerDatasetFlat(data, labels, **args_d)
    val_data = TransformerDatasetFlat(data_val, labels_val, **args_d)
    train_config["n_class"] = train_data.n_y
    # Initialize model
    model = model_class(**train_config)

    # Initialize Optimizer and loss
    opt = torch.optim.AdamW(model.parameters(), lr=train_config["LEARNING_RATE"], weight_decay=train_config["L2_REG"])

    early_stopper = EarlyStopping(*train_config["EARLY_STOPPING"].values())
    # -------------------------------
    # Prepare dataset
    training_loader = torch.utils.data.DataLoader(train_data, batch_size=train_config["BATCH_SIZE"],
                                                  num_workers=workers, shuffle=True,
                                                  collate_fn=lambda x: collate_batch(tokenizer, x, ml=multilabel))
    validation_loader = torch.utils.data.DataLoader(val_data, batch_size=train_config["TEST_BATCH_SIZE"],
                                                    num_workers=workers, shuffle=True,
                                                    collate_fn=lambda x: collate_batch(tokenizer, x, ml=multilabel))
    
    # -------------------------------
    w = None
    if train_config["CLASS_BALANCED_WEIGHTED_LOSS"] is True:
        w = compute_class_weights(train_data).to(train_config["DEVICE"])

    if not multilabel:
        if train_config["CLASS_BALANCED_WEIGHTED_LOSS"] is True:
            loss_func = lambda *x, **k: ce_loss(*x, **k, weight=w)
            print("Using weighted CE loss (multiclass)")
        else:
            loss_func = ce_loss
            print("Using CE loss (multiclass)")
    else:
        loss_func = get_loss_function(enc_, train_config, weights=w)

    # -------------------------------
    # Metrics
    metrics = MetricSet({
        "-": (
            tm.MetricCollection({
                "f1_macro": tm.F1Score(average="macro", num_classes=train_data.n_y),
                "acc_macro": tm.Accuracy(average="macro", num_classes=train_data.n_y, subset_accuracy=True)
            }),
            logits_fn
        )
    })
    # -------------------------------
    # Initiate training
    if train_config.get("gradient_accum_train"):
        print(f"Running with gradient accumulation: Simulated bs: {train_config['simulated_bs']}")
        trainer = GradientAccumulatorTrainer(model, train_config, loss_func, opt,
                                             early_stopper, metrics, unpack_flag=False)
    else:
        trainer = Trainer(model, train_config, loss_func, opt, early_stopper, metrics, unpack_flag=False,
                          add_start_time_folder=False)
    return trainer, training_loader, validation_loader


def _predict(model, loader, config, logits_fn: Callable, t=.5):
    model.train(False)
    y_pred = list()
    y_true = list()
    infer_time_batch: int = 0
    
    spl = config["spl"]    

    with torch.no_grad():
        for i, pred_data in tqdm(enumerate(loader), total=len(loader)):
            infer_time_start: int = time.perf_counter_ns()
            x, labels = pred_data #unpack the batch
            args = model(pred_data) #tuple of 4
            infer_time_end: int = time.perf_counter_ns()
            infer_time_batch += infer_time_end - infer_time_start
             # logits - labels - etc
            if spl:
                l, y, loss, *_ = args
                pred = (l > 0).long()

                # negative log likelihood and map = CE loss (output from circuit)
                #cmpe.set_params(thetas)
                #nll = cmpe.cross_entropy(y, log_space=True).mean()
            else:
                l, y, *_ = args
                pred, _ = logits_fn(args, device="cpu")
                pred = torch.where(pred > t, 1, 0)

            y_pred.append(pred.detach().cpu().numpy())
            y_true.append(y.cpu().numpy())
    return np.concatenate(y_pred), np.concatenate(y_true), infer_time_batch / len(loader)


def train_single_split(x_train, x_test, y_train, y_test, config: Dict, model_class: Type, workers: int, logits_fn,
                       enc_: Path, validation: bool = False):
    # FOR DEBUG
    # x_train = x_train[:20]
    # x_test = x_test[:20]
    # y_train = y_train[:20]
    # y_test = y_test[:20]

    if config["multilabel"]:
        # Assumes y_train and y_test are dataframes with two columns
        x_train, x_test, y_train, y_test = normalize_labels(x_train, x_test, y_train, y_test)
    else:
        # Previous function also convert Dataframe to list of lists
        y_train, y_test = y_train.tolist(), y_test.tolist()

    
    # x_train, x_test are pd.Series, targets are lists

    # Create and train a model
    trainer, train_load, val_load = _setup_training(train_config=config, model_class=model_class,
                                                    workers=workers,
                                                    data=x_train, labels=y_train,
                                                    data_val=x_test, labels_val=y_test, logits_fn=logits_fn,
                                                    enc_=enc_)
    # Comment if you want ES in test
    trainer.train(train_load, val_load if validation else None)
    # trainer.train(train_load, val_load)

    # TEST the model
    # First reload last improving epoch
    trainer.load_previous(trainer.last_saved_checkpoint, model_only=True)
    model = trainer.model
    # Use the model to predict test/validation samples
    y_pred, y_true = _predict(model, val_load, config, logits_fn)  # (samples, num_classes)

    # Compute metrics with sklearn
    metrics = compute_metrics(y_true, y_pred, argmax_flag=False)
    h_metrics = compute_hierarchical_metrics(y_true, y_pred, encoder_dump_or_mapping=enc_,
                                             taxonomy_path=Path(config["taxonomy_path"]))
    metrics.update(**h_metrics)

    # Necessary for sequential run. Empty cache should be automatic, but best be sure.
    del trainer, model
    torch.cuda.empty_cache()

    # Return metric for current fold
    return metrics


def _training_testing_loop(config_path: Path, model_class: Type, workers: int,
                           data_samples: Union[List[Dict[str, Any]], Callable[[str], List[Dict[str, Any]]]],
                           out_folder: Path, logits_fn, validation: bool = False, save_name: str = None, split_fun=None,
                           cv_splits: bool = False):
    config = load_yaml(config_path)

    multilabel, champ_loss, match_loss = config["multilabel"], \
                                         config.get("CHAMP_LOSS", False), \
                                         config.get("MATCH_LOSS", False)
    assert champ_loss <= multilabel, "Champ loss is only for multilabel settings"
    assert match_loss <= multilabel, "MATCH loss is only for multilabel settings"

    # General parameters
    fold_i = 0
    fold_tot = config["NUM_FOLD"]
    repeats = config["CV_REPEAT"]
    model_folder = out_folder
    enc_ = model_folder / "label_binarizer.jb"
    os.makedirs(model_folder, exist_ok=True)
    config["MODEL_FOLDER"] = str(model_folder)
    results = list()
    n_repeats = config.get("n_repeats", 1)

    tickets: List[str] = [d["text"] for d in data_samples]
    labels: List[str] = [d[config["LABEL"]] for d in data_samples]
    labels_all = labels
    if "ALL_LABELS" in config and multilabel:
        labels_all: List[List[str]] = [d[config["ALL_LABELS"]] for d in data_samples]
    seeds = load_yaml("config/random_seeds.yml")
       
    dataset = DatasetManager(config["dataset"], config)

    if cv_splits:
        # Start K-Fold CV, repeating it for better significance
        splitter = RepeatedStratifiedKFold(n_splits=fold_tot, n_repeats=repeats,
                                           random_state=seeds["stratified_fold_seed"]) 

        for train_index, test_index in splitter.split(tickets, labels):
            fold_i += 1
            print(f"Fold {fold_i}/{fold_tot * repeats} ({fold_tot} folds * {repeats} repeats)")
            config["MODEL_FOLDER"] = str(model_folder / f"fold_{fold_i}")
            os.makedirs(config["MODEL_FOLDER"], exist_ok=True)
            enc_ = Path(config["MODEL_FOLDER"]) / "label_binarizer.jb"

            if split_fun is not None:
                config.update(split_fun(fold_i))
            if "MODEL_L1" in config.keys():
                print(f"L1: {config['MODEL_L1']}")
            if "MODEL_L2" in config.keys():
                print(f"L2: {config['MODEL_L2']}")
            x_train, x_test = itemgetter(*train_index)(tickets), itemgetter(*test_index)(
                tickets)  # tickets.iloc[train_index], tickets.iloc[test_index]
            y_train, y_test = itemgetter(*train_index)(labels_all), itemgetter(*test_index)(labels_all)
            '''if validation is True:
                # Replace test set with validation set. Notice that the test set will be ignored,
                # and never used in validation or training
                x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2,
                                                                    random_state=seeds["validation_split_seed"],
                                                                    stratify=itemgetter(*train_index)(labels))'''

            metrics = train_single_split(x_train, x_test, y_train, y_test, config, model_class, workers, logits_fn,
                                         enc_,
                                         validation)
            results.append(metrics)
    else:
        print("No CV splits, using predefined splits...")
        if config["RELOAD"]:
            # RELOAD mode: no new repeats, just load the existing repeats
            print("Reload mode: loading trained repeats...")
            for r in range(1, n_repeats + 1):
                repeat_path = Path(config["PATH_TO_RELOAD"]) / f"repeat_{r}"
                enc_ = repeat_path / "label_binarizer.jb"
                (x_train, y_train), (x_test, y_test), (x_val, y_val) = dataset.get_split()
                x_train = x_train + x_val
                y_train = y_train + y_val
                # Load the trained model and compute metrics
                metrics = train_single_split(
                    x_train, x_test, y_train, y_test,
                    config, model_class, workers, logits_fn, enc_, validation
                )
                results.append(metrics)

        else:
            # Normal training: run new repeats
            for r in range(1, n_repeats + 1):
                print(f"\n=== Repeat {r}/{n_repeats} ===")
                repeat_folder = model_folder / f"repeat_{r}"
                os.makedirs(repeat_folder, exist_ok=True)
                enc_ = repeat_folder / "label_binarizer.jb"
                (x_train, y_train), (x_test, y_test), (x_val, y_val) = dataset.get_split()
                x_train = x_train + x_val
                y_train = y_train + y_val
                metrics = train_single_split(
                    x_train, x_test, y_train, y_test,
                    config, model_class, workers, logits_fn, enc_, validation
                )
                results.append(metrics)

    # Average metrics over all folds and save them to csv
    df_results = pd.DataFrame(results)
    df_results.loc["avg", :] = df_results.mean(axis=0)
    save_name = save_name if save_name is not None else f"results_{'val' if validation else 'test'}"
    save_path = model_folder / "results"
    os.makedirs(save_path, exist_ok=True)
    df_results.to_csv(save_path / (save_name + ".csv"))
    dump_yaml(config, save_path / (save_name + ".yml"))


def run_training(config_path: Path, dataset: str, model_class, out_folder: Path,
                 workers: int, validation, logits_fn: Callable, split_fun=None):
    cv_splits = False
    if dataset == "bugs":
        _, samples = read_bugs()
    elif dataset == "wos":
        samples = read_wos_dataset()
    elif dataset == "bgc":
        samples = get_bgc_split_txt
        cv_splits = False  # whether splits are pre-determined or not
    elif dataset == "rcv1":
        samples = get_rcv1_split
        cv_splits = False  # whether splits are pre-determined or not
    elif dataset == "amazon":
        pass
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'.")
    _training_testing_loop(config_path, model_class, workers, samples, out_folder,
                           logits_fn=logits_fn, validation=validation, split_fun=split_fun, cv_splits=cv_splits)
