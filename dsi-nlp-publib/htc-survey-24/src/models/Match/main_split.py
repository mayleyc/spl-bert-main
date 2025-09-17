# CODE BY: https://github.com/yuzhimanhua/MATCH
# Rights reserved to original authors. Only minor adaptations have been made.

import faulthandler
import os
from pathlib import Path

import click
import pandas as pd
import torch
from logzero import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from deepxml.data_utils import get_mlb
from deepxml.dataset import MultiLabelDataset
from deepxml.match import MATCH
from deepxml.models import Model
from src.dataset_tools import get_bgc_split_jsonl, get_rcv1_split
from src.models.Match.deepxml.additional_utils import get_data_dump_new
from src.utils.generic_functions import load_yaml, dump_yaml
from src.utils.metrics import compute_metrics, compute_hierarchical_metrics
from src.utils.text_utilities.multilabel import normalize_labels

faulthandler.enable()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# --reg 1 -d config/Match/old/BGC.yaml -m config/Match/old/model.yaml --mode train


@click.command()
@click.option('-d', '--data-cnf', type=click.Path(exists=True), help='Path of dataset configure yaml.')
@click.option('-m', '--model-cnf', type=click.Path(exists=True), help='Path of model configure yaml.')
@click.option('--mode', type=click.Choice(['train', 'eval']), default=None)
@click.option('--reg', type=click.BOOL, default=False)
def main(data_cnf, model_cnf, mode, reg):
    data_cnf, model_cnf = load_yaml(Path(data_cnf)), load_yaml(Path(model_cnf))
    model, model_name, data_name = None, model_cnf['name'], data_cnf['name']
    model_path = os.path.join(model_cnf['path'], F'{model_name}-{data_name}')
    out_path = os.path.join(data_cnf['output']["res"], F'{model_name}-{data_name}')
    logger.info(F'Model Name: {model_name}')

    if data_name == "BGC":
        train_d = get_bgc_split_jsonl("train")
        test_d = get_bgc_split_jsonl("test")
    elif data_name == "RCV1":
        train_d = get_rcv1_split("train")
        test_d = get_rcv1_split("test")
    else:
        raise ValueError
    fold_i = 0
    fold_tot = model_cnf["NUM_FOLD"]
    repeats = model_cnf["CV_REPEAT"]
    model_folder = Path(model_path)
    validation = model_cnf["validation"]
    # if data_name == "Bugs":
    #     labels = df[model_cnf["LABEL"]]
    #     labels_all: pd.DataFrame = df[model_cnf["ALL_LABELS"]]
    # else:
    y_train = [d["labels"] for d in train_d]
    labels = [ls[-1] for ls in y_train]
    x_train = [d["text"] for d in train_d]

    y_test = [d["labels"] for d in test_d]
    x_test = [d["text"] for d in test_d]

    # LOAD DATASET and EMBEDDINGS
    x_train, kv = get_data_dump_new(model_cnf, x_train, base_dump=Path(data_cnf["data_dump"]), fold=1, _set="train")
    x_test, _ = get_data_dump_new(model_cnf, x_test, base_dump=Path(data_cnf["data_dump"]), fold=1, _set="test")
    emb_init = kv.syn1neg

    if mode is None or mode == 'train':
        logger.info('Loading Training and Validation Set')
        # train_x, train_labels = get_data(data_cnf['train']['texts'], data_cnf['train']['labels'])
        # if 'size' in data_cnf['valid']:
        #     random_state = data_cnf['valid'].get('random_state', 1240)
        #     train_x, valid_x, train_labels, valid_labels = train_test_split(train_x, train_labels,
        #                                                                     test_size=data_cnf['valid']['size'],
        #                                                                     random_state=random_state)
        # else:
        #     valid_x, valid_labels = get_data(data_cnf['valid']['texts'], data_cnf['valid']['labels'])

        # Start K-Fold CV, repeating it for better significance
        results = list()
        seeds = load_yaml("config/random_seeds.yml")
        device = torch.device(model_cnf["device"])

        fold_i += 1
        print(f"Fold {fold_i}/{fold_tot * repeats} ({fold_tot} folds * {repeats} repeats)")
        model_folder_ = model_folder / f"fold_{fold_i}"
        data_cnf['labels_binarizer'] = str(model_folder_ / "labels_binarizer.jb")

        if validation is True:
            # Replace test set with validation set. Notice that the test set will be ignored,
            # and never used in validation or training
            x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2,
                                                                random_state=seeds["validation_split_seed"],
                                                                stratify=labels)
        # FOR DEBUG
        # x_train = x_train[:100]
        # x_test = x_test[:100]
        # y_train = y_train[:100]
        # y_test = y_test[:100]

        x_train, x_test, y_train, y_test = normalize_labels(x_train, x_test, y_train, y_test)

        # Split-specific binarizer
        mlb = get_mlb(Path(data_cnf['labels_binarizer']), y_train)

        y_train, y_test = mlb.transform(y_train), mlb.transform(y_test)
        labels_num = len(mlb.classes_)
        logger.info(F'Number of Labels: {labels_num}')
        logger.info(F'Size of Training Set: {len(x_train)}')
        logger.info(F'Size of {"Validation" if validation else "Test"} Set: {len(x_test)}')

        edges = set()
        if reg:
            classes = mlb.classes_.tolist()
            with open(data_cnf['hierarchy']) as fin:
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
            logger.info(F'Number of Edges: {len(edges)}')

        logger.info('Training')
        x_ml_ds_train = MultiLabelDataset(x_train.tolist(), y_train, vectors=kv.wv, config=model_cnf)
        train_loader = DataLoader(x_ml_ds_train,
                                  model_cnf['train']['batch_size'], shuffle=True, num_workers=0)
        valid_loader = DataLoader(
            MultiLabelDataset(x_test.tolist(), y_test, training=True, vectors=kv.wv, config=model_cnf),
            model_cnf['valid']['batch_size'], num_workers=0)
        model = Model(network=MATCH, labels_num=labels_num, model_path=model_folder_, emb_init=emb_init,
                      mode='train', reg=reg, hierarchy=edges,
                      **data_cnf['model'], **model_cnf['model'], label_binarizer=mlb, device=device)
        model.train(train_loader, valid_loader, **model_cnf['train'])
        logger.info('Finish Training')

        # Use the model to predict test/validation samples
        # y_pred, y_true = _predict(model.model, valid_loader)  # (samples, num_classes)
        y_pred, y_true, inf_time = model.predict(valid_loader)  # (samples, num_classes)

        # Compute metrics with sklearn
        metrics = compute_metrics(y_true, y_pred, argmax_flag=False)
        h_metrics = compute_hierarchical_metrics(y_true, y_pred,
                                                 encoder_dump_or_mapping=data_cnf["labels_binarizer"],
                                                 taxonomy_path=Path(data_cnf["taxonomy_path"]))
        metrics.update(**h_metrics)
        metrics = metrics | {"inf_time": inf_time}
        # Save metric for current fold
        results.append(metrics)

        # Necessary for sequential run. Empty cache should be automatic, but best be sure.
        del model
        torch.cuda.empty_cache()

        # Average metrics over all folds and save them to csv
        df = pd.DataFrame(results)
        df.loc["avg", :] = df.mean(axis=0)
        save_name = f"results_{'val' if validation else 'test'}"
        save_path = Path(out_path)
        os.makedirs(save_path, exist_ok=True)
        df.to_csv(save_path / (save_name + ".csv"))
        dump_yaml(model_cnf, save_path / (save_name + ".yml"))

    # if mode is None or mode == 'eval':
    #     logger.info('Loading Test Set')
    #     mlb = get_mlb(data_cnf['labels_binarizer'])
    #     labels_num = len(mlb.classes_)
    #     test_x, _ = get_data(data_cnf['test']['texts'], None)
    #     logger.info(F'Size of Test Set: {len(test_x)}')
    #
    #     logger.info('Predicting')
    #     test_loader = DataLoader(MultiLabelDataset(test_x), model_cnf['predict']['batch_size'], num_workers=0)
    #     if model is None:
    #         model = Model(network=MATCH, labels_num=labels_num, model_path=model_path, emb_init=emb_init, mode='eval',
    #                       **data_cnf['model'], **model_cnf['model'])
    #     scores, labels = model.predict(test_loader, k=model_cnf['predict'].get('k', 100))
    #     logger.info('Finish Predicting')
    #     labels = mlb.classes_[labels]
    #     output_res(data_cnf['output']['res'], F'{model_name}-{data_name}', scores, labels)


if __name__ == "__main__":
    main()
