#!/usr/bin/env python
# coding:utf-8
from pathlib import Path
from typing import List

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from src.utils.metrics import compute_metrics, compute_hierarchical_metrics
from hierarchy_dict_gen import AmazonTaxonomyParser, BGCParser


import sys
import os
from datetime import datetime

# TODO: append path to main github repo folder
sys.path.append("/mnt/cimec-storage6/users/nguyenanhthu.tran/2025thesis/dsi-nlp-publib/htc-survey-24")
sys.path.append("/mnt/cimec-storage6/users/nguyenanhthu.tran/2025thesis/dsi-nlp-publib/htc-survey-24/src")
print(sys.path)


def _precision_recall_f1(right, predict, total):
    """
    :param right: int, the count of right prediction
    :param predict: int, the count of prediction
    :param total: int, the count of labels
    :return: p(precision, Float), r(recall, Float), f(f1_score, Float)
    """
    p, r, f = 0.0, 0.0, 0.0
    if predict > 0:
        p = float(right) / predict
    if total > 0:
        r = float(right) / total
    if p + r > 0:
        f = p * r * 2 / (p + r)
    return p, r, f

def binary_to_multilabel(classes, y: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Convert binary labels to multilabel format with a given threshold.

    :param y: binary labels
    :param threshold: threshold in [0, 1] to consider a prediction positive (> t)
    :return: multilabel representation of the input
    """
    binary_labels = np.where(y > threshold, 1, 0).astype(int)

    # Decode using class_names
    decoded_labels = []
    for row in binary_labels:
        labels = [classes[i] for i, val in enumerate(row) if val == 1]
        decoded_labels.append(labels)
    return decoded_labels

def transform_manual(class_to_index, y: List[List[str]], classes: List[str]) -> np.ndarray:
    mhe_extended = np.zeros((len(y), len(classes)), dtype=int)
    for i, labels in enumerate(y):
        for label in labels:
            if label in class_to_index: # checking a dictionary is faster than checking a list
                mhe_extended[i, class_to_index[label]] = 1
    return mhe_extended

def evaluate_old(epoch_predicts, epoch_labels, id2label, tax_file: str, threshold=0.5, top_k=None, as_sample=False):
    """
    :param epoch_labels: List[List[int]], ground truth, label id
    :param epoch_predicts: List[List[Float]], predicted probability list
    :param vocab: data_modules.Vocab object
    :param threshold: Float, filter probability for tagging
    :param tax_file: txt file with label taxonomy (for our new metrics)
    :param top_k: int, truncate the prediction
    :return:  confusion_matrix -> List[List[int]],
    Dict{'precision' -> Float, 'recall' -> Float, 'micro_f1' -> Float, 'macro_f1' -> Float}
    """
    assert len(epoch_predicts) == len(epoch_labels), 'mismatch between prediction and ground truth for evaluation'
    # label2id = vocab.v2i['label']
    # id2label = vocab.i2v['label']
    # epoch_gold_label = list()
    # # get id label name of ground truth
    # for sample_labels in epoch_labels:
    #     sample_gold = []
    #     for label in sample_labels:
    #         assert label in id2label.keys(), print(label)
    #         sample_gold.append(id2label[label])
    #     epoch_gold_label.append(sample_gold)

    epoch_gold = epoch_labels

    # initialize confusion matrix
    confusion_count_list = [[0 for _ in range(len(id2label))] for _ in range(len(id2label))]
    right_count_list = [0 for _ in range(len(id2label))]
    gold_count_list = [0 for _ in range(len(id2label))]
    predicted_count_list = [0 for _ in range(len(id2label))]

    pred_labels: List[List[int]] = list()

    for sample_predict, sample_gold in zip(epoch_predicts, epoch_gold):
        if as_sample:
            sample_predict_id_list = sample_predict
        else:
            np_sample_predict = np.array(sample_predict, dtype=np.float32)
            sample_predict_descent_idx = np.argsort(-np_sample_predict)
            sample_predict_id_list = []
            if top_k is None:
                top_k = len(sample_predict)
            for j in range(top_k):
                if np_sample_predict[sample_predict_descent_idx[j]] > threshold:
                    sample_predict_id_list.append(sample_predict_descent_idx[j])

        for i in range(len(confusion_count_list)):
            for predict_id in sample_predict_id_list:
                confusion_count_list[i][predict_id] += 1

        # count for the gold and right items
        for gold in sample_gold:
            gold_count_list[gold] += 1
            for label in sample_predict_id_list:
                if gold == label:
                    right_count_list[gold] += 1

        # count for the predicted items
        for label in sample_predict_id_list:
            predicted_count_list[label] += 1

        pred_labels.append(sample_predict_id_list)

    precision_dict = dict()
    recall_dict = dict()
    fscore_dict = dict()
    right_total, predict_total, gold_total = 0, 0, 0

    for i, label in id2label.items():
        label = label + '_' + str(i)
        precision_dict[label], recall_dict[label], fscore_dict[label] = _precision_recall_f1(right_count_list[i],
                                                                                             predicted_count_list[i],
                                                                                             gold_count_list[i])
        right_total += right_count_list[i]
        gold_total += gold_count_list[i]
        predict_total += predicted_count_list[i]

    # Macro-F1
    precision_macro = sum([v for _, v in precision_dict.items()]) / len(list(precision_dict.keys()))
    recall_macro = sum([v for _, v in recall_dict.items()]) / len(list(precision_dict.keys()))
    macro_f1 = sum([v for _, v in fscore_dict.items()]) / len(list(fscore_dict.keys()))
    # Micro-F1
    precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
    recall_micro = float(right_total) / gold_total
    micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0.0

    # Transform labels into required format for hierarchical metrics (samples, labels)
    multi_label_binarizer = MultiLabelBinarizer()
    y_pred = [[id2label[label] for label in labels] for labels in pred_labels]
    y_true = [[id2label[label] for label in labels] for labels in epoch_labels]
    y_true_bin = multi_label_binarizer.fit_transform(y_true)
    y_pred_bin = multi_label_binarizer.transform(y_pred)

    metrics = compute_metrics(y_true_bin, y_pred_bin, argmax_flag=False, threshold=-1)
    h_metrics = compute_hierarchical_metrics(y_true_bin, y_pred_bin, encoder_dump_or_mapping=multi_label_binarizer,
                                             taxonomy_path=Path(tax_file))

    metrics = {f"{k}_ours": v for k, v in metrics.items()}

    return {'precision': precision_micro,
            'recall': recall_micro,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            **metrics,
            **h_metrics,
            'full': [precision_dict, recall_dict, fscore_dict, right_count_list, predicted_count_list, gold_count_list]}

def evaluate(epoch_predicts, epoch_labels, id2label, tax_file: str, threshold=0.5, top_k=None, as_sample=False):
    """
    :param epoch_labels: List[List[int]], ground truth, label id
    :param epoch_predicts: List[List[Float]], predicted probability list
    :param vocab: data_modules.Vocab object
    :param threshold: Float, filter probability for tagging
    :param tax_file: txt file with label taxonomy (for our new metrics)
    :param top_k: int, truncate the prediction
    :return:  confusion_matrix -> List[List[int]],
    Dict{'precision' -> Float, 'recall' -> Float, 'micro_f1' -> Float, 'macro_f1' -> Float}
    """
    assert len(epoch_predicts) == len(epoch_labels), 'mismatch between prediction and ground truth for evaluation'
    # label2id = vocab.v2i['label']
    # id2label = vocab.i2v['label']
    # epoch_gold_label = list()
    # # get id label name of ground truth
    # for sample_labels in epoch_labels:
    #     sample_gold = []
    #     for label in sample_labels:
    #         assert label in id2label.keys(), print(label)
    #         sample_gold.append(id2label[label])
    #     epoch_gold_label.append(sample_gold)
    if "amazon" in tax_file:
        tax = AmazonTaxonomyParser(tax_file)
        dataset_name = "amz"
    elif "bgc" in tax_file or "wos" in tax_file:   
        tax = BGCParser(tax_file)
        dataset_name = "bgc" if "bgc" in tax_file else "wos"
    tax.parse()  
    _, classes = tax._build_one_hot()
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}

    epoch_gold = epoch_labels

    # initialize confusion matrix
    confusion_count_list = [[0 for _ in range(len(id2label))] for _ in range(len(id2label))]
    right_count_list = [0 for _ in range(len(id2label))]
    gold_count_list = [0 for _ in range(len(id2label))]
    predicted_count_list = [0 for _ in range(len(id2label))]

    pred_labels: List[List[int]] = list()

    for sample_predict, sample_gold in zip(epoch_predicts, epoch_gold):
        if as_sample:
            sample_predict_id_list = sample_predict
        else:
            np_sample_predict = np.array(sample_predict, dtype=np.float32)
            sample_predict_descent_idx = np.argsort(-np_sample_predict)
            sample_predict_id_list = []
            if top_k is None:
                top_k = len(sample_predict)
            for j in range(top_k):
                if np_sample_predict[sample_predict_descent_idx[j]] > threshold:
                    sample_predict_id_list.append(sample_predict_descent_idx[j])

        for i in range(len(confusion_count_list)):
            for predict_id in sample_predict_id_list:
                confusion_count_list[i][predict_id] += 1

        # count for the gold and right items
        for gold in sample_gold:
            gold_count_list[gold] += 1
            for label in sample_predict_id_list:
                if gold == label:
                    right_count_list[gold] += 1

        # count for the predicted items
        for label in sample_predict_id_list:
            predicted_count_list[label] += 1

        pred_labels.append(sample_predict_id_list)

    precision_dict = dict()
    recall_dict = dict()
    fscore_dict = dict()
    right_total, predict_total, gold_total = 0, 0, 0

    for i, label in id2label.items():
        label = label + '_' + str(i)
        precision_dict[label], recall_dict[label], fscore_dict[label] = _precision_recall_f1(right_count_list[i],
                                                                                             predicted_count_list[i],
                                                                                             gold_count_list[i])
        right_total += right_count_list[i]
        gold_total += gold_count_list[i]
        predict_total += predicted_count_list[i]

    # Macro-F1
    precision_macro = sum([v for _, v in precision_dict.items()]) / len(list(precision_dict.keys()))
    recall_macro = sum([v for _, v in recall_dict.items()]) / len(list(precision_dict.keys()))
    macro_f1 = sum([v for _, v in fscore_dict.items()]) / len(list(fscore_dict.keys()))
    # Micro-F1
    precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
    recall_micro = float(right_total) / gold_total
    micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0.0

    # Transform labels into required format for hierarchical metrics (samples, labels)
    multi_label_binarizer = MultiLabelBinarizer()
    y_pred = [[id2label[label] for label in labels] for labels in pred_labels]
    y_true = [[id2label[label] for label in labels] for labels in epoch_labels]
    y_true_bin = transform_manual(class_to_index, y_true, classes)
    y_pred_bin = transform_manual(class_to_index, y_pred, classes)

    # save the binarized labels as csv files
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    results_dir = f"src/models/HBGL/hbgl_results/{dataset_name}/{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    np.savetxt(f"{results_dir}/y_true_bin.csv", y_true_bin, delimiter=",")
    np.savetxt(f"{results_dir}/y_pred_bin.csv", y_pred_bin, delimiter=",")

    metrics = compute_metrics(y_true_bin, y_pred_bin, argmax_flag=False, threshold=-1)
    h_metrics = compute_hierarchical_metrics(y_true_bin, y_pred_bin, encoder_dump_or_mapping=multi_label_binarizer,
                                             taxonomy_path=Path(tax_file))

    metrics = {f"{k}_ours": v for k, v in metrics.items()}

    return {'precision': precision_micro,
            'recall': recall_micro,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            **metrics,
            **h_metrics,
            'full': [precision_dict, recall_dict, fscore_dict, right_count_list, predicted_count_list, gold_count_list]}


def evaluate_seq2seq(batch_predicts, batch_labels, id2label):
    """_summary_

    Args:
        batch_predicts (_type_): one batch of predicted graph e.g [[0,0,1...],[0,1,...]],index is the corresponding label_id
        batch_labels (_type_): _description_ same as top,but the ground true label
        id2label (_type_): _description_

    Returns:
        _type_: _description_ return de micro,macro,precision and recall
    """
    assert len(batch_predicts) == len(batch_labels), 'mismatch between prediction and ground truth for evaluation'
    np_pred, np_labels = np.array(batch_predicts), np.array(batch_labels)
    np_right = np.bitwise_and(np_pred, np_labels)
    # [1]是True的索引,[0]是batch的索引，使用[1]就足够了
    pred_label_id = np.nonzero(np_pred)[1].tolist()
    labels_label_id = np.nonzero(np_labels)[1].tolist()
    right_label_id = np.nonzero(np_right)[1].tolist()

    # initialize confusion matrix
    confusion_count_list = [[0 for _ in range(len(id2label))] for _ in range(len(id2label))]
    right_count_list = [0 for _ in range(len(id2label))]
    gold_count_list = [0 for _ in range(len(id2label))]
    predicted_count_list = [0 for _ in range(len(id2label))]

    for x in pred_label_id: predicted_count_list[x] += 1
    for x in labels_label_id: gold_count_list[x] += 1
    for x in right_label_id: right_count_list[x] += 1

    precision_dict = dict()
    recall_dict = dict()
    fscore_dict = dict()
    right_total, predict_total, gold_total = 0, 0, 0

    for i, label in id2label.items():
        label = label + '_' + str(i)
        precision_dict[label], recall_dict[label], fscore_dict[label] = _precision_recall_f1(right_count_list[i],
                                                                                             predicted_count_list[i],
                                                                                             gold_count_list[i])
        right_total += right_count_list[i]
        gold_total += gold_count_list[i]
        predict_total += predicted_count_list[i]

    # Macro-F1
    precision_macro = sum([v for _, v in precision_dict.items()]) / len(list(precision_dict.keys()))
    recall_macro = sum([v for _, v in recall_dict.items()]) / len(list(precision_dict.keys()))
    macro_f1 = sum([v for _, v in fscore_dict.items()]) / len(list(fscore_dict.keys()))
    # Micro-F1
    precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
    recall_micro = float(right_total) / gold_total
    micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0.0

    return {'precision': precision_micro,
            'recall': recall_micro,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            }
