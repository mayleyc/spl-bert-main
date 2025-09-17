from pathlib import Path
from typing import Dict, List, Union, Optional

import joblib
import networkx as nx
import numpy as np
import torch
from networkx import read_adjlist, Graph, is_tree, shortest_path_length
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from hierarchy_dict_gen import AmazonTaxonomyParser, BGCParser



def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, argmax_flag: bool = True, threshold=.5) -> Dict[str, float]:
    """
    Computes regular metrics utilizing sklearn.

    :param threshold: threshold in [0, 1] to consider a prediction positive (> t)
    :param y_true: true labels
    :param y_pred: predicted labels
    :param argmax_flag: Whether to apply an "argmax" function over the predictions
    :return: metrics in a dictionary [metric_name]: value
    """
    # Compute metrics with sklearn
    if argmax_flag:
        y_pred = y_pred.argmax(axis=-1)
    # Thresholding
    if threshold != -1:  # NOTE: NEW: Special case for already transformed labels
        y_pred: np.ndarray = np.where(y_pred > threshold, 1, 0).astype(int)
        y_true: np.ndarray = np.where(y_true > threshold, 1, 0).astype(int)
    # Compute metrics
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred,
                                                                   average="macro",
                                                                   zero_division="warn")
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    metrics = {
        "accuracy": acc,
        "macro_f1": fscore,
        "macro_precision": precision,
        "macro_recall": recall
    }
    return metrics


def _assign_labels(index_labels, index_2_label):
    label_list = []
    for indexes in index_labels:
        labels = [index_2_label[int(i)] for i in indexes]
        label_list.append(labels)
    return label_list


def _assign_binarization(labels_2_convert, label_2_index, number_of_classes):
    indicized_labels = [[label_2_index[label] for label in label_list] for label_list in labels_2_convert]
    # multihot_labels = indicized_labels
    multihot_labels = []
    for labels in indicized_labels:
        # manual multihot
        multihot_labels.append(torch.zeros(number_of_classes).scatter(0, torch.Tensor(labels).long(), 1))

    multihot_labels = torch.stack(multihot_labels)
    # batch_size = len(indicized_labels)
    # max_length = max([len(sample) for sample in indicized_labels])
    # aligned_batch_labels = []
    # for sample_label in indicized_labels:
    #     aligned_batch_labels.append(sample_label + (max_length - len(sample_label)) * [sample_label[0]])
    # aligned_batch_labels = torch.Tensor(aligned_batch_labels).long()
    # multihot_labels = torch.zeros(batch_size, number_of_classes).scatter_(1, aligned_batch_labels, 1)

    return multihot_labels


# ************* HIERARCHICAL METRICS *************
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
def compute_hierarchical_metrics(y_true: Union[np.ndarray, List[List[str]]],
                                 y_pred: Union[np.ndarray, List[List[str]]],
                                 taxonomy_path: Path,
                                 encoder_dump_or_mapping: Optional[Union[Path, Dict, MultiLabelBinarizer]],
                                 threshold=.5) -> Dict[str, float]:
    """
    Compute some hierarchical metrics

    :param y_true: ground truth as MHE labels
    :param y_pred: prediction array with same size as 'y_true', with either probabilities or thresholded values
    :param taxonomy_path: path to file that describes the taxonomy tree
    :param encoder_dump_or_mapping: dump to sklearn MultiLabelBinarizer encoder object
    :param threshold: threshold in [0, 1] to consider a prediction positive (> t)
    :return: dictionary with AHC, and hierarchical metrics, as [metric_name]: value
    """
    # NOTE: for BGC we will need to use read_edgelist to read "blurb_edges.txt"
    g: Graph = read_adjlist(taxonomy_path, nodetype=str)
    assert is_tree(g), "Taxonomy file does not define a Tree, and it should :("
    assert y_pred.shape[0] == y_true.shape[0], f"Different size in pred ({y_pred.shape}) and truth ({y_true.shape})"

    if "amazon" in str(taxonomy_path).lower():
        tax = AmazonTaxonomyParser(str(taxonomy_path))
    elif "bgc" in str(taxonomy_path).lower() or "wos" in str(taxonomy_path).lower():   
        tax = BGCParser(str(taxonomy_path))
    tax.parse()  
    _, classes = tax._build_one_hot()

    '''# Convert predictions to str of labels
    if isinstance(encoder_dump_or_mapping, Path) or isinstance(encoder_dump_or_mapping, str):
        lb_encoder: MultiLabelBinarizer = joblib.load(encoder_dump_or_mapping)
        # Duplicated, but ok
        predicted_labels: List[List[str]] = lb_encoder.inverse_transform(np.where(y_pred > threshold, 1, 0).astype(int))
        true_labels: List[List[str]] = lb_encoder.inverse_transform(np.where(y_true > threshold, 1, 0).astype(int))
    elif isinstance(encoder_dump_or_mapping, MultiLabelBinarizer):
        predicted_labels: List[List[str]] = encoder_dump_or_mapping.inverse_transform(
            np.where(y_pred > threshold, 1, 0).astype(int))
        true_labels: List[List[str]] = encoder_dump_or_mapping.inverse_transform(
            np.where(y_true > threshold, 1, 0).astype(int))'''
    # Convert predictions to str of labels
    
    # Duplicated, but ok
    if isinstance(encoder_dump_or_mapping, Path) or isinstance(encoder_dump_or_mapping, str) or isinstance(encoder_dump_or_mapping, MultiLabelBinarizer):
        predicted_labels: List[List[str]] = binary_to_multilabel(classes, y_pred, threshold)
        true_labels: List[List[str]] = binary_to_multilabel(classes, y_true, threshold)
    
    # elif isinstance(encoder_dump_or_mapping, LabelBinarizer):
    #     print(y_pred)
    #     print("---------")
    #     predicted_labels: List[List[str]] = np.where(y_pred > threshold, 1, 0).astype(int)
    #     print(predicted_labels)
    #     true_labels: List[List[str]] = np.where(y_true > threshold, 1, 0).astype(int)
    elif encoder_dump_or_mapping is None:
        # lb_encoder = encoder_dump_or_mapping
        predicted_labels = y_pred
        true_labels = y_true
    else:
        # Convert multilabel HiAGM / HiMATCH encodings back to normal labels
        # nonzero = indices, np.where to convert float throughj threshold, 0 cause tuple, tolist for np array
        predicted_labels = [np.nonzero(x)[0].tolist() for x in np.where(y_pred > threshold, 1, 0)]
        true_labels = [np.nonzero(x)[0].tolist() for x in y_true]
        predicted_labels = _assign_labels(predicted_labels, encoder_dump_or_mapping.i2v['label'])
        true_labels = _assign_labels(true_labels, encoder_dump_or_mapping.i2v['label'])
    ahcs = list()
    for pred_sample, true_labels_sample in zip(predicted_labels, true_labels):
        # AHC
        d = 0.0
        for p in pred_sample:
            d += set_to_label_distance(g, p, true_labels_sample)
        d /= len(true_labels_sample)
        ahcs.append(d)
        # LCA ?

    ahc = np.mean(ahcs)
    h_m = h_multilabel_precision_recall_fscore(true_labels, predicted_labels, taxonomy_path, encoder_dump_or_mapping)
    h_m["ahc"] = ahc
    return h_m

def compute_hierarchical_metrics_old(y_true: Union[np.ndarray, List[List[str]]],
                                 y_pred: Union[np.ndarray, List[List[str]]],
                                 taxonomy_path: Path,
                                 encoder_dump_or_mapping: Optional[Union[Path, Dict, MultiLabelBinarizer]],
                                 threshold=.5) -> Dict[str, float]:
    """
    Compute some hierarchical metrics

    :param y_true: ground truth as MHE labels
    :param y_pred: prediction array with same size as 'y_true', with either probabilities or thresholded values
    :param taxonomy_path: path to file that describes the taxonomy tree
    :param encoder_dump_or_mapping: dump to sklearn MultiLabelBinarizer encoder object
    :param threshold: threshold in [0, 1] to consider a prediction positive (> t)
    :return: dictionary with AHC, and hierarchical metrics, as [metric_name]: value
    """
    # NOTE: for BGC we will need to use read_edgelist to read "blurb_edges.txt"
    g: Graph = read_adjlist(taxonomy_path, nodetype=str)
    assert is_tree(g), "Taxonomy file does not define a Tree, and it should :("
    assert y_pred.shape[0] == y_true.shape[0], f"Different size in pred ({y_pred.shape}) and truth ({y_true.shape})"

    # Convert predictions to str of labels
    if isinstance(encoder_dump_or_mapping, Path) or isinstance(encoder_dump_or_mapping, str):
        lb_encoder: MultiLabelBinarizer = joblib.load(encoder_dump_or_mapping)
        # Duplicated, but ok
        predicted_labels: List[List[str]] = lb_encoder.inverse_transform(np.where(y_pred > threshold, 1, 0).astype(int))
        true_labels: List[List[str]] = lb_encoder.inverse_transform(np.where(y_true > threshold, 1, 0).astype(int))
    elif isinstance(encoder_dump_or_mapping, MultiLabelBinarizer):
        predicted_labels: List[List[str]] = encoder_dump_or_mapping.inverse_transform(
            np.where(y_pred > threshold, 1, 0).astype(int))
        true_labels: List[List[str]] = encoder_dump_or_mapping.inverse_transform(
            np.where(y_true > threshold, 1, 0).astype(int))
 
    # elif isinstance(encoder_dump_or_mapping, LabelBinarizer):
    #     print(y_pred)
    #     print("---------")
    #     predicted_labels: List[List[str]] = np.where(y_pred > threshold, 1, 0).astype(int)
    #     print(predicted_labels)
    #     true_labels: List[List[str]] = np.where(y_true > threshold, 1, 0).astype(int)
    elif encoder_dump_or_mapping is None:
        # lb_encoder = encoder_dump_or_mapping
        predicted_labels = y_pred
        true_labels = y_true
    else:
        # Convert multilabel HiAGM / HiMATCH encodings back to normal labels
        # nonzero = indices, np.where to convert float throughj threshold, 0 cause tuple, tolist for np array
        predicted_labels = [np.nonzero(x)[0].tolist() for x in np.where(y_pred > threshold, 1, 0)]
        true_labels = [np.nonzero(x)[0].tolist() for x in y_true]
        predicted_labels = _assign_labels(predicted_labels, encoder_dump_or_mapping.i2v['label'])
        true_labels = _assign_labels(true_labels, encoder_dump_or_mapping.i2v['label'])
    ahcs = list()
    for pred_sample, true_labels_sample in zip(predicted_labels, true_labels):
        # AHC
        d = 0.0
        for p in pred_sample:
            d += set_to_label_distance(g, p, true_labels_sample)
        d /= len(true_labels_sample)
        ahcs.append(d)
        # LCA ?

    ahc = np.mean(ahcs)
    h_m = h_multilabel_precision_recall_fscore(true_labels, predicted_labels, taxonomy_path, encoder_dump_or_mapping)
    h_m["ahc"] = ahc
    return h_m

def label_distance(g: Graph, label1: str, label2: str) -> int:
    return shortest_path_length(g, source=label1, target=label2)


def set_to_label_distance(g: Graph, label: str, label_set: List[str]) -> int:
    # Return dict {target: dist} for each node in g
    return min([label_distance(g, label, k) for k in label_set])


def _extend_label_set(g: nx.DiGraph, labels: List[List[str]]) -> List[List[str]]:
    """
    Extends the label set with all ancestors of each label

    :param g: label taxonomy tree
    :param labels: set of labels to expand
    :return: the new expanded labels
    """
    labels_extended = list()
    for original_sample_labels in labels:
        extended_sample_labels = set(original_sample_labels)
        for label in original_sample_labels:
            a = nx.ancestors(G=g, source=label)
            extended_sample_labels.update(a)
        if "root" in extended_sample_labels:
            # This is a dummy node, and it's not a real label
            extended_sample_labels.remove("root")
        if "Root" in extended_sample_labels:
            extended_sample_labels.remove("Root")
        labels_extended.append(list(extended_sample_labels))
    return labels_extended

def transform_manual(class_to_index, y: List[List[str]], classes: List[str]) -> np.ndarray:
    mhe_extended = np.zeros((len(y), len(classes)), dtype=int)
    for i, labels in enumerate(y):
        for label in labels:
            if label in class_to_index: # checking a dictionary is faster than checking a list
                mhe_extended[i, class_to_index[label]] = 1
    return mhe_extended


def h_multilabel_precision_recall_fscore(labels_true: List[List[str]], labels_pred: List[List[str]],
                                         taxonomy_path: Path, encoder_dump_or_mapping: Union[Path, Dict]) -> Dict[
    str, float]:
    """
    Compute hierarchical precision, recall and f1-score, as documented in https://arxiv.org/abs/2206.08653

    :param labels_true: ground truth labels (>1) for each sample, as list of strings per sample
    :param labels_pred: predicted labels for each sample, as list of strings
    :param taxonomy_path: path to file with taxonomy
    :param encoder_dump_or_mapping: dump to MultiLabelBinarizer / mapping to i2v, already fitted on training set
    :return: dictionary of H-metrics
    """
    if "amazon" in str(taxonomy_path).lower():
            tax = AmazonTaxonomyParser(str(taxonomy_path))
    elif "bgc" in str(taxonomy_path).lower() or "wos" in str(taxonomy_path).lower():   
        tax = BGCParser(str(taxonomy_path))
    tax.parse()  
    _, classes = tax._build_one_hot()
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    g: nx.DiGraph = read_adjlist(taxonomy_path, nodetype=str, create_using=nx.DiGraph)

    labels_pred_extended: List[List[str]] = _extend_label_set(g, labels_pred)
    labels_true_extended: List[List[str]] = _extend_label_set(g, labels_true)


    if classes is not None:
        mhe_extended_predictions: np.ndarray = transform_manual(class_to_index, labels_pred_extended, classes)
        mhe_extended_truth: np.ndarray = transform_manual(class_to_index, labels_true_extended, classes)
    else:
        mhe_extended_predictions: np.ndarray = _assign_binarization(labels_pred_extended, class_to_index,
                                                                    len(g))  # -1 cause root, new one: no more root
        mhe_extended_truth: np.ndarray = _assign_binarization(labels_true_extended, class_to_index,
                                                              len(g))  # -1 cause root

    # note: this more damage than it helps
    # assert mhe_extended_predictions.shape == mhe_extended_truth.shape, \
    #     f"Different size in ext pred ({mhe_extended_predictions.shape}) and truth ({mhe_extended_truth.shape})"
    precision, recall, f_score, _ = precision_recall_fscore_support(y_true=mhe_extended_truth,
                                                                    y_pred=mhe_extended_predictions, average="macro",
                                                                    zero_division="warn")
    return dict(h_precision=precision, h_recall=recall, h_fscore=f_score)

def h_multilabel_precision_recall_fscore_old(labels_true: List[List[str]], labels_pred: List[List[str]],
                                         taxonomy_path: Path, encoder_dump_or_mapping: Union[Path, Dict]) -> Dict[
    str, float]:
    """
    Compute hierarchical precision, recall and f1-score, as documented in https://arxiv.org/abs/2206.08653

    :param labels_true: ground truth labels (>1) for each sample, as list of strings per sample
    :param labels_pred: predicted labels for each sample, as list of strings
    :param taxonomy_path: path to file with taxonomy
    :param encoder_dump_or_mapping: dump to MultiLabelBinarizer / mapping to i2v, already fitted on training set
    :return: dictionary of H-metrics
    """
    lb_encoder = None
    if isinstance(encoder_dump_or_mapping, Path) or isinstance(encoder_dump_or_mapping, str):
        lb_encoder: MultiLabelBinarizer = joblib.load(encoder_dump_or_mapping)
    elif isinstance(encoder_dump_or_mapping, MultiLabelBinarizer):
        lb_encoder = encoder_dump_or_mapping
    g: nx.DiGraph = read_adjlist(taxonomy_path, nodetype=str, create_using=nx.DiGraph)

    labels_pred_extended: List[List[str]] = _extend_label_set(g, labels_pred)
    labels_true_extended: List[List[str]] = _extend_label_set(g, labels_true)
    if lb_encoder is not None:
        mhe_extended_predictions: np.ndarray = lb_encoder.transform(labels_pred_extended)
        mhe_extended_truth: np.ndarray = lb_encoder.transform(labels_true_extended)
    else:
        mapping = encoder_dump_or_mapping.v2i['label']
        mhe_extended_predictions: np.ndarray = _assign_binarization(labels_pred_extended, mapping,
                                                                    len(g) - 1)  # -1 cause root
        mhe_extended_truth: np.ndarray = _assign_binarization(labels_true_extended, mapping,
                                                              len(g) - 1)  # -1 cause root

    # note: this more damage than it helps
    # assert mhe_extended_predictions.shape == mhe_extended_truth.shape, \
    #     f"Different size in ext pred ({mhe_extended_predictions.shape}) and truth ({mhe_extended_truth.shape})"
    precision, recall, f_score, _ = precision_recall_fscore_support(y_true=mhe_extended_truth,
                                                                    y_pred=mhe_extended_predictions, average="macro",
                                                                    zero_division="warn")
    return dict(h_precision=precision, h_recall=recall, h_fscore=f_score)

# def sphere_influence(g: Graph, label: str, label_set: List[str]):
#     # label: j'
#     # label_set: S
#     # g: L
#     label_dist_from_g: Dict = shortest_path_length(g, source=label)  # dict {target: dist} for each node in g
#     sphere = set()
#     for j in g.nodes:
#         j_S_dist = set_to_label_distance(g, j, label_set)
#         j_label_dist = label_dist_from_g[j]  # label_distance(g, j, label)
#         if j_label_dist == j_S_dist:
#             sphere.add(j)
#     return sphere
