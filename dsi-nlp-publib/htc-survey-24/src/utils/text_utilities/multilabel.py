from collections import Counter
from pprint import pprint
from typing import Set, List, Tuple

import pandas as pd

"""
Functions to clean categories after stratification for multilabel
NOTE: These are utilized for debugging purposes only, to operate on subsets of the datasets
"""


def _clean_categories(to_remove: Set[str], x: List[str], y: List[List[str]]):
    """
    Clean the y set from labels that are in "to_remove". If no label is left it also removes the x example.

    :param to_remove:
    :param x: training data
    :param y: labels
    :return: x and y sets with removed labels
    """
    x_fixed = list()
    y_fixed = list()

    for x_, ys in zip(x, y):
        ys = [y_ for y_ in ys if y_ not in to_remove]
        if ys:
            y_fixed.append(ys)
            x_fixed.append(x_)
    # else no labels, so remove sample
    return x_fixed, y_fixed


def normalize_labels(x_train: List[str], x_test: List[str], y_train: List[List[str]], y_test: List[List[str]]) \
        -> Tuple[pd.Series, pd.Series, List[List[str]], List[List[str]]]:
    """
    Normalize labels across training and test splits, ensuring that exactly the same set of labels is contained in both

    :param x_train: x train split
    :param x_test: x test split
    :param y_train: y train split (labels)
    :param y_test: y test split (labels)
    :return: 4 new splits with some examples removed to fix the number of labels
    """
    # y_train: List[List[str]] = y_train.values.tolist()
    # y_test: List[List[str]] = y_test.values.tolist()

    y_train_cats = set([a for b in y_train for a in b])
    y_test_cats = set([a for b in y_test for a in b])

    x_train_fixed = x_train
    x_test_fixed = x_test
    y_train_fixed = y_train
    y_test_fixed = y_test

    if y_train_cats == y_test_cats:
        print(f"*** Categories are well split! ({len(y_train_cats)} in both splits)")
    else:
        print(f"*** Train set has {len(y_train_cats)} and test/val set has {len(y_test_cats)} categories")
        test_remove: Set = y_test_cats - y_train_cats
        print(f"*** Test/val set has {len(test_remove)} categories that are not in train set")
        train_remove = y_train_cats - (y_test_cats - test_remove)
        print(f"*** Train set has {len(train_remove)} categories that are not in test/val set")
        assert (y_train_cats - train_remove) == (
                y_test_cats - test_remove), "Categories must be the same in train and test/val set"

        print(f"*** Removing {len(train_remove)} categories from training set...")

        x_train_fixed, y_train_fixed = _clean_categories(train_remove, x_train, y_train)

        print(f"*** Removing {len(test_remove)} categories from test/val set...")

        x_test_fixed, y_test_fixed = _clean_categories(test_remove, x_test, y_test)

        label_count_train = Counter([len(ys) for ys in y_train_fixed])
        label_count_test = Counter([len(ys) for ys in y_test_fixed])

        print("*** > Number of label per samples in training after cleaning:")
        pprint(label_count_train)
        print("*** > Number of label per samples in test/val after cleaning:")
        pprint(label_count_test)

    # Convert X data to series for compatibility with preprocessing module
    x_train_fixed, x_test_fixed = pd.Series(x_train_fixed), pd.Series(x_test_fixed)

    return x_train_fixed, x_test_fixed, y_train_fixed, y_test_fixed
