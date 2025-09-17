import math
from collections import Counter
from typing import List, Tuple, Iterable, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

plt.rcParams['font.family'] = 'Arial'
sns.set(font="Arial")

"""
Visualization tools for datasets
"""


def labels_histogram(*label_columns: Union[List, pd.Series], multilabel: bool = True, top_n: int = None,
                     title: str = None, fig: str = None) -> None:
    """
    Plot histogram of classes in the dataset. Argument is a list of pandas series, which are stacked.

    :param label_columns: series in which each element is a label (or list of)
    :param multilabel: whether the dataset is multilabel or not
    :param top_n: consider only the most frequent N labels
    :param title: title of the plot
    :param fig: where to save the plot
    """
    n_columns: int = len(label_columns)
    if n_columns == 0:
        raise ValueError("At least one array required as input")
    elif n_columns == 1:
        label_column: pd.Series = label_columns[0]
    else:
        label_column: pd.Series = label_columns[0]
        label_columns = label_columns[1:]
        for column in label_columns:
            label_column = label_column.append(column)

    labels = label_column  # [literal_eval(x) if multilabel else x for x in label_column.values]
    all_labels = [label for lbs in labels for label in lbs] if multilabel else labels
    labels_count = Counter(all_labels).most_common(n=top_n)
    # pd.DataFrame(labels_count).to_csv("freq_amz.csv")
    max_for_plot = 110
    for i in range(math.ceil(len(labels_count) / max_for_plot)):
        plt.figure(figsize=(15, 20), dpi=200)
        sns.set_style("white")
        offset = i * max_for_plot
        labs = labels_count[offset:(offset + max_for_plot)]
        y, x = zip(*labs)

        # assert len(x) == len(y)
        # with open(f"{fig}.csv", "w+") as csv_file:
        #     writer = csv.writer(csv_file, delimiter=',')
        #     level_counter = 0
        #     max_levels = len(x)
        #     while level_counter < max_levels:
        #         writer.writerow((y[level_counter], x[level_counter]))
        #         level_counter = level_counter + 1

        # x = [*x, np.mean(x)] # UNCOMMENT TO ADD MEAN BAR
        # y = [*y, "MEAN"]
        ax = sns.barplot(x=pd.Series(x), y=pd.Series(y), log=True,
                         order=[e for _, e in reversed(sorted(zip(x, y), key=lambda rr: rr[0]))])

        # Add number over plot
        for p in ax.patches:
            tx = "%.d" % p.get_width()
            ax.annotate(tx, xy=(p.get_width(), p.get_y() + p.get_height() / 2),
                        xytext=(5, 0), textcoords="offset points", ha="left", va="center")

        ax.set_title(title if title else "Labels with count (log scale)", fontsize=24)
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set(xlabel="Frequency", ylabel="Topics")
        ax.set_xticklabels([])
        # sns.despine(top=True, left=True)
        # plt.xticks([])
        plt.tight_layout()

        if fig:
            plt.savefig(f"{fig}.png", dpi=600)
        plt.show()


def label_count_histogram(*all_labels: Iterable[Iterable[str]], title: str = None, fig: str = None) -> None:
    """
    Barplot to visualize distribution of the number of labels

    :param all_labels: iterable containing the topics
    :param title: plot title
    :param fig: where to save the plot
    """

    labs = [t for labels in all_labels for t in labels]

    c = Counter([len(t) for t in labs])
    data: List[Tuple[int, int]] = sorted(c.items(), reverse=True)
    label_number = [d for d, r in data]
    doc_count = np.array([r for d, r in data])

    plt.figure(figsize=(15, 15))
    plt.xlabel("Number of documents", fontsize=18)
    plt.ylabel("Label count", fontsize=18)
    sns.set_style("darkgrid")
    ax = sns.barplot(x=doc_count, y=label_number, log=True, orient="h", ci=None)

    # Add number over plot
    for p in ax.patches:
        tx = "%.d" % p.get_width()
        ax.annotate(tx, xy=(p.get_width(), p.get_y() + p.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", ha="left", va="center")
    ax.set_title(title if title else "Number of labels per document", fontsize=24)
    plt.tight_layout()
    if fig:
        plt.savefig(f"{fig}.png", dpi=300)
    plt.show()
