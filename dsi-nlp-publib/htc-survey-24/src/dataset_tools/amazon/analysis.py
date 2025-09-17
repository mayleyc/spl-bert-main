import gzip
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import List, Any, Dict

import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from networkx.drawing.nx_pydot import write_dot
# from networkx.drawing.nx_agraph import write_dot
from tqdm import tqdm

from src.dataset_tools.other.multilabel_visualizer import labels_histogram, label_count_histogram

_samples_path = Path("data") / "Amazon" / "samples.jsonl"


def analyze_amz(data: List[Dict[str, Any]]) -> None:
    summaries = [d["text"] for d in data]
    topics = [d["labels"] for d in data]

    avg_len = np.mean(list(map(str.__len__, summaries)))

    print(f"Average len of reviews: {avg_len}")

    labels_histogram(topics, multilabel=True,
                     title=f"Amz topics frequency ({len([a for f in topics for a in f])}, log scale)")
    label_count_histogram(topics, title="Amz number of topics per document (log scale)")

    for level in range(2):
        # Setup
        title = f"Amazon L{level + 1}-topic distribution"
        sns.set_theme(style="whitegrid")
        f, ax1 = plt.subplots(1, 1, figsize=(30, 50), sharex=False)
        plt.suptitle(f"{title}\n", fontsize=30)

        l_topics: List[str] = [cats[level] for cats in topics if len(cats) > level]
        c_label = Counter(l_topics)

        # Label frequency
        y_label, x_count = zip(*c_label.most_common())
        sns.barplot(x=list(x_count), y=list(y_label), palette="dark", ax=ax1, orient="h")
        ax1.set_ylabel(f"Label (#: {len(c_label.keys())})", fontsize=24)
        ax1.set_xlabel(f"# of reviews ({len(l_topics)})", fontsize=24)
        for bars in ax1.containers:
            ax1.bar_label(bars)

        plt.tight_layout()
        plt.subplots_adjust(
            left=0.1,
            wspace=0.3
        )
        plt.show()


def get_graph():
    """
    Create a dot graph to analyze categories
    """
    tax = nx.DiGraph()
    raw_dump = Path("data") / "raw" / "Amazon" / "meta_Grocery_and_Gourmet_Food.json.gz"

    remove = {
        "chocolate",
        "biscuits",
        "chocolate-chip",
        "assortments-samplers",
        "sandwich",
        "wafers",
        "cold-cereals",
        "fresh-seafood",
        "energy-nutritional",
        "oatmeal",
        "chocolate",
        "peanut-butter",
        "fresh-fish",
        "granola",
        "wheat",
        "flatbread"

    }

    remove_from = {
        "crackers": ["water", "deli"]
    }

    products: Dict[str, List[str]] = dict()
    with gzip.open(raw_dump, "r") as cat_f:
        for product in tqdm(cat_f):
            d = json.loads(product)
            asin: str = d["asin"].strip()
            categories = [
                re.sub(r"\s+", "-", re.sub(r"[^a-zA-Z\s]", "", html.unescape(c.strip()).lower()).strip()).strip() for c
                in d["category"] if
                c.strip() != "</span></span></span>"]
            categories = [c for c in categories if (c not in remove)]

            to_be_pruned = set(categories) & set(remove_from.keys())
            for c in to_be_pruned:
                for cat in remove_from[c]:
                    if cat in categories:
                        categories.remove(cat)

            if categories:
                ex_cats = products.get(asin, None)
                assert ex_cats is None or ex_cats == categories, f"Error: asin '{asin}' duplicated with different categories:\n{ex_cats}\n{categories}"
                products[asin] = categories

    frequent_labels = {a for a, b in Counter([a for cats in products.values() for a in cats]).most_common(200)}
    for categories in products.values():
        prev = "root"
        depth = 0
        for succ in categories:
            depth += 1
            if succ not in frequent_labels or depth > 4:
                continue
            tax.add_edge(prev, succ)
            prev = succ
            assert prev is not None

    print(f"Read {len(products)} products.")

    write_dot(tax, "hier.dot")


if __name__ == "__main__":
    # read file
    samples = list()
    with open(_samples_path, "r") as s:
        for line in s:
            samples.append(json.loads(line))
    analyze_amz(samples)
