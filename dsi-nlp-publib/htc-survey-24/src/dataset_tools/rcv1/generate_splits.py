import datetime as dt
import json
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import List, Set, Tuple, Dict
from xml.dom.minidom import parse

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import regex as re
from tqdm import tqdm

from src.dataset_tools.other.multilabel_visualizer import labels_histogram, label_count_histogram
from src.dataset_tools.rcv1.utils import ReuterArticle, parse_newsitem

_test_file = Path("data") / "RCV1v2" / "test.jsonl"
_train_file = Path("data") / "RCV1v2" / "train.jsonl"


class BaseRCV:
    """
    Generation utility for RCV1 dataset, using RCV1-v2 splits
    """

    def __init__(self, data_folder: Path):
        """
        Initialize RCV2 dataset generator. Convert from XML to a pickle for convenience's sake.

        Parameters
        ----------
        path : path that contains the REUT_LAN_NUM folders
        """

        # Folder containing the numbered sub-folders
        self.__folder_path: Path = data_folder
        # List of articles formatted
        self.__articles: List[ReuterArticle] = list()
        self.__hierarchy_file = data_folder / "rcv1.topics.hier.expanded"
        self.__documents_file = data_folder / "rcv1v2-ids.dat"
        self.__topics_file = data_folder / "rcv1-v2.topics.qrels"
        self.__raw_dataset_folder = data_folder / "rcv1"
        self.__labels_graph: nx.DiGraph = None
        self.__topics: Dict[int, List[str]] = dict()

        self.NUM_CLASSES = 0

    def _read_hierarchy(self) -> None:
        cg = nx.DiGraph()
        with open(self.__hierarchy_file, "r", encoding="utf-8") as cats:
            for line in cats:
                p, c = re.findall("^parent:\\s+([a-zA-Z0-9]+)\\s+child:\\s+([a-zA-Z0-9]+)\\s+", line)[0]
                if p != "None" and p not in cg.nodes:
                    cg.add_node(p)
                if c not in cg.nodes:
                    cg.add_node(c)
                if p != "None":
                    cg.add_edge(p, c)
        # pos = nx.spring_layout(cg, k=.4) # prog="dot"
        # pos = nx.nx_agraph.graphviz_layout(cg, prog="dot", root="Root", args='-Gnodesep="2.0"')
        # nx.draw(cg, with_labels=True, pos=pos, node_size=500, font_size=2)
        # plt.tight_layout()
        # plt.savefig("labels.png", dpi=1200)
        assert nx.is_tree(cg), "Categories must form a tree"
        self.__labels_graph = cg

    def _read_documents(self) -> None:
        """
        Generate articles from XML files, organized in the sub-folders as provided
        by NIST (e.g. raw/RCV1V2/rcv1/[0-9]+/<files>)
        Transform into a pickle for convenience.
        """
        # Prepare or check for pickle file
        pickle_path = self.__folder_path / "PICKLE"
        pickle_path.mkdir(exist_ok=True)

        # Read v2 IDS
        with open(self.__documents_file, mode="r", encoding="utf-8") as ids_f:
            self.ids: Set[int] = set(map(int, ids_f.readlines()))
        print(f"Found {len(self.ids)} for RCV1-v2")

        # If pickled file doesn't exist, create it and move on
        if not (pickle_path / "rcv1v2.pkl").exists():
            subfolders: List[str] = os.listdir(self.__raw_dataset_folder)
            paths: List[Path] = [self.__raw_dataset_folder / sub for sub in subfolders if
                                 sub not in {"codes", "dtds", "MD5SUMS"}]
            # Parse XML files
            for path in tqdm(paths, desc="Processing folders..."):
                for xml_file in path.iterdir():
                    dom = parse(str(xml_file))
                    newsarticle: ReuterArticle = parse_newsitem(dom)
                    if newsarticle and newsarticle.itemid in self.ids and newsarticle.topics:
                        self.__articles.append(newsarticle)
            # Save as pickle
            with open(pickle_path / "rcv1v2.pkl", "wb") as f:
                pickle.dump(self.__articles, f)
            # Sanity check
            with open(pickle_path / "rcv1v2.pkl", "rb") as f:
                pickled = pickle.load(f)
            assert pickled == self.__articles, "Error: Pickled data is not equal to the one on memory"
        # If pickled file exists, just load that
        else:
            print("[INFO] Found pickled file, loading that instead.")
            with open(pickle_path / "rcv1v2.pkl", "rb") as f:
                self.__articles = pickle.load(f)

        assert len(self.ids) == len(
            self.__articles), f"There should be {len(self.ids)} in RCV1-v2, but only {len(self.__articles)} have been read"

    def _read_topics(self):
        """
        Associate topics with documents, using the updated "rcv1-v2.topics.qrels" file.
        Topics are ordered by depth (level 1 labels first, up to level 6)
        """
        labels: Dict[int, List[Tuple[str, int]]] = dict()
        with open(self.__topics_file, "r", encoding="utf-8") as topics_file:
            for line in topics_file:
                # E11 2286 1
                cat, d_id = re.findall("^([a-zA-Z0-9]+)\\s(\\d+)\\s\\d$", line)[0]
                d_id = int(d_id)
                assert cat in self.__labels_graph.nodes, f"Category {cat} not present in labels graph"
                assert d_id in self.ids, f"Document id={d_id} is not present in official list of RCV1v2 codes"
                depth: int = nx.shortest_path_length(self.__labels_graph, source="Root", target=cat)
                labels.setdefault(d_id, list()).append((cat, depth))
        self.__topics = {k: [c for c, _ in sorted(v, key=lambda x: x[1])] for k, v in labels.items()}

        # Prune label graph of nodes that are not used (13 codes mentioned in §3.2 of Lewis2004)
        unassigned_labels = set(self.__labels_graph.nodes) - {a for t in self.__topics.values() for a in t}
        unassigned_labels.remove("Root")
        for l in unassigned_labels:
            p = list(self.__labels_graph.predecessors(l))[0]
            for c in self.__labels_graph.successors(l):
                self.__labels_graph.add_edge(p, c)
            self.__labels_graph.remove_node(l)
        assert nx.is_tree(self.__labels_graph) and nx.is_directed_acyclic_graph(self.__labels_graph), "Not good"

        # Print number of nodes per-level (just to get stats)
        depths = [nx.shortest_path_length(self.__labels_graph, "Root", n) for n in self.__labels_graph.nodes]
        c = Counter(depths)
        print(c.most_common())

        # Print the graph to check
        pos = nx.nx_agraph.graphviz_layout(self.__labels_graph, prog="dot", root="Root", args='-Gnodesep="2.0"')
        nx.draw(self.__labels_graph, with_labels=True, pos=pos, node_size=500, font_size=2)
        plt.show()

        nx.write_adjlist(self.__labels_graph, "data/RCV1v2/rcv1_tax.txt")

    def _make_split(self, normalize_labels: bool = False):
        """
        Make the RCV1v2 split, if necessary forcing the set of labels in train and test split to be the same.
        :param normalize_labels: whether labels in training set should be the same that are in test set
        """
        train_samples: List[Dict] = list()
        test_samples: List[Dict] = list()
        train_labs: Set[str] = set()
        test_labs: Set[str] = set()
        train_range = dt.date(1996, 8, 20), dt.date(1996, 8, 31)
        lens = 0.0
        for article in self.__articles:
            labs = self.__topics[article.itemid]
            s = {"text": f"{article.headline} {article.article}", "labels": labs, "id": article.itemid}
            lens += len(s["text"].strip())
            if train_range[0] <= article.date <= train_range[1]:
                train_samples.append(s)
                train_labs |= set(labs)
            else:
                test_samples.append(s)
                test_labs |= set(labs)

        print(f"Avg len of whole dataset: {lens / len(self.__articles)}")
        print(f"Found {len(train_labs)} labels in training, and {len(test_labs)} in testing")

        train_strings = list()
        test_strings = list()
        if normalize_labels:
            to_remove_labels = (train_labs | test_labs) - (train_labs & test_labs)
            train_labs -= to_remove_labels
            test_labs -= to_remove_labels
            print(f"Removing labels: [{', '.join(list(to_remove_labels))}]")
            if to_remove_labels:
                removed_doc_count = 0
                for sample in train_samples:
                    text, topics, a_id = sample.values()
                    labels = [t for t in topics if t not in to_remove_labels]
                    self.__topics[a_id] = labels
                    if labels:
                        s = json.dumps({"text": text, "labels": labels}) + "\n"
                        train_strings.append(s)
                    else:
                        removed_doc_count += 1
                print(f"Removed {removed_doc_count} documents from training set to normalize labels.")
                removed_doc_count = 0
                for sample in test_samples:
                    text, topics, a_id = sample.values()
                    labels = [t for t in topics if t not in to_remove_labels]
                    self.__topics[a_id] = labels
                    if labels:
                        s = json.dumps({"text": text, "labels": labels}) + "\n"
                        test_strings.append(s)
                    else:
                        removed_doc_count += 1
                print(f"Removed {removed_doc_count} documents from testing set to normalize labels.")
                print(f"Now there are {len(train_labs)} labels in training, and {len(test_labs)} in testing")

                # NOTE: if some nodes are removed is it a problem for hierarchical metrics?
        else:
            train_strings = [json.dumps({"text": o["text"], "labels": o["labels"]}) + "\n" for o in train_samples]
            test_strings = [json.dumps({"text": o["text"], "labels": o["labels"]}) + "\n" for o in test_samples]

        _train_file.parent.mkdir(parents=True, exist_ok=True)
        _test_file.parent.mkdir(parents=True, exist_ok=True)
        with open(_train_file, "w", encoding="utf-8") as f:
            f.writelines(train_strings)
            print(f"Written {len(train_samples)} objects to training set file")
        with open(_test_file, "w", encoding="utf-8") as f:
            f.writelines(test_strings)
            print(f"Written {len(test_samples)} objects to testing set file")

    def generate(self, plot: bool = False, normalize_labels: bool = False) -> None:
        """
        Generate, extract topics, prune small categories and save to FastText compliant file (.txt)
        :param plot to plot label statistics in various steps
        :param normalize_labels: whether labels in training set should be the same that are in test set
        """
        self._read_hierarchy()
        # Populates the "self.articles" by either parsing XML files or loading the pickled file
        self._read_documents()
        # Read categories for each document
        self._read_topics()
        # Split train/test file
        self._make_split(normalize_labels)

        if plot:
            labels: List[List[str]] = list(self.__topics.values())
            topics, counts = np.unique(np.concatenate(labels), return_counts=True)
            labels_histogram(labels, multilabel=True, title=f"RCV1v2 topics frequency ({len(topics)}, log scale)")
            label_count_histogram(labels, title="RCV1v2 number of topics per document (log scale)")


def get_rcv1_split(split: str) -> List:
    if split == "train":
        if not _train_file.exists():
            raise ValueError("Training set not present, you need to run 'dataset_tools/rcv1/generate_splits.py'")
        with open(_train_file, mode="r") as trf:
            data = [json.loads(line) for line in tqdm(trf, f"Reading RCV1-v2 ({split})", total=23149)]
    elif split == "test":
        if not _test_file.exists():
            raise ValueError("Testing set not present, you need to run 'dataset_tools/rcv1/generate_splits.py'")
        with open(_test_file, mode="r") as tef:
            data = [json.loads(line) for line in tqdm(tef, f"Reading RCV1-v2 ({split})", total=781265)]
    else:
        raise ValueError(f"Unsupported split name {split}. Can only use 'train' or 'test'.")
    return data


if __name__ == "__main__":
    d = BaseRCV(Path("data/raw/RCV1v2"))
    d.generate(plot=True, normalize_labels=True)
