import json
import os
import random
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Dict, Tuple

import networkx as nx
from sklearn.model_selection import train_test_split
from tqdm import tqdm

BASE_CONFIG = {
    # Base configuration for common hyperparameters
    "model": {
        "embedding": {
            "type": "bert-base-uncased",  # should be set depending on dataset
            "dimension": 768,
            "dropout": 0.5
        },
        "structure_encoder": {
            "dimension": 100,
            "layers": 1,
            "dropout": 0.2  # may change
        },
        "feature_aggregation": {
            "dropout": 0.1
        },
        "capsule_setting": {
            "margin_loss": False,
            "dimension": 32,
            "type": "kde",
            "dropout": 0.15,
            "attention": True,
            "prune": 0.05
        }
    },
    "training": {
        "batch_size": 16,
        "num_epochs": 200,
        "max_text_length": 512,
        "focal_loss": True,
        "recursive_regularization_penalty": 0.0001,
        "label_contradiction_penalty": {
            "weight": 0.0005,  # 0.001,
            "absolute": True,
            "margin": 0.01
        },
        "optimizer": {
            "type": "Adam",
            "learning_rate": 0.0001
        },
        "schedule": {
            "patience": 5,
            "decay": 0.1,
            "early_stopping": 20
        },
        "finetune": {  # NOT FOR WOS
            "tune": True,
            "after": 2,
            "learning_rate": 0.00005
        }
    },
    "path": None,
    "testing": None,
    "device": {
        "cuda": [0],
        "num_workers": 0
    }
}

# Dictionary mapping to the taxonomy paths
tax_dict = {
    "bugs": "data/Bugs/bugs_tax.txt",
    "amazon": "data/Amazon/amazon_tax.txt",
    "bgc": "data/BGC/bgc_tax.txt",
    "rcv1": "data/RCV1/rcv1_tax.txt",
    "wos": "data/WebOfScience/wos_tax.txt"
}


def read_jsonl(file):
    with open(file, mode="r", encoding="utf-8") as tef:
        data = [json.loads(line) for line in tqdm(tef, f"Reading {file}...")]
    return data


def compute_prior(data: List[Dict], g: nx.DiGraph) -> Dict:
    priors = dict()
    for node in g.nodes:
        if node == "root":
            continue
        children = set(g.successors(node))
        if not children:
            continue
        labels = [lab for labs in data for lab in labs["label"] if lab in children]
        counter = Counter(labels)
        total_frequency = sum(counter.values())
        priors[node] = {c: f / total_frequency for c, f in counter.items()}
    return priors


def convert_split(data: list, g: nx.DiGraph) -> Tuple[List[str], List[Dict]]:
    """ Convert data from jsonl format to GACapsHTC format, sorting labels from leaf to root """

    new_data = list()
    new_data_dict = list()
    for line in data:
        labels = line["labels"]
        text = line["text"]
        # Each sample must have a list of labels sorted from leaf to root
        label_rank_dict = dict()
        for label in labels:
            # Find the path from root to label (includes the source and target nodes)
            label_path: List[str] = nx.shortest_path(g, source="root", target=label)
            # label_path = list(reversed(label_path[1:]))  # sort from leaf to root (exclude root)
            # Map each label to its depth and record it in a dictionary
            label_rank = [(lab, rank) for rank, lab in enumerate(label_path)]  # depth, label
            label_rank_dict.update(label_rank)

        # Remove "root" because it should be excluded
        label_rank_dict.pop("root")
        fixed_labels: List[str] = [lab for lab, _ in sorted(label_rank_dict.items(), key=lambda x: x[1], reverse=True)]

        new_data.append(json.dumps(dict(text=text, label=fixed_labels)))
        new_data_dict.append(dict(text=text, label=fixed_labels))

    return new_data, new_data_dict


def convert_dataset(dataset_name: str, dev: float = 0.2):
    if dataset_name == "amazon":
        test_data = read_jsonl("data/Amazon/amazon_test.jsonl")
        train_data = read_jsonl("data/Amazon/amazon_train.jsonl")
        tax_file = "data/Amazon/amazon_tax.txt"
    elif dataset_name == "bgc":
        test_data = read_jsonl("data/BGC/BlurbGenreCollection_EN_test.jsonl")
        train_data = read_jsonl("data/BGC/BlurbGenreCollection_EN_train.jsonl")
        tax_file = "data/BGC/bgc_tax.txt"
    elif dataset_name == "bugs":
        test_data = read_jsonl("data/Bugs/bugs_test.jsonl")
        train_data = read_jsonl("data/Bugs/bugs_train.jsonl")
        tax_file = "data/Bugs/bugs_tax.txt"
    elif dataset_name == "rcv1":
        test_data = read_jsonl("data/RCV1v2/test.jsonl")
        train_data = read_jsonl("data/RCV1v2/train.jsonl")
        tax_file = "data/RCV1v2/rcv1_tax.txt"
    elif dataset_name == "wos":
        test_data = read_jsonl("data/WebOfScience/wos_test.jsonl")
        train_data = read_jsonl("data/WebOfScience/wos_train.jsonl")
        tax_file = "data/WebOfScience/wos_tax.txt"
    else:
        raise ValueError(f"Invalid dataset '{dataset_name}'")

    config = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    dev_data = None
    if dev:
        if dataset_name == "bgc":
            dev_data = read_jsonl("data/BGC/BlurbGenreCollection_EN_dev.jsonl")
        else:
            random.seed(7)
            # Split the data
            train_data, dev_data = train_test_split(train_data, test_size=dev, random_state=7)

    dataset_folder = Path("src") / "models" / "GACapsHTC" / "data" / dataset_name
    if not os.path.exists(dataset_folder):
        dataset_folder.mkdir(parents=True, exist_ok=True)

    dataset_train_name = f"{dataset_name}_train.json"
    dataset_test_name = f"{dataset_name}_test.json"
    dataset_dev_name = f"{dataset_name}_dev.json"

    # Initial config
    config["path"]["checkpoints"] = str(dataset_folder / "checkpoints")
    config["path"]["initial_checkpoint"] = ""
    config["path"]["log"] = {"filename": str(dataset_folder / "log.out"), "level": "DEBUG"}
    # config["path"]["log"] = f"LOG {str(dataset_folder / 'log.out')}"

    # 1. ADJUST TRAINING/DEV/TEST files
    g: nx.DiGraph = nx.read_adjlist(tax_file, nodetype=str, create_using=nx.DiGraph)

    train_data, train_data_d = convert_split(train_data, g)
    dev_data, dev_data_d = convert_split(dev_data, g)
    test_data, test_data_d = convert_split(test_data, g)

    with open(dataset_folder / dataset_train_name, mode="w", encoding="utf-8") as fo:
        fo.write("\n".join(train_data))
    with open(dataset_folder / dataset_dev_name, mode="w", encoding="utf-8") as fo:
        fo.write("\n".join(dev_data))
    with open(dataset_folder / dataset_test_name, mode="w", encoding="utf-8") as fo:
        fo.write("\n".join(test_data))
    config["path"]["data"]["train"] = str(dataset_folder / dataset_train_name)
    config["path"]["data"]["val"] = str(dataset_folder / dataset_dev_name)
    config["path"]["data"]["test"] = str(dataset_folder / dataset_test_name)

    # 2. Name for labels mapping file that will be generated
    config["path"]["data"]["labels"] = str(dataset_folder / "labels_name2id.json")

    # 3. Create prior file with prior frequency for each label
    # config.path.data.prior ----> DICTIONARY OF {PARENT: {CHILD 1: PRIOR 1, CHILD 2: PRIOR 2, ...}}
    prior_path = dataset_folder / "prior.json"
    # EQUAL PRIOR VERSION, compute_prior computes the real priors
    # prior = dict()
    # for line in open(tax_file, mode="r", encoding="utf-8"):
    #     line = line.strip()
    #     labels = line.split(" ")
    #     if "root" in labels:
    #         labels = ["Root", *labels[1:]]
    #     parent, children = labels[0], labels[1:]
    #     fake_prior = 1 / len(children)
    #     prior[parent] = {c: fake_prior for c in children}
    prior = compute_prior(train_data_d + dev_data_d, g)
    with open(prior_path, mode="w", encoding="utf-8") as prior_out:
        json.dump(prior, prior_out, indent=2)
    config["path"]["data"]["prior"] = str(prior_path)

    # 4. Create hierarchy TSV file
    # config.path.data.hierarchy ---> This is a tsv file where each line is written as follows:    <parent_label_name>\t<child_1_label_name>\t<child_2_label_name>...
    hier_path = dataset_folder / "hierarchy.tsv"
    all_lines: List[str] = list()
    for line in open(tax_file, mode="r", encoding="utf-8"):
        line = line.strip()  # remove '\n'
        labels = line.split(" ")
        if "root" in labels:
            labels = ["Root", *labels[1:]]
        all_lines.append("\t".join(labels))
    with open(hier_path, mode="wt", encoding="utf-8") as tax_out:
        tax_out.write("\n".join(all_lines))
    config["path"]["data"]["hierarchy"] = str(hier_path)

    # 5. Adds info for testing
    config["testing"]["taxonomy_file"] = tax_dict[dataset_name]

    # 6. Write everything to JSON, adding to BASE fields
    config_path: Path = dataset_folder.parent.parent / "configs" / f"{dataset_name}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    BASE_CONFIG.update(config)
    with open(config_path, mode="w", encoding="utf-8") as config_o:
        json.dump(BASE_CONFIG, config_o, indent=2)


if __name__ == "__main__":
    convert_dataset("wos")
