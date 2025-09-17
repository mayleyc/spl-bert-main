import json
import os
import random
from pathlib import Path
from typing import Union, List

from sklearn.model_selection import train_test_split
from tqdm import tqdm


def read_jsonl(file):
    with open(file, mode="r", encoding="utf-8") as tef:
        data = [json.loads(line) for line in tqdm(tef, f"Reading {file}...")]
    return data


def write_jsonl_hbgl(samples: List[dict], path: Union[Path, str]) -> None:
    data = list()
    mi = 1000
    ma = 0
    # tt = 0
    for sample in samples:
        t = sample["text"]
        l = sample["labels"]
        # if len(t) > tt:
        #     tt = len(t.split(" "))
        if len(l) > ma:
            ma = len(l)
        if len(l) < mi:
            mi = len(l)
        # ss = json.dumps({"doc_token": t, "doc_label": l})
        ss = json.dumps({"doc_token": t, "doc_label": l, "doc_topic": [], "doc_keyword": []})
        data.append(ss)

    print(f"Max len: {ma}\nMin len: {mi}")
    # print(f"Max text len: {tt}")
    # return

    with open(path, mode="w", encoding="utf-8") as fo:
        fo.write("\n".join(data))


def convert_taxonomy(path: Union[Path, str], out_name: str) -> None:
    tax = list()
    for line in open(path, mode="r", encoding="utf-8"):
        tokens = line.strip().split(" ")
        if not tax:
            tokens = [t if t != "root" else "Root" for t in tokens]
        tax.append("\t".join(tokens))
    # return

    with open(out_name, mode="w", encoding="utf-8") as fo:
        fo.write("\n".join(tax))


def convert_dataset(dataset_name: str, dev: float = 0.2):
    if dataset_name == "amazon":
        test_data = read_jsonl("data/Amazon/amazon_test.jsonl")
        train_data = read_jsonl("data/Amazon/amazon_train.jsonl")
        tax_file = "data/Amazon/amazon_tax_old.txt"
    elif dataset_name == "bgc":
        test_data = read_jsonl("data/BGC/BlurbGenreCollection_EN_test.jsonl")
        train_data = read_jsonl("data/BGC/BlurbGenreCollection_EN_train.jsonl")
        tax_file = "data/BGC/bgc_tax_old.txt"
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
        tax_file = "data/WebOfScience/wos_tax_old.txt"
    else:
        raise ValueError(f"Invalid dataset '{dataset_name}'")

    dev_data = None
    if dev:
        if dataset_name == "bgc":
            dev_data = read_jsonl("data/BGC/BlurbGenreCollection_EN_dev.jsonl")
        else:
            random.seed(7)
            train_data, dev_data = train_test_split(train_data, test_size=dev, random_state=7)

    dataset_folder = f"src/models/HBGL/data_ours/{dataset_name}"
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder, exist_ok=True)

    dataset_train_name = f"{dataset_name}_train.json"
    dataset_test_name = f"{dataset_name}_test.json"
    dataset_dev_name = f"{dataset_name}_dev.json"
    write_jsonl_hbgl(train_data, os.path.join(dataset_folder, dataset_train_name))
    write_jsonl_hbgl(test_data, os.path.join(dataset_folder, dataset_test_name))
    if dev:
        write_jsonl_hbgl(dev_data, os.path.join(dataset_folder, dataset_dev_name))

    # taxonomy
    convert_taxonomy(tax_file, out_name=os.path.join(dataset_folder, f"{dataset_name}.taxnomy"))


if __name__ == "__main__":
    #convert_dataset("wos")
    convert_dataset("bgc")
    convert_dataset("amazon")
    
