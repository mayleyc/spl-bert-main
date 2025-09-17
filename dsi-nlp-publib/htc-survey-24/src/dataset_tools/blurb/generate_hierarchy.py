import json
import logging
import re
from pathlib import Path
from random import shuffle
from typing import List, Union, Dict, Any
import sys
import os

import networkx as nx
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.dataset_tools.blurb.analysis import analyze_blurb

logging.getLogger().setLevel(logging.DEBUG)
sys.path.append("./src")



_train_file = Path("data") / "BGC" / "BlurbGenreCollection_EN_train.txt"
_test_file = Path("data") / "BGC" / "BlurbGenreCollection_EN_test.txt"
_val_file = Path("data") / "BGC" / "BlurbGenreCollection_EN_dev.txt"

_train_file_l = Path("data") / "BGC" / "BlurbGenreCollection_EN_train.jsonl"
_test_file_l = Path("data") / "BGC" / "BlurbGenreCollection_EN_test.jsonl"
_val_file_l = Path("data") / "BGC" / "BlurbGenreCollection_EN_dev.jsonl"
_tax_path = Path("data") / "BGC" / "bgc_tax.txt"


def write_bgc_jsonl(filepath: Path, data: List):
    with open(filepath, "w") as out_f:
        for s in data:
            out_f.write(f"{json.dumps(s)}\n")


def get_bgc_split_txt(split: str) -> List:
    if split == "train":
        data = parse_books(_train_file)
    elif split == "test":
        data = parse_books(_test_file)
    elif split == "val":
        data = parse_books(_val_file)
    else:
        raise ValueError(f"Unsupported split name {split}. Can only use 'train' or 'test'.")

    data = [{"text": d["text"], "labels": d["labels"]} for d in data]

    return data


def get_bgc_split_jsonl(split: str) -> List:
    if split == "train":
        if not _train_file_l.exists():
            raise ValueError("Training set not present, run 'dataset_tools/blurb/generate_splits/write_jsonl")
        with open(_train_file_l, mode="r", encoding='utf-8') as trf:
            data = [json.loads(line) for line in tqdm(trf, f"Reading BGC ({split})", total=58715)]
    elif split == "test":
        if not _test_file_l.exists():
            raise ValueError("Testing set not present, run 'dataset_tools/blurb/generate_splits/write_jsonl")
        with open(_test_file_l, mode="r", encoding='utf-8') as tef:
            data = [json.loads(line) for line in tqdm(tef, f"Reading BGC ({split})", total=18394)]
    elif split == "val":
        if not _val_file_l.exists():
            raise ValueError("Validation set not present, run 'dataset_tools/blurb/generate_splits/write_jsonl")
        with open(_val_file_l, mode="r", encoding='utf-8') as tef:
            data = [json.loads(line) for line in tqdm(tef, f"Reading BGC ({split})", total=14785)]
    else:
        raise ValueError(f"Unsupported split name {split}. Can only use 'train' or 'test'.")
    return data


def parse_books(file_path: Union[Path, str]) -> List[Dict[str, Any]]:
    """
    Parse a "book" DOM file extracted from the TXT splits.

    :param file_path: training/test/validation file as shared by dataset's authors
    :return list of dictionary with summaries and list of topics
    """

    logging.info(f"Parsing {file_path}...")
    data = list()
    soup = BeautifulSoup(open(file_path, "rt", encoding='utf-8').read(), "html.parser")
    for book in tqdm(soup.find_all("book"), desc="Reading BGC"):
        categories = set()
        per_level_cats = dict()
        book_soup = BeautifulSoup(str(book), "html.parser")
        summary = str(book_soup.find("body").string)
        title = str(book_soup.find("title").string)
        summary = title + ". " + summary
        for t in book_soup.find_all("topics"):
            s1 = BeautifulSoup(str(t), "html.parser")
            structure = ["d0", "d1", "d2", "d3"]
            for i, level in enumerate(structure):
                l_cat = set()
                for t1 in s1.find_all(level):
                    cat = re.sub(r"\s+", "-", re.sub(r"[^\w\s]", "", str(t1.string).strip())).strip()
                    categories.add(cat)
                    l_cat.add(cat)
                per_level_cats[f"d{i}_topics"] = list(l_cat)
        if summary and categories:
            data.append({"text": summary, "labels": list(categories), "flat_label": "_".join(list(categories)),
                         **per_level_cats})
        else:
            logging.warning("Skipping: no summary or category.")

    shuffle(data)
    return data


def generate_edgelist(taxonomy_path: Union[Path, str]) -> None:
    """
    Generate list of edges as txt file.
    Use same format as the provided hierarchy.txt, but with names fixed and root node added.

    :param taxonomy_path: path to hierarchy.txt
    """
    first_level_cats = dict()
    fixed_lines: List[str] = list()
    with open(taxonomy_path, "r") as tax:
        for raw_line in tax:
            clean_line = list()
            for i, n in enumerate(raw_line.split("\t")):
                clean_cat = re.sub(r"\s+", "-", re.sub(r"[^\w\s]", "", str(n).strip())).strip()
                clean_line.append(clean_cat)
                first_level_cats.setdefault(clean_cat, True)
                if i > 0:
                    first_level_cats[clean_cat] = False
            fixed_lines.append("\t".join(clean_line))

    roots = [k for k, v in first_level_cats.items() if v]
    assert len(roots) == 7, f"Found {len(roots)} first-level categories!"

    with open(taxonomy_path.parent / "blurb_edges.txt", "w") as tax:
        for fl in roots:
            tax.write(f"root\t{fl}\n")
        for line in fixed_lines:
            tax.write(f"{line}\n")

    g = nx.read_edgelist(taxonomy_path.parent / "blurb_edges.txt")
    nx.write_adjlist(g, _tax_path)


def test():
    # TO make PLOTS
    my_data = parse_books(_train_file)
    val_set = parse_books(_val_file)
    test_set = parse_books(_test_file)
    my_data.extend(test_set)
    my_data.extend(val_set)
    analyze_blurb(my_data)

    raise NotImplementedError("The hierarchy given by the website is wrong. Disregard this method.")
    generate_edgelist(Path("data") / "raw" / "BGC" / "hierarchy.txt")

    # Verify that the same topics are in all splits
    t_t = {a for d in my_data for a in d["labels"]}
    v_t = {a for d in val_set for a in d["labels"]}
    d_t = {a for d in test_set for a in d["labels"]}
    print(t_t == v_t == d_t)
    print(len(t_t))
    print(len(v_t))
    print(len(d_t))


def write_jsonl():
    # ---------------n"
    _train_file_l.parent.mkdir(parents=True, exist_ok=True)
    data_1 = parse_books(_train_file)
    _data_1 = [{"text": d["text"], "labels": d["labels"]} for d in data_1]
    '''list_data = list()
    for sample in _data_1:
        list_data.extend(sample["labels"])
    write_bgc_jsonl(_train_file_l, list_data)'''
    write_bgc_jsonl(_train_file_l, _data_1)
    # ---------------"
    data_2 = parse_books(_test_file)
    _data_2 = [{"text": d["text"], "labels": d["labels"]} for d in data_2]
    
    '''list_data = list()
    for sample in _data_2:
        list_data.extend(sample["labels"])
    write_bgc_jsonl(_test_file_l, list_data)'''
    write_bgc_jsonl(_test_file_l, _data_2)
    # ---------------
    data_3 = parse_books(_val_file)
    _data_3 = [{"text": d["text"], "labels": d["labels"]} for d in data_3]
    
    '''list_data = list()
    for sample in _data_3:
        list_data.extend(sample["labels"])
    write_bgc_jsonl(_val_file_l, list_data)'''
    write_bgc_jsonl(_val_file_l, _data_3)
    print("")


if __name__ == "__main__":
    # test()

    write_jsonl()
