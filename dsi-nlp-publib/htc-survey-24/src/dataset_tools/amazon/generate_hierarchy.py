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

from sklearn.model_selection import train_test_split

logging.getLogger().setLevel(logging.DEBUG)
sys.path.append("./src")

_train_file_l = Path("data") / "Amazon" / "amazon_train.jsonl"
_test_file_l = Path("data") / "Amazon" / "amazon_test.jsonl"
_full_file_l = Path("data") / "Amazon" / "samples.jsonl"
_tax_path = Path("data") / "Amazon" / "amazon_tax.txt"


def get_amz_split_jsonl(split: str) -> List:
    if split == "train":
        if not _train_file_l.exists():
            raise ValueError("Training set not present")
        with open(_train_file_l, mode="r", encoding='utf-8') as trf:
            data = [json.loads(line) for line in tqdm(trf, f"Reading Amazon ({split})", total=333333)]
    elif split == "test":
        if not _test_file_l.exists():
            raise ValueError("Testing set not present")
        with open(_test_file_l, mode="r", encoding='utf-8') as tef:
            data = [json.loads(line) for line in tqdm(tef, f"Reading Amazon ({split})", total=166667)]
    elif split == "full":
        if not _full_file_l.exists():
            raise ValueError("samples.jsonl not present")
        with open(_full_file_l, mode="r", encoding='utf-8') as tef:
            data = [json.loads(line) for line in tqdm(tef, f"Reading Amazon ({split})", total=500000)]
    else:
        raise ValueError(f"Unsupported split name {split}. Can only use 'train' or 'test'.")
    return data

def get_amz_val(train_data):
    x = [doc['text'] for doc in train_data]
    y = [doc['labels'] for doc in train_data]
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=66667, train_size=266666, shuffle=False)
    return X_train, X_val, y_train, y_val

if __name__ == "__main__":
    train_data = get_amz_split_jsonl("train")
    X_train, X_val, y_train, y_val = get_amz_val(train_data)
    print(X_train[0], y_train[0])
