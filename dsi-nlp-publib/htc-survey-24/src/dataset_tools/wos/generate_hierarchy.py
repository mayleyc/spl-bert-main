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

_train_file_l = Path("data") / "WebOfScience" / "wos_train.jsonl"
_test_file_l = Path("data") / "WebOfScience" / "wos_test.jsonl"
_tax_path = Path("data") / "WebOfScience" / "wos_tax.txt"

def get_wos_split_jsonl(split: str) -> List:
    if split == "train":
        if not _train_file_l.exists():
            raise ValueError("Training set not present")
        with open(_train_file_l, mode="r", encoding='utf-8') as trf:
            data = [json.loads(line) for line in tqdm(trf, f"Reading WOS ({split})", total=31306)]
    elif split == "test":
        if not _test_file_l.exists():
            raise ValueError("Testing set not present")
        with open(_test_file_l, mode="r", encoding='utf-8') as tef:
            data = [json.loads(line) for line in tqdm(tef, f"Reading WOS ({split})", total=15654)]
    else:
        raise ValueError(f"Unsupported split name {split}. Can only use 'train' or 'test'.")
    return data

def get_wos_val(train_data):
    x = [doc['text'] for doc in train_data]
    y = [doc['labels'] for doc in train_data]
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=6262, train_size=25044, shuffle=False) # total train file size is just 31306?
    return X_train, X_val, y_train, y_val
