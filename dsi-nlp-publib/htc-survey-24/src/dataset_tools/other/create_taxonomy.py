import json
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def create_taxonomy_match(df: pd.DataFrame, out_file: Path):
    # for lab in set(df["label"].tolist()):
    os.makedirs(out_file.parent, exist_ok=True)
    g: pd.DataFrame = df.groupby(by="label")["flattened_label"].apply(lambda x: list(np.unique(x)))
    g: dict = g.to_dict()
    with open(out_file, mode="w", encoding="utf-8") as f:
        for l, children in g.items():
            f.write(" ".join([l] + children) + os.linesep)


def create_taxonomy_with_root(df: pd.DataFrame, out_file: Path):
    # for lab in set(df["label"].tolist()):
    os.makedirs(out_file.parent, exist_ok=True)
    g: pd.DataFrame = df.groupby(by="label")["flattened_label"].apply(lambda x: list(np.unique(x)))
    g: dict = g.to_dict()
    with open(out_file, mode="w", encoding="utf-8") as f:
        f.write(" ".join(["root"] + list(g.keys())) + os.linesep)
        for l, children in g.items():
            f.write(" ".join([l] + children) + os.linesep)


def create_taxonomy_hi_match(df: pd.DataFrame, out_file: Path):
    os.makedirs(out_file.parent, exist_ok=True)
    g: pd.DataFrame = df.groupby(by="label")["flattened_label"].apply(lambda x: list(np.unique(x)))
    g: dict = g.to_dict()

    child_to_parent = dict()
    for p, children in g.items():
        for c in children:
            parents = child_to_parent.setdefault(c, set())
            parents.add(p)
    child_to_parent = {k: list(v) for k, v in child_to_parent.items()}
    for r in g.keys():
        parents = child_to_parent.setdefault(r, list())
        parents.append("Root")
    with open(out_file.parent / "bugs_prob_child_parent.json", mode="w", encoding="utf-8") as e:
        json.dump(child_to_parent, e)

    with open(out_file, mode="w", encoding="utf-8") as f:
        f.write("\t".join(["Root"] + list(g.keys())) + os.linesep)
        for p, children in g.items():
            f.write("\t".join([p] + children) + os.linesep)


def create_taxonomy_rlhr(df: pd.DataFrame, out_file: Path):
    # for lab in set(df["label"].tolist()):
    os.makedirs(out_file.parent, exist_ok=True)
    g: pd.DataFrame = df.groupby(by="label")["flattened_label"].apply(lambda x: list(np.unique(x)))
    g: dict = g.to_dict()

    tax = [
        {"node": "root", "children": list(g.keys())},
        *[{"node": lab, "children": slabs} for lab, slabs in g.items()]
    ]

    with open(out_file, mode="w", encoding="utf-8") as f:
        for line in tax:
            s = json.dumps(line)
            f.write(s + os.linesep)

    # Write seen labels
    with open(out_file.parent / "seen_labels.txt", mode="w", encoding="utf-8") as f:
        all_labs: List = np.unique(list(g.keys()) + [a for ls in g.values() for a in ls]).tolist()
        for line in all_labs:
            f.write(line + os.linesep)

    with open(out_file.parent / "unseen_labels.txt", mode="w", encoding="utf-8") as b:
        pass
