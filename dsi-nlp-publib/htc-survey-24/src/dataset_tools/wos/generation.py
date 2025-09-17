import json
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

_output_data = Path("data") / "WebOfScience" / "samples.jsonl"
_train_data = Path("data") / "WebOfScience" / "wos_train.jsonl"
_test_data = Path("data") / "WebOfScience" / "wos_test.jsonl"

_tax_data = Path("data") / "WebOfScience" / "wos_tax.txt"

_raw_folder = Path("data") / "raw" / "WebOfScience"


def read_wos_dataset() -> List:
    with open(_output_data, mode="r") as trf:
        data = [json.loads(line) for line in tqdm(trf, desc="Reading WoS", total=46960)]
    assert len(data) == 46960
    print(f"Len {len(data)} samples")
    return data


def generate_wos_dataset() -> None:
    _output_data.parent.mkdir(parents=True, exist_ok=True)
    file_metadata = _raw_folder / "Meta-data" / "Data.xlsx"
    df = pd.read_excel(file_metadata, index_col=False, usecols=["Domain", "area", "Abstract"], engine="openpyxl")

    df["L1"] = df["Domain"].map(lambda x: re.sub(r"\s+", "-", re.sub(r"[^\w\s]", "", x.strip())).strip())
    df["area"] = df["area"].map(lambda x: re.sub(r"\s+", "-", re.sub(r"[^\w\s]", "", x.strip())).strip())
    df["text"] = df["Abstract"].map(lambda x: re.sub(r"\s+", " ", x.strip()).strip())
    df["L2"] = df["L1"].str.cat(df["area"].values, sep="_")
    df = df[["text", "L1", "L2"]]

    # Remove 7 categories that appear less than 10 times (25 documents overall)
    labels_to_remove: List[str] = [k for k, v in Counter(df["L2"].tolist()).items() if v < 10]
    df["L2"] = df["L2"].replace(to_replace=labels_to_remove, value=np.nan)
    df = df.dropna()
    assert df.shape[0] == 46985 - 25

    # Generate tax file
    tax_levels: pd.DataFrame = df[["L1", "L2"]].groupby("L1").agg(["unique"])
    taxonomy: Dict = dict(zip(tax_levels.index.tolist(), [a[0].tolist() for a in tax_levels.values.tolist()]))
    with open(_tax_data, "w", encoding="utf-8") as tax_f:
        tax_f.write(f"root {' '.join(taxonomy.keys())}\n")
        for par, children in taxonomy.items():
            tax_f.write(f"{par} {' '.join(children)}\n")

    # Generate JSONL file
    with open(_output_data, "w", encoding="utf-8") as f:
        for index, row in df.iterrows():
            line_s = json.dumps({"text": row["text"], "labels": [row["L1"], row["L2"]]})
            f.write(f"{line_s}\n")

def merge_jsonl_files(input_files: List[Path]) -> None:
    _output_data.parent.mkdir(parents=True, exist_ok=True)
    with open(_output_data, "w", encoding="utf-8") as out_f:
        for input_file in input_files:
            with open(input_file, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    out_f.write(line)

if __name__ == "__main__":
    merge_jsonl_files([_train_data, _test_data])
    #generate_wos_dataset()
    # read_wos_dataset()
