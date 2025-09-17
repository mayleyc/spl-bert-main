from collections import Counter
from typing import List

import pandas as pd

from src.dataset_tools import get_rcv1_split


def get_frequency(split: str):
    if split == "all":
        data_train = get_rcv1_split("train")
        data = get_rcv1_split("test")
        data.extend(data_train)
    else:
        data: List = get_rcv1_split(split)
    c_label = Counter([a for sample in data for a in sample["labels"]])
    pd.DataFrame(c_label.most_common()).to_csv(f"freq_rcv1.csv")


if __name__ == "__main__":
    get_frequency("all")
