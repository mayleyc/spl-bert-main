import json
from pathlib import Path
from tqdm import tqdm
from typing import List
import pandas as pd
import numpy as np

#file = Path("data") / "Bugs" / "linux_bugs.csv"
file = Path("data") / "Amazon" / "amazon_train.jsonl"

_output_data = Path("data") / "Amazon" / "samples.jsonl"
#Path("data") / "BGC" / "BlurbGenreCollection_EN_train.txt"
def parse_jsonl(file_path: Path) -> List[dict]:
    with open(file_path, mode="r", encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def search_in_jsonl(file_path: Path, search_term: str):
    
    data = parse_jsonl(file_path)
    search_term_lower = search_term.lower()
    found = False

    for idx, item in enumerate(data):
        for k, v in walk(item):
            if isinstance(v, str) and search_term_lower in v.lower():
                print(f"Match found on line {idx + 1} in field '{k}':")
                print(json.dumps(item, indent=2))
                found = True
                break  # Stop at first match in this item

    if not found:
        print("No matches found.")


def parse_json(file_path: Path) -> dict:
    with open(file_path, mode="r", encoding='utf-8') as f:
        data = json.load(f)
    return data
def parse_txt(file_path: Path) -> List[dict]:
    with open(file_path, mode="r", encoding='utf-8') as f:
        data = f.readlines()
    return data
def parse_csv(file_path: Path, lines: int) -> List[dict]:
    data = pd.read_csv(file_path, sep="\t")
    for i in range(lines):
        print(f"Line {i}: {data.iloc[i].to_dict()}")
def parse_npy(file_path: Path) -> np.ndarray:
    data = np.load(file_path, allow_pickle=True)
    return data

# This function calculates the maximum text length in a JSON-like structure.
def max_text_length(d):
    #return max((len(v) for k, v in walk(d) if k == "text" and isinstance(v, str)), default=0)
    return max((len(v) for k, v in walk(d)), default=0)
def walk(d):
    if isinstance(d, dict):
        for k, v in d.items():
            if k == "text" and isinstance(v, str):
                yield (k, v)
            if isinstance(v, (dict, list)):
                yield from walk(v)
    elif isinstance(d, list):
        for item in d:
            yield from walk(item)

def walk_all(data):
    for entry in data:
        yield from walk(entry)

def max_texts(jsonl_data):
    texts = list(walk_all(jsonl_data))
    if not texts:
        return 0, []
    max_len = max(len(t[1]) for t in texts)
    max_texts = [t[1] for t in texts if len(t[1]) == max_len]
    return max_len, max_texts

# This function merges multiple JSONL files into a single file.
def merge_jsonl_files(input_files: List[Path]) -> None:
    _output_data.parent.mkdir(parents=True, exist_ok=True)
    with open(_output_data, "w", encoding="utf-8") as out_f:
        for input_file in input_files:
            with open(input_file, "r", encoding="utf-8") as in_f:
                for line in tqdm(in_f):
                    out_f.write(line)

if __name__ == "__main__":
    search_in_jsonl(file, "it was interesting to see all the positive one and two word reviews but very few stating")

    #with open(_output_data, "r", encoding="utf-8") as f:
    #    print(f.readline())
    #print(f"Parsed {len(data)} lines from {file}")
    #print(f"Sample book data: \n{data[:100] if data else 'No data available'}")
    #max_length, max_texts = max_texts(data)
    #print(f"Max text length: {max_length}")
    '''if max_texts:
        print(f"Sample max text: {max_texts[:5]}")  # Show first 5 max texts
    else:
        print("No texts found.")'''
    
    print("Done processing book.")
    #data = parse_csv(file, 2)
    #merge_jsonl_files([Path("data") / "Amazon" / "amazon_train.jsonl", Path("data") / "Amazon" / "amazon_test.jsonl"])
