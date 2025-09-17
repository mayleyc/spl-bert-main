import json
import os
from operator import itemgetter
from pathlib import Path
from typing import Dict

from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from tqdm import tqdm

from src.dataset_tools import read_bugs
from src.models.HiAGM.construct_label_desc import get_label_embedding
from src.models.HiAGM.data_modules.preprocess import preprocess_line
from src.models.HiAGM.data_modules.vocab import Vocab
from src.models.HiAGM.helper.hierarchy_tree_statistic import generate_tree_stat
from src.utils.generic_functions import load_yaml


def prepare_dataset(config: Dict):
    data_cnf = config["data"]
    if data_cnf["dataset"] == "bugs":
        df, _ = read_bugs(return_df=True)
    else:
        raise ValueError

    # Transform your dataset to json format file {'token': List[str], 'label': List[str]}
    corpus_data = list()
    for [doc, lab, slab] in df[["message", *data_cnf["labels"]]].values.tolist():
        sample_tokens = preprocess_line(doc)
        sample_tokens["label"] = [lab, slab]
        # raw_data.append({'token': line.rstrip(), 'label': []})
        corpus_data.append(json.dumps(sample_tokens))

    # Split and write file
    seeds = load_yaml("config/random_seeds.yml")
    splitter = RepeatedStratifiedKFold(n_splits=config["stratifiedCV"], n_repeats=config["n_repeats"],
                                       random_state=seeds["stratified_fold_seed"])
    strat_labels = df[data_cnf["labels"][1]].tolist()

    for train_index, test_index in splitter.split(corpus_data, strat_labels):
        # fold_i += 1
        # print(f"Fold {fold_i}/{fold_tot * repeats} ({fold_tot} folds * {repeats} repeats)")

        data_train, data_test = list(itemgetter(*train_index)(corpus_data)), list(itemgetter(*test_index)(corpus_data))

        if data_cnf["val_file"] is not None:
            # Replace test set with validation set. Notice that the test set will be ignored,
            # and never used in validation or training
            val_labels = list(itemgetter(*train_index)(strat_labels))
            data_train, data_val = train_test_split(data_train, test_size=0.2,
                                                    random_state=seeds["validation_split_seed"],
                                                    stratify=val_labels)
        else:
            data_val = list()

        yield data_train, data_test, data_val


def write_split(config, data_train, data_test, data_val):
    data_cnf = config["data"]

    train_file = Path(data_cnf["data_dir"]) / data_cnf["train_file"]
    test_file = Path(data_cnf["data_dir"]) / data_cnf["test_file"]
    val_file = Path(data_cnf["data_dir"]) / data_cnf["val_file"]
    os.makedirs(train_file.parent, exist_ok=True)
    os.makedirs(test_file.parent, exist_ok=True)
    os.makedirs(val_file.parent, exist_ok=True)

    with open(train_file, mode="w", encoding="utf-8") as f, open(test_file, mode="w", encoding="utf-8") as g, open(
            val_file, mode="w", encoding="utf-8") as h:
        for line in data_train:
            f.write(line + os.linesep)
        for line in data_test:
            g.write(line + os.linesep)
        for line in data_val:
            h.write(line + os.linesep)

    # Preprocess the taxonomy format (data/wos.taxnomy and data/wos_prob_child_parent.json)
    # Extract Label Prior Probability
    generate_tree_stat(config)

    corpus_vocab = Vocab(config,
                         min_freq=5,
                         max_size=config.vocabulary.max_token_vocab)

    # We use classic TD-IDF to extract the representative words for each label.
    voc_data = config["vocabulary"]
    path_voc = Path(voc_data["dir"])
    get_label_embedding(path_voc / voc_data["label_dict"], train_file, train_file.parent / data_cnf["label_desc_file"])
    return corpus_vocab


def process_entry(text, labels):
    dict_entry: Dict = preprocess_line(text)
    dict_entry["label"] = labels
    return dict_entry


def write_jsonl_split(config, data_train, data_test, data_val):
    data_cnf = config["data"]

    train_file = Path(data_cnf["data_dir"]) / data_cnf["train_file"]
    os.makedirs(train_file.parent, exist_ok=True)

    test_file = Path(data_cnf["data_dir"]) / data_cnf["test_file"]
    os.makedirs(test_file.parent, exist_ok=True)

    val_file = Path(data_cnf["data_dir"]) / data_cnf["val_file"]
    os.makedirs(val_file.parent, exist_ok=True)

    with open(train_file, mode="w", encoding="utf-8") as f, \
            open(test_file, mode="w", encoding="utf-8") as g, \
            open(val_file, mode="w", encoding="utf-8") as h:

        for text, labels in tqdm(zip(*data_train), desc="Preparing train set...", total=len(data_train[0])):
            f.write(json.dumps(process_entry(text, labels)) + os.linesep)
        for text, labels in tqdm(zip(*data_test), desc="Preparing test set...", total=len(data_test[0])):
            g.write(json.dumps(process_entry(text, labels)) + os.linesep)
        for text, labels in tqdm(zip(*data_val), desc="Preparing validation set...", total=len(data_val[0])):
            h.write(json.dumps(process_entry(text, labels)) + os.linesep)

    # Preprocess the taxonomy format (data/wos.taxnomy and data/wos_prob_child_parent.json)
    # Extract Label Prior Probability
    generate_tree_stat(config)

    corpus_vocab = Vocab(config,
                         min_freq=5,
                         max_size=config.vocabulary.max_token_vocab)

    # We use classic TD-IDF to extract the representative words for each label.
    voc_data = config["vocabulary"]
    path_voc = Path(voc_data["dir"])
    get_label_embedding(path_voc / voc_data["label_dict"], train_file, train_file.parent / data_cnf["label_desc_file"])
    return corpus_vocab
