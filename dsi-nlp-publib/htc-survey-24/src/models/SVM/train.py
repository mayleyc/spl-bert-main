from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer

from src.dataset_tools import read_bugs
from src.models.SVM.flat_svm import run_svm_classifier
from src.models.SVM.preprocessing import process_df_dataset
from src.utils.generic_functions import dump_yaml, load_yaml
from src.utils.metrics import compute_metrics


def prepare_train_test_data(dataset: str) -> Tuple[List[List[str]], List[str], List[str]]:
    if dataset == "bugs":
        df, *_ = read_bugs()
        # df = df.head(1000)
    else:
        raise NotImplementedError("Add other datasets")

    print("Starting preprocessing ...")
    # This preprocessing does tokenization and text cleanup as in deeptriage
    # Simple filtering is done in read functions
    data, labels, _, flattened_labels = process_df_dataset(df, return_what="all",
                                                           remove_garbage=True,
                                                           stop_words_removal=True)
    print("Preprocessing complete.\n")
    # We skip sub_labels, use flattened to simplify
    return data, labels, flattened_labels


def classify(tickets: List[List[str]], all_labels: List[List[str]], labels_to_stratify: List[str], config: Dict):
    seeds: Dict = load_yaml(SEEDS_PATH)
    # Variables that handle folds
    fold_i = 0
    fold_tot: int = config["stratifiedCV"]

    # Data and labels
    tickets: np.ndarray = np.array(tickets, dtype=object)

    mlb = MultiLabelBinarizer()
    mlb.fit(all_labels)  # Fit first, transform later
    # Start K-Fold CV
    results: List = list()
    splitter: RepeatedStratifiedKFold = RepeatedStratifiedKFold(n_splits=fold_tot, n_repeats=config['n_repeats'],
                                                                random_state=seeds["stratified_fold_seed"])
    # Operate on each split
    # NOTE: IMPORTANT!! STRATIFIED SPLIT ONLY ON SUBLABELS (can't have everything...)
    for train_index, test_index in splitter.split(tickets, labels_to_stratify):
        fold_i += 1
        print(f"Fold {fold_i}/{fold_tot * config['n_repeats']} ({fold_tot} folds * {config['n_repeats']} repeats)")
        # Assing split indices
        transformed_labels = mlb.transform(all_labels)
        x_train, x_test = tickets[train_index], tickets[test_index]
        y_train, y_test = transformed_labels[train_index], transformed_labels[test_index]
        # Create feature vectors
        # It's already tokenized; we tell it to return identity
        vectorizer = TfidfVectorizer(max_features=config["MAX_FEATURES"], ngram_range=(1, 2))
        # Train the feature vectors
        train_vectors = vectorizer.fit_transform([" ".join(x) for x in x_train])
        test_vectors = vectorizer.transform([" ".join(x) for x in x_test])

        y_pred = run_svm_classifier(train_vectors, y_train, test_vectors, config)
        metrics = compute_metrics(y_test, y_pred, False)

        all_metrics = metrics  # | h_metrics  # join
        # Save metric for current fold
        results.append(all_metrics)

    # Average metrics over all folds and save them to csv
    df = pd.DataFrame(results)
    df.loc["avg", :] = df.mean(axis=0)
    results_folder: Path = Path("out/results/SVM")
    results_folder.mkdir(exist_ok=True, parents=True)
    df.to_csv(results_folder / f"test_results.csv")
    dump_yaml(config, results_folder / f"test_config.yml")


CONFIG_PATH: Path = Path("config/SVM/svm_config.yml")
SEEDS_PATH: Path = Path("config/random_seeds.yml")


def main():
    config: Dict = load_yaml(CONFIG_PATH)
    data, labels, sub_labels = prepare_train_test_data(dataset="bugs")
    ml_labels = [[x, y] for x, y, in zip(labels, sub_labels)]
    classify(data, ml_labels, sub_labels, config)


if __name__ == "__main__":
    main()
