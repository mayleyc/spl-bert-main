import datetime as dt
import os
import time
from operator import itemgetter
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.utils.data as td
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import Whitespace
from torchmetrics import F1Score
from tqdm import tqdm

from src.dataset_tools.dataset_manager import DatasetManager
from src.models.BERT_flat.utility_functions import get_loss_function
from src.models.SVM.preprocessing import process_list_dataset
from src.models.XMLCNN.dataset import EmbeddingDataset, load_vectors
from src.models.XMLCNN.xmlcnn import XmlCNN
from src.training_scripts.script_utils import save_results
from src.utils.generic_functions import load_yaml, get_model_dump_path
from src.utils.metrics import compute_metrics, compute_hierarchical_metrics
from src.utils.torch_train_eval.early_stopper import EarlyStopping
from src.utils.torch_train_eval.evaluation import MetricSet
from src.utils.torch_train_eval.trainer import Trainer


def predict(model, data: td.DataLoader):
    model.train(False)
    y_pred = list()
    y_true = list()
    infer_time_batch: int = 0
    with torch.no_grad():
        for i, pred_data in tqdm(enumerate(data), total=len(data)):
            infer_time_start: int = time.perf_counter_ns()
            y_pred_t, y_true_t, *_ = model(pred_data)
            infer_time_end: int = time.perf_counter_ns()
            infer_time_batch += infer_time_end - infer_time_start

            y_len = y_pred_t.sigmoid()  # (bs, categories)

            y_pred.append(y_len.detach().cpu().numpy())
            y_true.append(y_true_t.cpu().numpy())
    return np.concatenate(y_pred), np.concatenate(y_true), infer_time_batch / len(data)


def run_training(config: Dict, train_set: str, out_folder: Path, split_fun=None):
    ds_manager = DatasetManager(dataset_name=train_set, training_config=config)
    os.makedirs(out_folder, exist_ok=True)
    results = list()
    # Train in splits
    fold_i: int = 0
    seeds = load_yaml("config/random_seeds.yml")

    # Create feature vectors
    vectors, vocab = load_vectors(config["EMBEDDINGS"], max_vectors=500000)
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.normalizer = BertNormalizer(strip_accents=True, lowercase=True)
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.enable_padding()
    tokenizer.enable_truncation(max_length=config["NUM_TOKENS"])

    # Trainer config
    model_folder = config["MODEL_FOLDER"]

    for (x_train, y_train), (x_test, y_test), idx_train in ds_manager.get_split_with_indices():
        if config["validation"] is True:
            # Replace test set with validation set. Notice that the test set will be ignored,
            # and never used in validation or training
            x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2,
                                                                random_state=seeds["validation_split_seed"],
                                                                stratify=itemgetter(*idx_train)(
                                                                    ds_manager.labels_for_splitter))
        # x_train = x_train[:40000]
        # y_train = y_train[:40000]
        # x_test = x_test[:40000]
        # y_test = y_test[:40000]
        fold_i += 1

        if split_fun is not None:
            config.update(split_fun(fold_i))

        print("Starting preprocessing ...")
        x_train, x_test = process_list_dataset(x_train, x_test,
                                               remove_garbage=False,
                                               stop_words_removal=config.get("stop_words_removal", True))
        print("Preprocessing complete.\n")
        config["MODEL_FOLDER"] = str(model_folder / f"fold_{fold_i}")
        # Setup model and optimizer
        early_stopper = EarlyStopping(*config["EARLY_STOPPING"].values()) if config["validation"] else None
        num_class = len(ds_manager.binarizer.classes_)

        model_config = config["CLASSIFIER_CONF"]
        model_config["num_classes"] = num_class
        model_config["words_num"] = config["NUM_TOKENS"]
        model_config["words_dim"] = vectors.dim
        model_config["vectors"] = vectors.vectors

        network = XmlCNN(**model_config)
        del model_config["vectors"]

        opt = torch.optim.AdamW(network.parameters(), lr=config["LEARNING_RATE"], weight_decay=config["L2_REG"])
        loss_fn = get_loss_function(ds_manager.binarizer, config)
        metrics = MetricSet({"f1": (
            F1Score(num_classes=num_class, threshold=.5, average="macro"),
            lambda logits_y, **kwargs: (logits_y[0].sigmoid(), logits_y[1])
        )})
        t = Trainer(network, config, loss_fn, opt, early_stopper, metrics=metrics, add_start_time_folder=False)
        # This preprocessing does tokenization and text cleanup
        # Simple filtering is done in read functions

        train_set = EmbeddingDataset(x_train, y_train, tokenizer)
        train_loader = td.DataLoader(train_set, shuffle=True, batch_size=config["BATCH_SIZE"])
        test_set = EmbeddingDataset(x_test, y_test, tokenizer)
        test_loader = td.DataLoader(test_set, shuffle=False, batch_size=config["TEST_BATCH_SIZE"])

        t.train(train_loader, test_loader)

        # TEST the model
        # First reload last improving epoch
        t.load_previous(t.last_saved_checkpoint, model_only=True)
        model = t.model

        y_pred, y_true, inf_time = predict(model, test_loader)
        metrics = compute_metrics(y_true, y_pred, False, threshold=.5)
        h_metrics = compute_hierarchical_metrics(y_true, y_pred,
                                                 encoder_dump_or_mapping=ds_manager.binarizer,
                                                 taxonomy_path=Path(config["taxonomy_path"]),
                                                 threshold=.5)
        all_metrics = metrics | h_metrics | {"inf_fime": inf_time / config["TEST_BATCH_SIZE"]}  # join
        # Save metric for current fold
        results.append(all_metrics)
        # CLEANUP
        del t, model
        torch.cuda.empty_cache()
        # ---------------------------------------
        # Save results at each fold (overwrite)
        save_results(results, out_folder, config)


def run_configuration():
    # Paths
    config_base_path: Path = Path("config") / "XMLCNN"  # / "validation"
    output_path: Path = Path("dumps") / "XMLCNN_last_tests"
    # config_list: List = ["config_bgc.yml", "config_bugs.yml", "config_rcv1.yml", "config_bugs_match.yml"]
    config_list: List = ["config_rcv1_champ.yml", "config_rcv1_match.yml", "config_rcv1.yml"]
    for c in config_list:
        # Prepare configuration
        config_path: Path = (config_base_path / c)
        config: Dict = load_yaml(config_path)
        # config["DEVICE"] = "cpu"
        # config["n_repeats"] = 1
        specific_model = config["name"]
        print(f"Specific model: {specific_model}")
        print(f"Dataset: {config['dataset']}")
        # Prepare output
        out_folder = output_path / specific_model / f"run_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        config["MODEL_FOLDER"] = out_folder
        kw = dict()
        if config["RELOAD"] is True:
            reload_path = Path(config["PATH_TO_RELOAD"])
            kw = dict(split_fun=lambda f: get_model_dump_path(reload_path, f, config.get("EPOCH_RELOAD", None)))
            out_folder = reload_path
        # Train
        run_training(config=config,
                     train_set=config["dataset"],
                     out_folder=out_folder, **kw)


if __name__ == "__main__":
    run_configuration()
