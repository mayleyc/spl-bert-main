import os
from pathlib import Path
from typing import Dict, List

from src.dataset_tools.dataset_manager import DatasetManager
from src.models.SVM.flat_svm import run_svm_classifier
from src.training_scripts.script_utils import save_results
from src.utils.generic_functions import load_yaml
from src.utils.metrics import compute_metrics, compute_hierarchical_metrics


def run_training(config: Dict, dataset: str, out_folder: Path):
    ds_manager = DatasetManager(dataset_name=dataset, training_config=config)
    os.makedirs(out_folder, exist_ok=True)
    results = list()
    # Train in splits
    fold_i: int = 0
    for (x_train, y_train), (x_test, y_test) in ds_manager.get_split():
        fold_i += 1
        y_pred, inf_time = run_svm_classifier(x_train, y_train, x_test, config)
        metrics = compute_metrics(y_test, y_pred, False, threshold=-1)
        if config['objective'] == "multilabel":
            h_metrics = compute_hierarchical_metrics(y_test,
                                                     y_pred,
                                                     taxonomy_path=Path(config["taxonomy_path"]),
                                                     encoder_dump_or_mapping=ds_manager.binarizer)
        elif config['objective'] == "multiclass":
            h_metrics = {}
        else:
            raise ValueError

        all_metrics = metrics | h_metrics | {"inf_time": inf_time}  # join
        # Save metric for current fold
        results.append(all_metrics)
    # ---------------------------------------
    save_results(results, out_folder, config)


def run_configuration():
    # Paths
    config_base_path: Path = Path("config") / "SVM"
    output_path: Path = Path("dumps") / "SVM"
    config_list: List = [
        "svm_config_blurb_mc.yml",
        # "svm_config_blurb_ml.yml",

        "svm_config_bugs_mc.yml",
        # "svm_config_bugs_ml.yml",
        #
        "svm_config_rcv1_mc.yml",
        # "svm_config_rcv1_ml.yml",
        #
        "svm_config_wos_mc.yml",
        # "svm_config_wos_ml.yml",
        #
        "svm_config_amazon_mc.yml",
        # "svm_config_amazon_ml.yml",
    ]

    for c in config_list:
        # Prepare configuration
        config_path: Path = (config_base_path / c)
        config: Dict = load_yaml(config_path)
        specific_model = f"SVM"
        print(f"Specific model: {specific_model}")
        print(f"Dataset: {config['dataset']}")
        # Prepare output
        out_folder = output_path / f"{config['dataset']}_{config['objective']}"
        # Train
        run_training(config=config,
                     dataset=config["dataset"],
                     out_folder=out_folder)


if __name__ == "__main__":
    run_configuration()
