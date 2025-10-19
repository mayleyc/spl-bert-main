import datetime as dt
from pathlib import Path
from typing import List, Dict

from src.models.BERT_flat.bert.bert_classifier import BERTForClassification
from src.training_scripts.flat.bert import run_training
from src.utils.generic_functions import load_yaml, get_model_dump_path
import time


def run_configuration():
    # Paths
    config_base_path: Path = Path("config") / "BERT"
    output_path: Path = Path("dumps") / "BERT_MATCH"
    config_list: List = ["bert_bgc_match.yml", "bert_amz_match.yml", "bert_wos_match.yml"] #  , "bert_rcv1_match.yml", "bert_bgc_match.yml", 

    for c in config_list:
        # Prepare configuration
        config_path: Path = (config_base_path / c)
        config: Dict = load_yaml(config_path)
        specific_model = f"{config['name']}_{config['CLF_STRATEGY']}_{config['CLF_STRATEGY_NUM_LAYERS']}"
        print(f"Specific model: {specific_model}")
        print(f"Dataset: {config['dataset']}")
        # Prepare output
        out_folder = output_path / specific_model / f"run_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        kw = dict()
        if config["RELOAD"] is True:
            reload_path = Path(config["PATH_TO_RELOAD"])
            kw = dict(split_fun=lambda f: get_model_dump_path(reload_path, f, config.get("EPOCH_RELOAD", None)))
            out_folder = reload_path

        # Train
        run_training(config=config,
                     dataset=config["dataset"],
                     model_class=BERTForClassification,
                     out_folder=out_folder, **kw)


if __name__ == "__main__":
    start_time = time.time()
    run_configuration()
    print("--- %s seconds ---" % (time.time() - start_time))
