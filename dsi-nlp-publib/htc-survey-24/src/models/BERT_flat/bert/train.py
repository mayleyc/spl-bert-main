import datetime as dt
from pathlib import Path
from typing import Dict

from src.models.BERT_flat.bert.bert_classifier import BERTForClassification
from src.models.BERT_flat.utility_functions import run_training
from src.utils.generic_functions import load_yaml, get_model_dump_path

# DEBUG PARAMETERS
workers = 0


# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def fn_sigmoid(p, device):
    return p[0].sigmoid().to(device), p[1].to(device)


def fn_softmax(p, device):
    return p[0].softmax(dim=1).to(device), p[1].to(device)


def train_bert4c():
    config_base: Path = Path("config") / "ours"  # / "lr_validation"
    out_base: Path = Path("out") / "bert4c"
    configs = ["bert_ml_config.yml"]

    for c in configs:
        config_path = (config_base / c)
        config: Dict = load_yaml(config_path)
        mod_cls = BERTForClassification
        specific_model = f"{config['name']}_{config['CLF_STRATEGY']}_{config['CLF_STRATEGY_NUM_LAYERS']}"
        print(f"Specific model: {specific_model}")
        print(f"Dataset: {config['dataset']}")
        postprocess_logits = fn_sigmoid if config["multilabel"] else fn_softmax

        kw = dict()
        out_folder = out_base / specific_model / f"run_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        if config["RELOAD"] is True:
            reload_path = Path(config["PATH_TO_RELOAD"])
            kw = dict(split_fun=lambda f: get_model_dump_path(reload_path, f, config.get("EPOCH_RELOAD", None)))
            out_folder = reload_path

        run_training(config_path, config["dataset"], mod_cls, out_folder, workers, validation=config["validation"],
                     logits_fn=postprocess_logits, **kw)


if __name__ == "__main__":
    train_bert4c()
