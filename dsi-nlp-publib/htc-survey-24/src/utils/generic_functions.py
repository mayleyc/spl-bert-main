import json
import os
import random
import re
import string
from pathlib import Path
from typing import Union, Any, List, Generator, Optional, Pattern, Dict

import yaml


def load_yaml(path: Union[str, Path]) -> Any:
    """
    Load YAML as python dict

    @param path: path to YAML file
    @return: dictionary containing data
    """
    with open(path, encoding="UTF-8") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    return data


def dump_yaml(data, path: Union[str, Path]) -> None:
    """
    Load YAML as python dict

    @param path: path to YAML file
    @param data: data to dump
    @return: dictionary containing data
    """
    with open(path, encoding="UTF-8", mode="w") as f:
        yaml.dump(data, f, Dumper=yaml.SafeDumper)


def load_json(path: Union[str, Path]) -> Any:
    with open(path, mode="r", encoding="UTF-8") as f:
        data = json.load(f)
    return data


def extract_number(f):
    s = re.findall("\d+$", f)
    return int(s[0]) if s else -1, f


def chunk_list(to_chunk: List, chunk_size: int) -> Generator:
    """
    Split list into chunks of chunksize (apart from last, which will be partial
    to the remaining elements). The number of chunks is
    ceil(len(data) / chunksize)

    @param to_chunk    : data that needs to be split into chunks
    @param chunk_size: how much data at most can be in each chunks

    @return: generator of chunks of list
    """
    for start_index in range(0, len(to_chunk), chunk_size):
        yield to_chunk[start_index:start_index + chunk_size]


def load_model_from_path(path: Union[str, Path]) -> "torch.nn.Module":
    """
    Load a model from a configuration dictionary

    :param path: path to the model folder
    :return: the model instance (torch module)
    """
    # Open configuration dictionary
    path = Path(path) / "run_config.json"
    dict_config = load_json(path)
    # Load relevant data to create instance
    class_name = dict_config["model_name"]  # Name of model class
    module_name: str = dict_config["model_class_src"]  # src/model_classes/...
    kwargs = dict_config["kwargs"]

    importlib = __import__("importlib")

    # Import module and instantiate class
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)

    instance = class_(**kwargs)
    return instance


def load_model_weights(model: "torch.nn.Module", weights: Union[str, Path], device=None) -> None:
    """
    Load torch model weights in a model instance. The operation is done in-place.

    :param model: the instance of nn.Module
    :param weights: path to the .pt file with the model parameters
    :param device: optional device to move the loaded model to
    """
    import torch
    model_path = Path(weights)
    # Load model
    args = dict(map_location=device) if device is not None else dict()
    model.load_state_dict(torch.load(model_path, **args))


def simplify_text(s: str):
    s = s.strip()

    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", ' ', s)

    s = s.strip()

    # Replace all runs of whitespace with a single dash
    s = re.sub(r"\s+", '-', s)

    return s


def random_string(length: int = 10):
    s = string.ascii_lowercase + string.digits
    return ''.join(random.choice(s) for _ in range(length))


def print_gpu_obj():
    """
    Print objects on gpu
    """
    import gc
    import torch
    count = 0
    for tracked_object in gc.get_objects():
        if torch.is_tensor(tracked_object):
            if tracked_object.is_cuda:
                count += 1
                print("{} {} {}".format(
                    type(tracked_object).__name__,
                    " pinned" if tracked_object.is_pinned() else "",
                    tracked_object.shape
                ))
    print(f'\nTHERE ARE {count} OBJECTS ON GPU')


def get_dump_for_fold(dump_root: Union[str, Path], regex: Pattern[str]) -> Path:
    model_dump = None
    for root, dirs, files in os.walk(dump_root):
        for folder in dirs:
            if regex.match(folder):
                model_dump = Path(root) / folder
                break
        if model_dump is not None:
            break
    return model_dump


def get_model_dump_path(run_path: Path, fold_n: int, epoch_n: Optional[int] = None) -> Dict[str, str]:
    """
    Get path to model dump for a specific fold and epoch number

    :param run_path: path to run folder containing dumps for every fold
    :param epoch_n: (optional) epoch to load. If none, loads latest
    :param fold_n: fold number
    :return: dictionary containing correct directories for models 2 load
    """
    model_dump_name = get_dump_for_fold(run_path, re.compile(f"fold_{fold_n}.*"))
    if epoch_n is None:
        # Prende le cartelle, prende l'ultima, prende il numero
        epochs = next(os.walk(model_dump_name))[1]
        epoch_n = max([int(x.split("_")[1]) for x in epochs])

    model_path = str(model_dump_name / f"epoch_{epoch_n}")

    return dict(PATH_TO_RELOAD=model_path)


def read_dict_to_list_tree(taxonomy_path: Union[Path, str]):
    hierarchy: Dict = {}
    with open(taxonomy_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            parent, *children = line.split(" ")
            hierarchy[parent] = [c.strip() for c in children]
    return hierarchy
