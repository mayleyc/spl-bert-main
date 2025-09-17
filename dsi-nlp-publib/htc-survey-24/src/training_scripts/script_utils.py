import os
from pathlib import Path
from typing import List

import pandas as pd


def save_results(results: List, out_folder: Path, config: dict):
    # Average metrics over all folds and save them to csv
    df_results = pd.DataFrame(results)
    df_results.loc["avg", :] = df_results.mean(axis=0)
    save_name = "results_test"
    save_path = out_folder / "results"
    os.makedirs(save_path, exist_ok=True)
    df_results.to_csv(save_path / (save_name + ".csv"))
    # dump_yaml(config, save_path / (save_name + ".yml"))
