from __future__ import annotations
import pandas as pd
#from plotting import dataframe_to_table_image, plot_line_chart
#import math
from pathlib import Path
from typing import Optional, Sequence, Union, Tuple, Dict, Any
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import product
from tqdm import tqdm

from check_pred_y_all import folders

def find_latest_result(folder, pattern="run_*"):
    runs = sorted(Path(folder).glob(pattern), key=os.path.getmtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"No runs found in {folder}")
    latest = runs[0]
    return latest

def find_specific_result(folder, run_name):
    run_path = Path(folder) / run_name
    if not run_path.exists():
        raise FileNotFoundError(f"Run {run_name} not found in {folder}")
    return run_path

def fp_search_htc(dataset, model, folders, hard_enf = False, **abl_args):
    # Validate model
    folder = folders.get(model)
    if folder is None:
        raise ValueError(f"Unknown model: {model}")

    # Dataset → name mappings
    dataset_map = {
        "amz": "AMZ",
        "bgc": "BGC",
        "wos": "WOS",
    }
    dataset_ohe_map = {
        "amz": "amazon",
        "bgc": "bgc",
        "wos": "wos",
    }

    if dataset not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset}")

    dataset_tag_ohe = dataset_ohe_map[dataset]
    dataset_tag = dataset_map[dataset]

    # Build model folder path or file path
    if model == "ohe":
        # Direct CSV file lookup
        filename = f"{dataset_tag_ohe.lower()}_tax_one_hot.csv"
        filepath = Path(folder) / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Expected OHE file not found: {filepath}")
        return [str(filepath)], folder

    # Subfolder BERT models
    subfolder = f"bert_multilabel_{dataset_tag}_concat_cls_3"
    if model == "spl_bert" or model == "spl_bert_h":
        subfolder += "_SPL"
        if model == "spl_bert_h":
            subfolder += f"hyper_S{abl_args['S']}_gates{abl_args['gates']}"
    folder = Path(folder) / subfolder

    if not folder.exists():
        raise FileNotFoundError(f"Model folder not found: {folder}")

    # Find latest run folder by timestamp in name
    if dataset in ["amz", "wos"] and model == "spl_bert":
        if hard_enf == True:
            run_amz = "run_2025-10-24_20-25-20_hard"
            run_wos = "run_2025-10-24_18-29-51_hard"
        else:
            run_amz = "run_2025-10-22_13-22-34"
            run_wos = "run_2025-10-22_11-21-04"
        latest_run = find_specific_result(folder, run_name=run_amz if dataset == "amz" else run_wos)
    else:
        latest_run = find_latest_result(folder) #pattern unused
    
    results_file = latest_run / "results" / "results_test.csv"

    #preds, run = find_specific_run(folder, run_name=f"run_2025-10-24_18-29-51_soft", contains="pred")

    return results_file, str(folder)

def fp_search_vio(dataset, model, hard_enf = False, **abl_args):
    if model == "spl_bert_h":
        vio_csv = f"logs/avg_check_pred_y_all_output_{dataset}_{model}_S{abl_args['S']}_gates{abl_args['gates']}.csv"
    elif hard_enf == True:
        vio_csv = f"logs/avg_check_pred_y_all_output_{dataset}_{model}_hard.csv"
    else:
        vio_csv = f"logs/avg_check_pred_y_all_output_{dataset}_{model}.csv"
    return vio_csv

def read_htc_result_single(result_file: str):
    results = {}
    #header = pd.read_csv(result_file, nrows=0)
    df = pd.read_csv(result_file)
    metrics = [m for m in df.columns.tolist() if m != "Unnamed: 0"]
    for metric in metrics:
        values = df.iloc[:-1][metric].tolist()
        mean_val = df.iloc[-1][metric]
        std_val = np.std(values)
        results[metric] = [mean_val, std_val]
        #print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
    return results
        


    #print(metrics)

def compile_htc_results(result_files: dict):
    for key, file in result_files.items():
        read_htc_result_single(file)

def export_htc_results(dataset, model, folders, S, gates, hard_enf = False):
    result_file, _ = fp_search_htc(dataset=dataset, model=model, folders=folders, hard_enf=hard_enf, S=S, gates=gates)
    compiled_results = {}
    results = read_htc_result_single(result_file)
    #result_filename = os.path.basename(result_file).split('.')[0]
    #compiled_results[result_filename] = results
    return results

def read_vio_results_single(vio_csv: dict):
    results = {}
    #header = pd.read_csv(vio_csv, nrows=0)
    df = pd.read_csv(vio_csv)
    metrics_rough = [m for m in df.columns.tolist() if m not in ['Unnamed: 0', 'dataset', 'model', 'level']]
    means = [m for m in metrics_rough if 'mean' in m]
    stds = [s for s in metrics_rough if 'std' in s]
    #print("Metrics found:", metrics)
    for l in df['level']:
        level_lb = f"l{l}"
        results[level_lb] = {}
        for m, s in zip(means, stds):
            mean = df[df['level'] == l][m].values.tolist()[0]
            std = df[df['level'] == l][s].values.tolist()[0]
            metric = m.replace('_mean', '')
            results[level_lb][metric] = [mean, std]

            #print(f"{level_lb} - {metric}: {results[level_lb][metric]}")
    return results

def export_vio_results(dataset, model, folders, S, gates, hard_enf=False):
    vio_csv = fp_search_vio(dataset=dataset, model=model, folders=folders, hard_enf=hard_enf, S=S, gates=gates)
    compiled_results = {}
    results = read_vio_results_single(vio_csv)
    #result_filename = os.path.basename(result_file).split('.')[0]
    #compiled_results[result_filename] = results
    return results
        

def htc_exp():
    
    #read_htc_result_single("dumps/BERT_SPLHYPER/bert_multilabel_AMZ_concat_cls_3_SPLhyper_S0_gates1/run_2025-10-24_16-26-54/results/results_test.csv")
    S = 0
    gates = 1

    '''#Automate for exp 1 and 2: BERT, BERT-MATCH and SPL-BERT
    print("Exporting HTC results for experiments 1 and 2...")
    for dataset, model in tqdm(product(["amz", "bgc", "wos"], ["bert", "bert_match", "spl_bert"])):
        output_folder = f"dumps/htc_{dataset}_{model}.pkl"
        results = export_htc_results(dataset=dataset, model=model, folders=folders, S=S, gates=gates)
        pd.to_pickle(results, output_folder)
        print(f"Exported HTC results for {dataset} {model} to {output_folder}")
    '''
    '''#Automate for exp 3: SPL-BERT-HYPER with different S and gates
    print("Exporting HTC results for experiment 3 (SPL-BERT-HYPER)...")
    for dataset in tqdm(["amz", "bgc", "wos"]):
        for S, gates in tqdm(product([0, 4], [1, 3])):
            output_folder = f"dumps/htc_{dataset}_spl_bert_hyper_S{S}_gates{gates}.pkl"
            results = export_htc_results(dataset=dataset, model="spl_bert_h", folders=folders, S=S, gates=gates)
            pd.to_pickle(results, output_folder)
            print(f"Exported HTC results for {dataset} SPL-BERT-HYPER S={S} gates={gates} to {output_folder}")'''
    #Automate for exp 3 hard enforcement
    print("Exporting HTC results for experiments 3 on hard enforcement...")
    for dataset, model in tqdm(product(["amz", "wos"], ["spl_bert"])):
        hard_enf = True
        output_folder = f"dumps/htc_{dataset}_{model}_hard.pkl"
        results = export_htc_results(dataset=dataset, model=model, folders=folders, S=S, gates=gates, hard_enf=hard_enf)
        pd.to_pickle(results, output_folder)
        print(f"Exported HTC results for {dataset} {model} to {output_folder}")
def vio_exp():
    S = 0
    gates = 1
    '''#Automate for exp 1 and 2: BERT, BERT-MATCH and SPL-BERT
    print("Exporting Violation count results for experiments 1 and 2...")
    for dataset, model in tqdm(product(["amz", "bgc", "wos"], ["bert", "bert_match", "spl_bert"])):
        vio_csv = f"logs/avg_check_pred_y_all_output_{dataset}_{model}.csv"
        results = export_vio_results(dataset, model, folders, S, gates)
        output_folder = f"dumps/vio_{dataset}_{model}.pkl"
        pd.to_pickle(results, output_folder)
        print(f"Exported Violation results for {dataset} {model} to {output_folder}")
    #Automate for exp 3: SPL-BERT-HYPER with different S and gates
    print("Exporting Violation count results for experiment 3 (SPL-BERT-HYPER)...")
    for dataset in tqdm(["amz", "bgc", "wos"]):
        for S, gates in tqdm(product([0, 4], [1, 3])):
            vio_csv = f"logs/avg_check_pred_y_all_output_{dataset}_spl_bert_h_S{S}_gates{gates}.csv"
            results = export_vio_results(dataset, "spl_bert_h", folders, S, gates)
            output_folder = f"dumps/vio_{dataset}_spl_bert_hyper_S{S}_gates{gates}.pkl"
            pd.to_pickle(results, output_folder)
            print(f"Exported Violation results for {dataset} SPL-BERT-HYPER S={S} gates={gates} to {output_folder}")'''
    #Automate for exp 3 hard enforcement
    print("Exporting Violation count results for experiments 3 on hard enforcement...")
    for dataset, model in tqdm(product(["amz", "wos"], ["spl_bert"])):
        hard_enf = True
        vio_csv = f"logs/avg_check_pred_y_all_output_{dataset}_{model}_hard.csv"
        results = export_vio_results(dataset, model, folders, S, gates, hard_enf=hard_enf)
        output_folder = f"dumps/vio_{dataset}_{model}_hard.pkl"
        pd.to_pickle(results, output_folder)
        print(f"Exported Violation results for {dataset} {model} to {output_folder}")
if __name__ == "__main__":
    htc_exp()
    vio_exp()