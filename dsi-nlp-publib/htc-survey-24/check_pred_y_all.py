
import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
import datetime
import argparse
import ast
from itertools import product


'''bert_pred_y_fp = "dumps/BERT_MATCH/bert_multilabel_BGC_concat_cls_3/run_2025-10-20_11-46-10/y_pred_repeat_3_2025-10-20_14-08-05.csv"
bert_match_pred_y_fp = "dumps/BERT_MATCH/bert_multilabel_WOS_concat_cls_3/run_2025-10-02_12-49-17/all_folds_pred_2025-10-02_14-38-56.csv"
emb_pckl = "/mnt/cimec-storage6/users/nguyenanhthu.tran/2025thesis/spl-bert/spl/C-HMCNN/embeddings/emb_bert-base-uncased_amazon_20250806-183616_val_batch0.pickle"

#spl_pred_y_fp = "dumps/BERT/bert_multilabel_WOS_concat_cls_3_SPL/run_2025-10-02_12-51-48/all_folds_pred_2025-10-02_14-32-28.csv"
#ohe_dict_from_csv = "csv/bgc_tax_one_hot.csv" # Replace with csv path

species_only = False'''
#check mutual exclusivity errors (2 leaves or more)

def load_pickle(filepath):
    """
    Load a Python object from a pickle file.

    Args:
        filepath (str): Path to the .pickle file.

    Returns:
        Any: The object stored in the pickle file.
    """
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data

def count_me(predictions, n_nodes):
    """
    Yields ME (mutual exclusivity) and zero-prediction violations per level.
    
    Args:
        predictions (np.ndarray): shape (num_samples, total_classes)
        n_leaves (List[int]): number of labels per level
        
    Yields:
        dict: {
            "level": int,
            "non_exclusive_rows": np.ndarray,
            "zero_rows": np.ndarray,
            "non_exclusive_count": int,
            "zero_count": int,
        }
    """
    start = 0
    level = 1

    if args.species_only:
        n_leaves = [n_nodes[-1]]
    else:
        n_leaves = n_nodes
        
    for length in n_leaves:
        end = start + length
        if args.species_only:
            cut = predictions
        else:
            cut = predictions[:, start:end]  # get slice for this level (note the fix here!)
        row_sums = np.sum(cut, axis=1)

        non_exclusive_rows = np.where(row_sums != 1)[0]
        more_pred_rows = np.where(row_sums > 1)[0]
        zero_rows = np.where(row_sums == 0)[0]

        '''
        n = 5
        if len(non_exclusive_rows) > n and len(zero_rows) > n:
            for i in range(n):
                print(f"Level {level}; no.{i}:")
                print(f"  non_exclusive_row {non_exclusive_rows[i]} → {predictions[non_exclusive_rows[i]]}")
                print(f"  zero_row          {zero_rows[i]} → {predictions[zero_rows[i]]}")
        '''
        yield {
            "level": level,
            "non_exclusive_rows": non_exclusive_rows,
            "zero_rows": zero_rows,
            "more_pred_rows": more_pred_rows,
            "non_exclusive_count": len(non_exclusive_rows),
            "zero_count": len(zero_rows),
            "more_pred_count": len(more_pred_rows),
        }

        start = end
        level += 1
    
def compare_hierarchy_violations(predictions, ohe_dict): #convert to tuples for hashability -> faster?
    # Convert all values to tuples once and store in a set
    allowed_set = {tuple(v) for v in ohe_dict.values()}
    
    #print(list(allowed_set)[0])

    viol_rows = []
    for i, row in enumerate(predictions):
        row_tuple = tuple(row)  # Convert prediction row to tuple
        if row_tuple not in allowed_set:
            viol_rows.append(i)
    return viol_rows

def get_dataset(pred_dir):
    pred_dir = pred_dir.lower()
    dataset_keywords = {
        "amz": ["amz", "amazon"],
        "bgc": ["bgc"],
        "wos": ["wos"]
    }

    for dataset, keywords in dataset_keywords.items():
        if any(keyword in pred_dir for keyword in keywords):
            return dataset
    return None

dataset_to_n_leaves = {
    "amz": [5, 25],
    "bgc": [7, 46, 77, 16],
    "wos": [7, 138],
}
folders = {"bert": "dumps/BERT",
"bert_match": "dumps/BERT_MATCH",
"spl_bert": "dumps/BERT_SPL",
"spl_bert_h":"dumps/BERT_SPLHYPER",
"ohe": "csv"}

def find_latest_run(folder, pattern="run_*", contains="pred"):
    runs = sorted(Path(folder).glob(pattern), key=os.path.getmtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"No runs found in {folder}")
    latest = runs[0]
    preds = [f for f in latest.glob("*") if contains in f.name]
    return preds, str(latest)

def find_specific_run(folder, run_name, contains="pred"):
    run_path = Path(folder) / run_name
    if not run_path.exists():
        raise FileNotFoundError(f"Run {run_name} not found in {folder}")
    preds = [f for f in run_path.glob("*") if contains in f.name]
    return preds, str(run_path)

def fp_search(dataset, model, folders, **abl_args):
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
    preds, latest_run = find_latest_run(folder, pattern="run_*")
    #preds, run = find_specific_run(folder, run_name=f"run_2025-10-24_18-29-51_soft", contains="pred")

    return [str(p) for p in preds], str(folder)

def average_violation_csvs(csv_paths, output_path):
    # Read CSVs
        dfs = [pd.read_csv(p) for p in csv_paths]
        
        # Convert stringified lists back to lists
        for df in dfs:
            for col in df.columns:
                if "rows" in col:
                    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else [])
        
        merged = pd.concat(dfs)

        # Function to merge row lists across repeats (union)
        def merge_rows(series):
            all_rows = set()
            for lst in series:
                all_rows.update(lst)
            return sorted(all_rows)

        # Compute mean and std for numeric columns, and union for row lists
        agg_funcs = {
            "non_exclusive_count": ["mean", "std"],
            "zero_count": ["mean", "std"],
            "more_pred_count": ["mean", "std"],
            "total_HV": ["mean", "std"],
            "other_HV": ["mean", "std"],
        }

        grouped = (
            merged.groupby(["dataset", "model", "level"], as_index=False)
            .agg(agg_funcs)
        )

        # Flatten MultiIndex column names created by mean/std aggregation
        grouped.columns = [
            "_".join(col).strip("_") if isinstance(col, tuple) else col
            for col in grouped.columns.values
        ]


        # Save to CSV
        grouped.to_csv(output_path, index=False)
        print(f"Averaged results (with std) saved to {output_path}")

        return grouped

def check_one(repeat_id, **abl_args):
    # Get prediction files and model folder
    pred_files, folder = fp_search(dataset=args.dataset, model=args.model, folders=folders, **abl_args)

    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    repeat_csv = logs_dir / f"check_pred_y_all_output_{args.dataset}_{args.model}_S{abl_args['S']}_gates{abl_args['gates']}_repeat{repeat_id}.csv"
    records = []

    pred_file = pred_files[repeat_id - 1]  # Select file based on repeat_id
    # Process each prediction file
    
    labels = pd.read_csv(pred_file, header=None).iloc[1:].to_numpy(dtype=np.int64)
    dataset = get_dataset(pred_file)
    n_leaves = dataset_to_n_leaves.get(dataset)
    if n_leaves is None:
        raise ValueError(f"Unknown dataset key: {dataset}")

    # Load OHE dict
    ohe_path, _ = fp_search(dataset=args.dataset, model="ohe", folders=folders)
    ohe_df = pd.read_csv(ohe_path[0], index_col=0)
    ohe_dict = ohe_df.astype(int).to_dict(orient='index')
    for k, v in ohe_dict.items():
        ohe_dict[k] = list(ohe_dict[k].values())

    predictions = np.array(labels)

    # enable to check species only
    if args.species_only:
        # slice the dictionary values
        ohe_dict_species = dict()
        #k, v = next(iter(ohe_dict.items()))
        #new_v = v[-n_items:]

        for k, v in ohe_dict.items():
            new_v = v[-n_leaves[-1]:]
            ohe_dict_species[k] = new_v

        predictions_slice = [i[-n_leaves[-1]:] for i in predictions]
        pred = predictions_slice
        ohe = ohe_dict_species
    else:
        n_leaves = dataset_to_n_leaves.get(dataset)
        if n_leaves is None:
            raise ValueError(f"Unknown dataset key: {dataset}")
        pred = predictions
        ohe = ohe_dict
    # Initialize counters
    ME_all = zero_rows_all = more_pred_all = 0
    me_set = zero_set = more_pred_set = set()

    for stats in count_me(pred, n_leaves):
        me_set.update(stats['non_exclusive_rows'])
        ME_all += stats['non_exclusive_count']
        zero_set.update(stats['zero_rows'])
        zero_rows_all += stats['zero_count']
        more_pred_set.update(stats['more_pred_rows'])
        more_pred_all += stats['more_pred_count']

    viol_rows = compare_hierarchy_violations(pred, ohe)
    other_violations = set(viol_rows) - me_set

    for stats in count_me(pred, n_leaves):
    # Append a record per level
        records.append({
    "dataset": dataset,
    "model": args.model,
    "file": Path(pred_file).name,
    "level": stats['level'],
    "non_exclusive_count": stats['non_exclusive_count'],
    "zero_count": stats['zero_count'],
    "more_pred_count": stats['more_pred_count'],
    "ME_rows": list(stats['non_exclusive_rows']),
    "zero_rows": list(stats['zero_rows']),
    "more_pred_rows": list(stats['more_pred_rows']),
})

    # Add a TOTAL row
    records.append({
        "dataset": dataset,
        "model": args.model,
        "file": Path(pred_file).name,
        "level": "TOTAL",
        "non_exclusive_count": sum(stats['non_exclusive_count'] for stats in count_me(pred, n_leaves)),
        "zero_count": sum(stats['zero_count'] for stats in count_me(pred, n_leaves)),
        "more_pred_count": sum(stats['more_pred_count'] for stats in count_me(pred, n_leaves)),
        "ME_rows": sorted(set().union(*(stats['non_exclusive_rows'] for stats in count_me(pred, n_leaves)))),
        "zero_rows": sorted(set().union(*(stats['zero_rows'] for stats in count_me(pred, n_leaves)))),
        "more_pred_rows": sorted(set().union(*(stats['more_pred_rows'] for stats in count_me(pred, n_leaves)))),
        "total_HV": len(viol_rows),
        "other_HV": len(other_violations),
    })

    # Save the CSV
    records_df = pd.DataFrame(records)
    records_df.to_csv(repeat_csv, index=False)
    print(f"Summary CSV saved: {repeat_csv}")

    return repeat_csv
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check prediction hierarchy and exclusivity violations.")
    parser.add_argument("--dataset", required=True, help="Dataset key (e.g., amz, bgc, wos)")
    parser.add_argument("--model", required=True, help="Model key (e.g., bert, bert_match, spl_bert, ohe)")
    parser.add_argument("--species-only", action="store_true", default=False, help="Check species-level predictions only")

    args = parser.parse_args()
    
    abl_args = {}
    if args.model == "spl_bert_h":
        for a, b in product([0, 4], [1, 3]):
            abl_args = {"S": a, "gates": b}

            repeat_csvs = [
                check_one(repeat_id=1, **abl_args),
                check_one(repeat_id=2, **abl_args),
                check_one(repeat_id=3, **abl_args),
            ]

            avg_csv = Path("logs") / f"avg_check_pred_y_all_output_{args.dataset}_{args.model}_S{abl_args['S']}_gates{abl_args['gates']}.csv"
            average_violation_csvs(repeat_csvs, avg_csv)
    else:
        repeat_csvs = [
            check_one(repeat_id=1),
            check_one(repeat_id=2),
            check_one(repeat_id=3),
        ]

        avg_csv = Path("logs") / f"avg_check_pred_y_all_output_{args.dataset}_{args.model}.csv"
        average_violation_csvs(repeat_csvs, avg_csv)
