
import numpy as np
import pandas as pd
import pickle

bert_pred_y_fp = "dumps/BERT/bert_multilabel_AMZ_concat_cls_3/run_2025-07-09_09-35-30/all_folds_pred_2025-07-11_11-09-38.csv"
bert_match_pred_y_fp = "dumps/BERT_MATCH/bert_multilabel_AMZ_concat_cls_3/run_2025-07-11_14-08-25/all_folds_pred_2025-07-15_13-42-12.csv"
emb_pckl = "/mnt/cimec-storage6/users/nguyenanhthu.tran/2025thesis/spl-bert/spl/C-HMCNN/embeddings/emb_bert-base-uncased_amazon_20250806-183616_val_batch0.pickle"


spl_pred_y_fp = "/mnt/cimec-storage6/users/nguyenanhthu.tran/2025thesis/spl-bert/spl/C-HMCNN/pred_y/20250825-170504_250825_model-d_wos_me_nz_full/predicted_test_wos_oeFalse_250825_model-d_wos_me_nz_full_20250825-170504.csv"
ohe_dict_from_csv = "/mnt/cimec-storage6/users/nguyenanhthu.tran/2025thesis/spl-bert/spl/C-HMCNN/wos_tax_one_hot.csv" # Replace with csv path

species_only = False
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

    if species_only:
        n_leaves = [n_nodes[-1]]
    else:
        n_leaves = n_nodes
        
    for length in n_leaves:
        end = start + length
        if species_only:
            cut = predictions
        else:
            cut = predictions[:, start:end]  # get slice for this level (note the fix here!)
        row_sums = np.sum(cut, axis=1)

        non_exclusive_rows = np.where(row_sums != 1)[0]
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
            "non_exclusive_count": len(non_exclusive_rows),
            "zero_count": len(zero_rows),
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

def get_dataset(pred_file):
    pred_file = pred_file.lower()
    dataset_keywords = {
        "amz": ["amz", "amazon"],
        "bgc": ["bgc"],
        "wos": ["wos"]
    }

    for dataset, keywords in dataset_keywords.items():
        if any(keyword in pred_file for keyword in keywords):
            return dataset
    return None

dataset_to_n_leaves = {
    "amz": [5, 25],
    "bgc": [7, 19, 120],
    "wos": [7, 138],
}

#include ME violations as hierarchy violations


def main(): 
    # For pckl files

    #pred_file = emb_pckl
    #df = pd.read_csv(bert_match_pred_y_fp)
    #texts, embeddings, labels = load_pickle(pred_file)
    #labels = [l.numpy() for l in pred_file]
    print(f"SPECIES ONLY: {species_only}")
    # For csv (prediction) files
    pred_file = spl_pred_y_fp
    
    labels = pd.read_csv(pred_file, header=None).to_numpy(dtype=np.int64)
        
    # Checking accordance with OHE dict
    ohe_df = pd.read_csv(ohe_dict_from_csv, index_col=0)
    ohe_dict = ohe_df.astype(int).to_dict(orient='index')
    for k, v in ohe_dict.items():
        ohe_dict[k] = list(ohe_dict[k].values())

    #first_key, first_value = next(iter(ohe_dict.items()))

    #print("First key:", first_key)
    #print("First value as list:", first_value)
    #quit()

    # Convert to NumPy array
    predictions = np.array(labels)
    dataset = get_dataset(pred_file)
    n_leaves = dataset_to_n_leaves.get(dataset)
    # enable to check species only
    if species_only:
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
        pred = predictions
        ohe = ohe_dict

    if n_leaves is None:
        raise ValueError(f"Unknown dataset key: {dataset}")
    ME_all = 0
    zero_rows_all = 0
    me_set = set()
    zero_set = set()
    
    for stats in count_me(pred, n_leaves):
        print(f"Level {stats['level']}:")
        me_set.update(stats['non_exclusive_rows'])
        print(f"  Non-exclusive: {stats['non_exclusive_count']}")
        ME_all += stats['non_exclusive_count']

        zero_set.update(stats['zero_rows'])
        print(f"  Zero predictions: {stats['zero_count']}")
        zero_rows_all += stats['zero_count']

        print("Rows with >2 predictions in this level:")
        print(np.setdiff1d(stats['non_exclusive_rows'], stats['zero_rows'])+1) # see which rows have 2 and above guesses
    
    
    viol_rows = compare_hierarchy_violations(pred, ohe)

    viol_set = set(viol_rows)
    

    # Get rows that are in violations but NOT in ME or zero prediction violations
    other_violations = viol_set - me_set - zero_set

        
    print(f"Total HV: {len(viol_rows)}/{len(predictions)}")
    print(f"Other HV (Total HV minus ME and zero): {len(other_violations)}") #might not work if a row has multiple zero/ME predictions
if __name__ == "__main__":
    main()