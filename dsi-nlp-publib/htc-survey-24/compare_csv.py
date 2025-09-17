import pandas as pd

def count_matching_y(pred, y):
    # Read CSVs (headers automatically parsed)
    df1 = pd.read_csv(pred)
    df2 = pd.read_csv(y)
    
    # Ensure both have the same number of rows to compare, or trim to smaller size
    min_len = min(len(df1), len(df2))
    df1 = df1.iloc[:min_len]
    df2 = df2.iloc[:min_len]
    
    # Check row-wise equality (all columns)
    matches = (df1 == df2).all(axis=1)
    
    # Count how many rows match exactly
    matching_count = matches.sum()
    return matching_count

def count_matching_ohe(pred):
    df1 = pd.read_csv(pred).astype(int)
    ohe_df = pd.read_csv(ohe_csv, index_col=0).astype(int)

    # Check if number of columns match
    if df1.shape[1] != ohe_df.shape[1]:
        raise ValueError(f"Number of columns do NOT match! Pred has {df1.shape[1]} columns, ohe_csv has {ohe_df.shape[1]} columns.")
    
    # Convert ohe rows to tuples and put into a set
    ohe_rows_set = set(tuple(row) for row in ohe_df.values)
    
    match_count = 0
    
    for _, row in df1.iterrows():
        if tuple(row) in ohe_rows_set:
            match_count += 1
    
    return match_count

#def me_errors(pred, levels):


# Example usage:
pred = 'dumps/BERT_MATCH/bert_multilabel_AMZ_concat_cls_3/run_2025-06-09_12-52-04/all_folds_pred_2025-06-16_13-02-23.csv'
y = 'dumps/BERT_MATCH/bert_multilabel_AMZ_concat_cls_3/run_2025-06-09_12-52-04/all_folds_true_2025-06-16_13-02-26.csv'
ohe_csv = "amazon_tax_one_hot.csv"

print("y columns (if any):")
print(pd.read_csv(y, index_col=0, nrows=3))


print("\nohe columns:")
print(pd.read_csv(ohe_csv, index_col=0, nrows=3))

'''match_y = count_matching_y(pred, y)
match_ohe = count_matching_ohe(y)

print(f"Number of matching rows (excluding headers): {match_y}/{min(len(pd.read_csv(pred)), len(pd.read_csv(y)))}")
print(f"Number of matching rows in OHE: {match_ohe}/{len(pd.read_csv(y))}")'''