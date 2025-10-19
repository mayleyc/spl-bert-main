import sys
import pandas as pd
from collections import defaultdict

def check_hierarchy(one_hot_file, levels_file):
    """
    Checks if any class has multiple annotations on the same hierarchy level.
    """
    print(f"Checking file: {one_hot_file}")
    print(f"Using hierarchy levels from: {levels_file}")

    # Load ground truth levels
    levels_df = pd.read_csv(levels_file)
    node_to_level = pd.Series(levels_df.level.values, index=levels_df.node).to_dict()

    # Load the one-hot file
    try:
        df = pd.read_csv(one_hot_file, index_col=0)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    errors_found = False
    # Iterate over each row in the dataframe
    for index_name, row in df.iterrows():
        annotated_classes = row[row == 1].index.tolist()
        
        levels_for_row = defaultdict(list)
        for cls in annotated_classes:
            if cls in node_to_level:
                level = node_to_level[cls]
                levels_for_row[level].append(cls)
        
        for level, classes in levels_for_row.items():
            if len(classes) > 1:
                errors_found = True
                print(f'  - CONFLICT in row "{index_name}": Found {len(classes)} annotations at level {level}: {", ".join(classes)}')

    if not errors_found:
        print("No classes found with multiple annotations on the same hierarchy level.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python check_multilabel_integrity.py <path_to_one_hot_file> <path_to_levels_file>")
        sys.exit(1)
    
    one_hot_file_path = sys.argv[1]
    levels_file_path = sys.argv[2]
    check_hierarchy(one_hot_file_path, levels_file_path)