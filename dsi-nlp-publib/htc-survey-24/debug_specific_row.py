import pandas as pd
from collections import defaultdict

# The row you provided
target_node = "world-war-ii-military-history"

# Load the ground truth for levels
levels_df = pd.read_csv("data/BGC/bgc_levels.csv")
node_to_level = pd.Series(levels_df.level.values, index=levels_df.node).to_dict()

# Load the one-hot file
df = pd.read_csv("csv/bgc_tax_one_hot.csv", index_col=0)

# Get the specific row
try:
    row = df.loc[target_node]
except KeyError:
    print(f"Error: Node '{target_node}' not found in the CSV file.")
    exit()

# Find all annotated classes in that row
annotated_classes = row[row == 1].index.tolist()

print(f"Analyzing annotations for node: '{target_node}'\n")
print("Found the following annotated classes and their levels:")

levels_for_row = defaultdict(list)
for cls in annotated_classes:
    if cls in node_to_level:
        level = node_to_level[cls]
        levels_for_row[level].append(cls)
        print(f"  - Class: {cls}, Level: {level}")
    else:
        print(f"  - Class: {cls}, Level: Not Found")

print("\nChecking for conflicts (more than one class at the same level)...")
errors_found = False
for level, classes in levels_for_row.items():
    if len(classes) > 1:
        errors_found = True
        print(f'  - CONFLICT DETECTED at level {level}: {", ".join(classes)}')

if not errors_found:
    print("No conflicts found for this row.")
