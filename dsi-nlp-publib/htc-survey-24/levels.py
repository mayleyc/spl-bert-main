import os
from collections import deque, defaultdict
import pandas as pd

tax_files = ["data/BGC/bgc_tax.txt", "data/Amazon/amazon_tax.txt", "data/WebOfScience/wos_tax.txt"]

# Build parent → children map
for tax_file in tax_files:
    children = defaultdict(list)
    with open(tax_file) as f:
        for line in f:
            parts = line.strip().split()
            parent, kids = parts[0], parts[1:]
            children[parent].extend(kids)

    # BFS to assign levels and keep order
    levels_in_order = []  # list of (node, level)
    queue = deque([("root", 0)])  # root is level 0

    while queue:
        node, lvl = queue.popleft()
        levels_in_order.append((node, lvl))
        for child in children.get(node, []):
            queue.append((child, lvl + 1))

    # Optional: Save to CSV
    output = os.path.join(os.path.dirname(tax_file), os.path.basename(tax_file).split(".")[0] + "_levels.csv")
    df = pd.DataFrame(levels_in_order, columns=["node", "level"])
    df.to_csv(output, index=False)

    '''# Print node-level pairs
    for node, lvl in levels_in_order:
        print(node, lvl)
    '''
    print(f"Levels saved to {output}.")