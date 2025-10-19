'''from collections import deque, defaultdict

# Build parent → children map
children = defaultdict(list)
with open("data/BGC/bgc_tax.txt") as f:
    for line in f:
        parts = line.strip().split()
        parent, kids = parts[0], parts[1:]
        children[parent].extend(kids)

# BFS to assign levels
levels = {}
queue = deque([("root", 0)])  # root is level 0

while queue:
    node, lvl = queue.popleft()
    levels[node] = lvl
    for child in children.get(node, []):
        queue.append((child, lvl + 1))

# Count nodes per level
level_counts = defaultdict(int)
for lvl in levels.values():
    level_counts[lvl] += 1

# Save to CSV
import pandas as pd
#pd.DataFrame(levels.items(), columns=["node", "level"]).to_csv("data/BGC/bgc_levels.csv", index=False)
print(level_counts)

print("Levels saved to data/BGC/bgc_levels.csv")
'''

from collections import deque, defaultdict
import pandas as pd

# Build parent → children map
children = defaultdict(list)
with open("data/BGC/bgc_tax.txt") as f:
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
df = pd.DataFrame(levels_in_order, columns=["node", "level"])
df.to_csv("data/BGC/bgc_levels.csv", index=False)

# Print node-level pairs
for node, lvl in levels_in_order:
    print(node, lvl)

print("Levels saved to data/BGC/bgc_levels_ordered.csv")