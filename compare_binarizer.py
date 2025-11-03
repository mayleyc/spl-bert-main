from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

# Example hierarchy (parent -> leaf)
hierarchy = {
    "root": [],
    "nonfiction": ["biography-memoir", "cooking"],
    "fiction": ["fantasy", "mystery-suspense"]
}

# Sample labels
samples = [
    ["nonfiction", "biography-memoir"],  # full path to leaf
    ["nonfiction"],                       # stops at parent
    ["fiction", "fantasy"],               # full path to leaf
    ["fiction"]                            # stops at parent
]

# Real examples
samples_train_list = get_bgc_split_jsonl("train")

# -------------------------------
# MultiLabelBinarizer approach
mlb = MultiLabelBinarizer()
mlb.fit(samples)
print("MLB classes:", mlb.classes_)

mlb_encoded = mlb.transform(samples)
print("\nMLB encoded matrix:")
print(mlb_encoded)

# Notice: parent labels are encoded only if present in the sample
# No automatic ancestor propagation

# -------------------------------
# Hierarchy-aware OHE
# Build node_to_index for all nodes
all_nodes = ["nonfiction", "biography-memoir", "cooking", "fiction", "fantasy", "mystery-suspense"]
node_to_index = {node: i for i, node in enumerate(all_nodes)}

# Function to get ancestors
parent_map = {}
for parent, children in hierarchy.items():
    for child in children:
        parent_map[child] = parent

def get_ancestors(node):
    ancestors = []
    while node in parent_map:
        node = parent_map[node]
        ancestors.append(node)
    return ancestors

# Build OHE with ancestor propagation
def ohe_hierarchy(labels_list):
    result = []
    for labels in labels_list:
        vec = np.zeros(len(all_nodes), dtype=int)
        for label in labels:
            vec[node_to_index[label]] = 1
            for ancestor in get_ancestors(label):
                if ancestor in node_to_index:
                    vec[node_to_index[ancestor]] = 1
        result.append(vec)
    return np.array(result)

ohe_encoded = ohe_hierarchy(samples)
print("\nHierarchy-aware OHE encoded matrix:")
print(ohe_encoded)
