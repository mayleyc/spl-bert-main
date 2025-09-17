import os
import json
from collections import defaultdict

root_dir = "embeddings"
splits = ["train", "val", "test"]
datasets = defaultdict(dict)

for dirpath, _, filenames in os.walk(root_dir):
    for fn in filenames:
        fn_lower = fn.lower()
        for split in splits:
            if split in fn_lower:
                parts = fn.split("_")
            
                dataset = parts[-4]
                full_path = os.path.join(dirpath, fn)
                mod_time = os.path.getmtime(full_path)

                # If no file stored yet or this one is newer
                current = datasets[dataset].get(split)
                if current is None or mod_time > current[1]:
                    datasets[dataset][split] = (full_path, mod_time)

# Flatten to string paths only
final_config = {
    dataset: {
        split: path_time[0]
        for split, path_time in split_dict.items()
    }
    for dataset, split_dict in datasets.items()
}

# Write to JSON
with open("embeddings_config.json", "w") as f:
    json.dump(final_config, f, indent=2)
print("JSON created!")
