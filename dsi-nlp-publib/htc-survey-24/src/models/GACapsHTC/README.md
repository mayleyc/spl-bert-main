# GACaps-HTC: Graph Attention Capsule Network for Hierarchical Text Classification

We used the official implementation
of [GACaps-HTC: Graph Attention Capsule Network for Hierarchical Text Classification](https://github.com/jinhyun95/GACapsHTC).

## Create environment

Steps:

1. Create a conda or Pip env with Python 3.8;
2. Install PyTorch 1.7.1
   with `conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch` or with the pip
   equivalent;
3. Install other packages with PyPy `pip install -r src/models/GACapsHTC/requirements.txt`

## Prepare data

The processed datasets must be in jsonl format file, like {'text': str, 'label': List[str]}.
Please place the 5 datasets in the `data` folder, and run `python src/models/GACapsHTC/prepare_data.py` from the main
repo folder once for each dataset to create the dataset files in the correct format. Edit the script to run on
different datasets.

This will create the run configuration in `src/models/GACapsHTC/configs/{dataset}.json`, that reference the data files
created in `src/models/GACapsHTC/data/{dataset}`. Specifically, the configuration contains the following fields:

- `config.path.data.train`, `config.path.data.val`, and `config.path.data.test`

    - These files are .json files where each line is a JSON object composed as follows:

      `{"text": "this is an example document", "label": ["<study_unit_label_name>", "<parent_topic_label_name>", ancester topics]}`

    - The labels are required to be sorted from the leaf node to the node just below the root.

      e.g. `["Vector, Matrix, and Tensor", "Basics of Linear Algebra", "Linear Algebra", "AI Prerequisites"]`

- `config.path.labels`: This is a json file containing a dictionary where label names and label indices are keys and
  values, respectively.

- `config.path.prior`

    - This is a json file containing a dictionary of the following format:

      `{"<parent_label_name>": {"<child_1_label_name>": <prior 1>, "<child_2_label_name>": <prior 2> ...} ...}`

      e.g. `{"AI Prerequisites": {"Calculus": 0.4, "Linear Algebra": 0.3, "Probability": 0.2, "Statistics": 0.1} ...}`

    - For each parent node - child node pair, prior is simply obtained as the proportion of the number of instances.

- `config.path.hierarchy`: This is a tsv file where each line is written as
  follows:  `<parent_label_name>\t<child_1_label_name>\t<child_2_label_name>...`

- `config.training`: Contains the training hyperparameters, like batch size, max input length, etc.

## Run

Run using the `run.py` script, that takes the path to a configuration file as its argument.
E.g., to run on the WoS: `python src/models/GACapsHTC/run.py src/models/GACapsHTC/configs/wos.json`.

- Checkpoints are saved in the `src/models/GACapsHTC/data/{dataset}/checkpoints` folder;
- When training finishes, results on the test set are dumped to a JSON file
  in `src/models/GACapsHTC/results/{dataset}.json`;
- Script `src/models/GACapsHTC/run_all.sh` allows to run all tests on the five datasets sequentially.

