# HBGL: Exploiting Global and Local Hierarchies for Hierarchical Text Classifications

We used the official implementation
of [Exploiting Global and Local Hierarchies for Hierarchical Text Classification](https://github.com/kongds/HBGL).

## Create environment

Tested on Linux and Windows. Steps:

1. Create a conda env with Python 3.8
2. Install PyTorch 1.7.1
   with `conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch`
3. Install other packages with PyPy `pip install -r src/models/HBGL/requirements.txt`

## Prepare data

The preprocessed datasets must be in jsonl format file, like {'token': List[str], 'label': List[str]}.

1. Please place the 5 datasets in the "data" folder, and run `python src/models/HBGL/prepare_datasets.py` once for each
   dataset to create the dataset files in the correct format. Edit the script to run on different datasets.

2. Pre-process them using the code in the corresponding folder. First `cd src/models/HBGL`, then:

    + WoS: `python preprocess.py wos`
    + RCV1-V2: `python preprocess.py rcv1`
    + Linux Bugs: `python preprocess.py bugs`
    + BGC: `python preprocess.py bgc`
    + Amazon5x5: `python preprocess.py amazon`

## Train & Evaluation

Launch from the main repo folder with:

``` shell
bash src/models/HBGL/run_wos.sh

bash src/models/HBGL/run_rcv1.sh

bash src/models/HBGL/run_bgc.sh

bash src/models/HBGL/run_amaz.sh

bash src/models/HBGL/run_bugs.sh
```

In the scripts, one can use the "--only_test" and "--only_test_path" flags to test performance from checkpoints and the

It might be necessary to export PYTHONPATH variable in accordance with the system configuration.
Additionally, on Windows one should add `sys.path.append("path_to_folder\\htc-survey")` in `src/models/HBGL/eval.py`.

Code is based on [s2s-ft](https://github.com/microsoft/unilm/tree/master/s2s-ft).
