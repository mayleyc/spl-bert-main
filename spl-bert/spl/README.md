# Semantic Probabilistic Layers

This repository holds the code for the NeurIPS 2022 paper [Semantic Probabilistic Layers](https://openreview.net/forum?id=o-mxIWAY1T8)
by Kareem Ahmed, Stefano Teso, Kai-Wei Chang, Guy Van den Broeck and Antonio Vergari

We introduce **Semantic Probabilistic Layers**, a drop-in replacement for the Softmax layer that *guarantees* the consistency of the neural
network's predictions with a given set of symbolic constraints, while retaining probabilistic semantics, supporting arbitrary constraints,
all while being tractable.

-------------------- 

## Installation
```
conda env create -f environment.yml
```

and if you encounter a pypsdd related error, running the following should solve the issue
```
pip install -vvv --upgrade --force-reinstall --no-binary :all: --no-deps pysdd
```

## Commands
Each of the four tasks includes a .sh script for training and testing.


## Hyperparameters (mayleyc note)
To use the CUB mini dataset, enter the --dataset hyperparameter as cub_mini. For full CUB, use cub_others. For other datasets, type the name followed by the dataset type, e.g. expr_FUN or enron_others.
