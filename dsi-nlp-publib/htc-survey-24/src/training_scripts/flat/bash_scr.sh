#!/bin/bash

N_REPEATS=3   

for i in $(seq 1 $N_REPEATS); do
  echo "=== Global Repeat $i/$N_REPEATS ==="

  echo "Running bert_match.py"
  python -m src.training_scripts.flat.bert_match

  echo "Running bert.py"
  python -m src.training_scripts.flat.bert

  echo "Running spl-bert.py"
  python -m src.training_scripts.flat.spl-bert

done