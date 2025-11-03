#!/bin/bash

# Define the datasets and models to iterate over
DATASETS=("amz" "bgc" "wos")
MODELS=("bert" "bert_match" "spl_bert")

# Loop through each dataset and model, then execute the Python script
for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    echo "Running check_pred_y_all.py for dataset: $dataset, model: $model"
    python check_pred_y_all.py --dataset "$dataset" --model "$model"
    echo "--------------------------------------------------"
  done
done

echo "All checks completed."
