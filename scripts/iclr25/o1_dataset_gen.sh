#!/bin/bash

BASE_DIR="./o1_preview_experiments"
MAX_BATCHES=1

export PYTHONHASHSEED=0

for ((i=0; i < $MAX_BATCHES; i++)); do
    python3 dataset_generator.py --filter-field num_operators --dataset-type plogic --base-dir "$BASE_DIR"/"batch$i"/"plogic"/ --sample-count 5
    python3 dataset_generator.py --filter-field num_operators --dataset-type fol_human --base-dir "$BASE_DIR"/"batch$i"/"fol_human"/ --sample-count 5
done
