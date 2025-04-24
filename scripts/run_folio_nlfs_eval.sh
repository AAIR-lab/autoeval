 #!/bin/bash
 
TOTAL_RUNS=5
MODELS=$*

# Use an absolute path here if possible.
# If relative, then it execution begins from one level above project root (..)
# So, all relative paths must be specified relative to ../
DATAPATH="results/folio_premises/folio_premises/"

export PYTHONHASHSEED=0
export PYTHONPATH=.:$PYTHONPATH

for model in $MODELS
do
for (( run=0; run < $TOTAL_RUNS; run++)) do
    dataset_file="${DATAPATH}/run${run}/dataset_${model}.json"
    python3 folio_eval/nlfs_eval.py --dataset-file $dataset_file
done
done
