 #!/bin/bash
 
TOTAL_RUNS=10
DATASETS="ksat plogic fol fol_human regex"
MODELS=$*
DATAPATH="results"

export PYTHONHASHSEED=0

for model in $MODELS
do
for dataset in $DATASETS
do
for (( batch=0; batch < $TOTAL_RUNS; batch++)) do

    # echo $model $dataset $batch

    sleep 2
    pkill -u `whoami` node
    pkill -u `whoami` python3
    sleep 2

    python3 evaluation.py \
        --auto \
        --skip-nlfs \
        --total-batches 1 \
        --base-dir "$DATAPATH"/"zero-shot" \
        --datasets $dataset \
        --models $model \
        --start-batch $batch
done
done
done
