 #!/bin/bash
 
TOTAL_RUNS=10
DATASETS="ksat plogic fol fol_human regex"
MODELS="gpt-3.5-turbo gpt-4o claude phi-3 mistral llama-3-8b"

PROMPT_DIR=$1
PROMPT_VERIFIED_DIR="$PROMPT_DIR"

mkdir -p $PROMPT_DIR
mkdir -p $PROMPT_VERIFIED_DIR

for (( run=0; run < $TOTAL_RUNS; run++)) do
    for dataset in $DATASETS; do
    
        mkdir -p "$PROMPT_DIR"/"batch$run"/"$dataset"
        mkdir -p "$PROMPT_VERIFIED_DIR"/"batch$run"/"$dataset"
        cp "run$run"/"$dataset"/"dataset.json" "$PROMPT_DIR"/"batch$run"/"$dataset"
    
        for model in $MODELS; do
        
            cp "run$run"/"$dataset"/"dataset_nlfs_"$model"_verfied.json" "$PROMPT_VERIFIED_DIR"/"batch$run"/"$dataset"/"$model".verified.json
        done
    done
done
