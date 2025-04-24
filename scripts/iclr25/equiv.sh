 #!/bin/bash
 
TOTAL_RUNS=10
START_RUN=0
DATASETS="ksat plogic fol fol_human regex"
#DATASETS="ksat plogic"
MODELS="llama-3-70b"

EQUIV_DIR="/scratch/rkaria/equiv_results"

for model in $MODELS; do
    #mkdir -p "$EQUIV_DIR"/"${model}_equiv_results"
    for (( run=$START_RUN; run < $TOTAL_RUNS; run++)) do
        #mkdir -p "$EQUIV_DIR"/"${model}_equiv_results/run$run"
        for dataset in $DATASETS; do
            #mkdir -p "$EQUIV_DIR"/"${model}_equiv_results/run$run"/"$dataset"
            #cp "$EQUIV_DIR"/"gpt-3.5-turbo_equiv_results/run$run"/"$dataset/dataset_nlfs_gpt-4o_verfied.json" "$EQUIV_DIR"/"${model}_equiv_results/run$run"/"$dataset/dataset_nlfs_gpt-4o_verfied.json"
            PYTHONPATH=".";PYTHONBUFFERED=1;PYTHONHASHSEED=0 python3 evaluation.py --dataset-filepath "$EQUIV_DIR"/"compiled_results/run$run"/"$dataset/dataset_nlfs_gpt-4o_verfied.json" --skip-nlfs --model-name $model --dataset-type "$dataset" --llm-verify "$model"
        done
    done
done
