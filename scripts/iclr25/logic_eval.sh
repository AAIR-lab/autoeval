#!/bin/bash

base_dir="/tmp/folio_and_logieval/"
folio_dir="$base_dir"/"folio"
logieval_dir="$base_dir"/"logieval"

total_runs=5
model_name="gpt-4o-mini"

create_and_run_dataset() {

    base_dir=$1
    run_no=$2
    eval_script=$3

    dataset_file="$base_dir"/"dataset.json"
    run_dir="$base_dir"/"run$run_no"
    mkdir -p $run_dir
    cp $dataset_file $run_dir

    echo "Run $i: PYTHONPATH=. python3 $eval_script $run_dir/dataset.json $model_name"
    PYTHONPATH=. python3 $eval_script  $run_dir/dataset.json $model_name
}

for (( i=0; i<total_runs; i+=1)) do
    
    create_and_run_dataset $folio_dir $i "folio_eval/evaluate_folio.py"
    create_and_run_dataset $logieval_dir $i "multi_logieval_eval/evaluate.py "
done
