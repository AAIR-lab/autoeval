 #!/bin/bash
 
TOTAL_RUNS=10
DATASETS="ksat plogic fol fol_human regex"
MODEL="yi-1.5-34b"
DATAPATH="results"
TYPE="all"
ENV="explain-llm"

if [ "$TYPE" == "" ]; then
	echo "Invalid option"
	exit 1
elif [ "$TYPE" == "autoeval" ]; then
	PYTHONBUFFERED=1;PYTHONHASHSEED=0 python evaluation.py --auto --skip-verify --total-batches 10 --base-dir "$DATAPATH"/"zero-shot" --models "$MODEL"
elif [ "$TYPE" == "logieval" ]; then
	for (( run=0; run < 5; run++)) do
		xfce4-terminal --title="logieval-${MODEL}-run${run}" --hold --tab -e "bash -c 'module load mamba/latest && source activate ${ENV} && PYTHONBUFFERED=1; PYTHONHASHSEED=0; PYTHONPATH=. python multi_logieval_eval/evaluate.py ../results/Multi-LogiEval/Multi-LogiEval/run${run}/dataset.json ${MODEL}'"
	done
elif [ "$TYPE" == "folio-pr" ]; then
	for (( run=0; run < 5; run++)) do
		xfce4-terminal --title="folio-pr-${MODEL}-run${run}" --hold --tab -e "bash -c 'module load mamba/latest && source activate ${ENV} && PYTHONBUFFERED=1; PYTHONHASHSEED=0; PYTHONPATH=. python folio_eval/evaluate_folio.py results/folio_premises/folio_premises/run${run}/dataset.json ${MODEL}'"
	done
elif [ "$TYPE" == "folio" ]; then
	for (( run=0; run < 5; run++)) do
		xfce4-terminal --title="folio-${MODEL}-run${run}" --hold --tab -e "bash -c 'module load mamba/latest && source activate ${ENV} && PYTHONBUFFERED=1; PYTHONHASHSEED=0; PYTHONPATH=. python folio/evaluate.py ../results/folio_results/results/run${run}/folio_dataset.json ${MODEL}'"
	done

elif [ "$TYPE" == "all" ]; then
	PYTHONBUFFERED=1;PYTHONHASHSEED=0 python evaluation.py --auto --skip-verify --total-batches 10 --base-dir "$DATAPATH"/"zero-shot" --models "$MODEL"
	for (( run=0; run < 5; run++)) do
		xfce4-terminal --title="logieval-${MODEL}-run${run}" --hold --tab -e "bash -c 'module load mamba/latest && source activate ${ENV} && PYTHONBUFFERED=1; PYTHONHASHSEED=0; PYTHONPATH=. python multi_logieval_eval/evaluate.py ../results/Multi-LogiEval/Multi-LogiEval/run${run}/dataset.json ${MODEL}'"
		xfce4-terminal --title="folio-pr-${MODEL}-run${run}" --hold --tab -e "bash -c 'module load mamba/latest && source activate ${ENV} && PYTHONBUFFERED=1; PYTHONHASHSEED=0; PYTHONPATH=. python folio_eval/evaluate_folio.py results/folio_premises/folio_premises/run${run}/dataset.json ${MODEL}'"
		xfce4-terminal --title="folio-${MODEL}-run${run}" --hold --tab -e "bash -c 'module load mamba/latest && source activate ${ENV} && PYTHONBUFFERED=1; PYTHONHASHSEED=0; PYTHONPATH=. python folio/evaluate.py ../results/folio_results/results/run${run}/folio_dataset.json ${MODEL}'"
	done
fi



# FOLIO premises runs
#for (( run=0; run < 5; run++)) do
#xfce4-terminal --title="${MODEL}-run${run}" --hold --tab -e "bash -c 'module load mamba/latest && source activate ${ENV} && PYTHONBUFFERED=1; PYTHONHASHSEED=0; PYTHONPATH=. python multi_logieval_eval/evaluate.py ../results/Multi-LogiEval/Multi-LogiEval/run${run}/dataset.json ${MODEL}'"
#done

# logieval runs
#for (( run=0; run < 5; run++)) do
 #   PYTHONBUFFERED=1;PYTHONHASHSEED=0;PYTHONPATH=. python multi_logieval_eval/evaluate.py "../results/Multi-LogiEval/Multi-LogiEval"/"run$run"/"dataset.json" "$MODEL"
#done

# folio runs
#for (( run=0; run < 5; run++)) do
 #   PYTHONBUFFERED=1;PYTHONHASHSEED=0;PYTHONPATH=. python folio/evaluate.py "../results/folio_results/results"/"run$run"/"folio_dataset.json" "$MODEL"
#done

# python3 -m v${ENV}.entrypoints.openai.api_server --model mistralai/Ministral-8B-Instruct-2410 --download-dir /scratch/drbrambl/model_cache --enforce-eager --disable-log-requests
