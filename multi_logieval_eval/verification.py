import json
from tqdm import tqdm
import re

def matching_word_found(target,target_word):
    # Assumes special characters have been removed.
    indices = [m.start() for m in re.finditer(target_word, target)]
    for x in reversed(range(len(indices))):
        if indices[x] + len(target_word) < len(target):
            if target[indices[x] + len(target_word)] != " ":
                del indices[x]
                continue
        if indices[x] != 0:
            if target[indices[x] - 1] != " ":
                del indices[x]
                continue
    return len(indices) > 0

def verify(target,gt):
    lower_case = target.lower()
    lower_case = re.sub("yes or no", '', lower_case)
    lower_case = re.sub('[!,*)@#%(&$_?^:;"\'\n\"]', ' ', lower_case)
    while '  ' in lower_case:
        lower_case = lower_case.replace('  ', ' ')
    while lower_case.endswith(' '):
        lower_case = lower_case[:-1]

    if lower_case.endswith('.'):
        lower_case = lower_case[:-1]
    last_sentence = lower_case.split('.')[-1]
    lower_case = lower_case.replace('.','')
    yes_found = "yes" in last_sentence
    no_found = "no" in last_sentence

    answer = "unknown"

    if yes_found and no_found:
        yes_found = matching_word_found(last_sentence, "yes")
        no_found = matching_word_found(last_sentence, "no")


    if yes_found and not no_found:
        answer = "yes"
    elif no_found and not yes_found:
        answer = "no"

    else:
        yes_found = "answer: yes" in lower_case or "answer is: yes" in lower_case or "answer is yes" in lower_case
        no_found = "answer: no" in lower_case or "answer is: no" in lower_case or "answer is no" in lower_case
        if yes_found and not no_found:
            answer = "yes"
        elif no_found and not yes_found:
            answer = "no"

        if yes_found and no_found:
            if lower_case.endswith("yes"):
                answer = "yes"
            elif lower_case.endswith("no"):
                answer = "no"
    if answer == "unknown":
        target_ref = target.lower()
        if target_ref.endswith('\n'):
            target_ref = target_ref[:-1]
        target_ref = target_ref.split('\n')[-1]
        yes_found = "answer: yes" in target_ref or "answer is: yes" in target_ref or "answer is yes" in target_ref or "answer to the question is yes" in target_ref
        no_found = "answer: no" in target_ref or "answer is: no" in target_ref or "answer is no" in target_ref or "answer to the question is no" in target_ref
        if yes_found and not no_found:
            answer = "yes"
        elif no_found and not yes_found:
            answer = "no"


    if answer == "unknown":
        lower_case = re.sub('[!,*)@#%(&$_?.^:;"\']', '', lower_case)
        if lower_case.startswith("yes ") or lower_case.startswith(" yes ") or lower_case.startswith("answer yes ") or lower_case.startswith(" answer yes "):
            answer = "yes"
        elif lower_case.startswith("no ") or lower_case.startswith(" no ") or lower_case.startswith("answer no ") or lower_case.startswith(" answer no "):
            answer = "no"

    return{
        "answer_given": answer,
        "correct": answer == gt.lower()
    }

for run_number in range(0,5):
    for model_name in ["gemma-2-9b","granite","llama-3.1-8b","llama-3.1-8b-prop-logic-lora","llama-3.2-1b","llama-3-8B-informalization", "llama-3-70b","ministral","phi-3.5-mini","qwen-2.5-1.5b","qwen-2.5-14b","yi-1.5-34b","gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini", "phi-3", "mistral", "llama-3-8b"]:
    #for model_name in ["phi-3"]:
        with open("final_final_data/Multi-LogiEval_verified/Multi-LogiEval/run%i/dataset_%s.json"%(run_number,model_name), 'r') as dataset_file:
            dataset = json.load(dataset_file)
            for idx in tqdm(range(len(dataset['data']))):
                dataset['data'][idx]["verification"] = verify(dataset['data'][idx]["llm_answer"],dataset['data'][idx]["answer"])
        with open("final_final_data/Multi-LogiEval_verified/Multi-LogiEval/run%i/dataset_%s.json"%(run_number,model_name), 'w') as output_file:
            json.dump(dataset, output_file, indent=4, ensure_ascii=False)