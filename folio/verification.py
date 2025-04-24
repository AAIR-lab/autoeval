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
    lower_case = re.sub('[!,*)@#%(&$_?^:;#"\'\n\"]', ' ', lower_case)
    while '  ' in lower_case:
        lower_case = lower_case.replace('  ', ' ')
    while lower_case.endswith(' '):
        lower_case = lower_case[:-1]

    if lower_case.endswith('.'):
        lower_case = lower_case[:-1]
    last_sentence = lower_case.split('.')[-1]
    lower_case = lower_case.replace('.','')
    true_found = "true" in last_sentence
    false_found = "false" in last_sentence
    uncertain_found = "uncertain" in last_sentence

    answer = "unknown"

    if (true_found and false_found) or (true_found and uncertain_found) or (false_found and uncertain_found):
        true_found = matching_word_found(last_sentence, "true")
        false_found = matching_word_found(last_sentence, "false")
        uncertain_found = matching_word_found(last_sentence, "uncertain")

    if true_found and not false_found and not uncertain_found:
        answer = "true"
    elif false_found and not true_found and not uncertain_found:
        answer = "false"
    elif uncertain_found and not true_found and not false_found:
        answer = "uncertain"

    else:
        true_found = "answer: true" in lower_case or "answer is: true" in lower_case or "answer is true" in lower_case
        false_found = "answer: false" in lower_case or "answer is: false" in lower_case or "answer is false" in lower_case
        uncertain_found = "answer: uncertain" in lower_case or "answer is: uncertain" in lower_case or "answer is uncertain" in lower_case
        if true_found and not false_found and not uncertain_found:
            answer = "true"
        elif false_found and not true_found and not uncertain_found:
            answer = "false"
        elif uncertain_found and not true_found and not false_found:
            answer = "uncertain"

        if (true_found and false_found) or (true_found and uncertain_found) or (false_found and uncertain_found):
            if lower_case.endswith("true"):
                answer = "true"
            elif lower_case.endswith("false"):
                answer = "false"
            elif lower_case.endswith("uncertain"):
                answer = "uncertain"
    if answer == "unknown":
        target_ref = target.lower()
        if target_ref.endswith('\n'):
            target_ref = target_ref[:-1]
        target_ref = target_ref.split('\n')[-1]
        true_found = "answer: true" in target_ref or "answer is: true" in target_ref or "answer is true" in target_ref or "answer to the question is true" in target_ref or "conclusion is true" or "overall conclusion true" or "conclusion as true"
        false_found = "answer: false" in target_ref or "answer is: false" in target_ref or "answer is false" in target_ref or "answer to the question is false" in target_ref or "conclusion is false" or "overall conclusion false" or "conclusion is false"
        uncertain_found = "answer: uncertain" in target_ref or "answer is: uncertain" in target_ref or "answer is uncertain" in target_ref or "answer to the question is uncertain" in target_ref or "conclusion is uncertain" or "overall conclusion uncertain" or "conclusion as uncertain"
        if true_found and not false_found and not uncertain_found:
            answer = "true"
        elif false_found and not true_found and not uncertain_found:
            answer = "false"
        elif uncertain_found and not true_found and not false_found:
            answer = "uncertain"

    if answer == "unknown":
        target_ref = target.lower()
        target_ref = re.sub('[!,*)@#%(&$_?^;#"\'\n\"]', '', target_ref)
        if target_ref.endswith('\n'):
            target_ref = target_ref[:-1]
        target_ref = target_ref.replace(" ","")
        true_found = "answer:true" in target_ref or "answeris:true" in target_ref or "answeristrue" in target_ref or "answertothequestionistrue" in target_ref or "conclusionistrue" in target_ref or "overallconclusiontrue" in target_ref or "conclusionastrue" in target_ref
        false_found = "answer:false" in target_ref or "answeris:false" in target_ref or "answerisfalse" in target_ref or "answertothequestionisfalse" in target_ref or "conclusionisfalse" in target_ref or "overallconclusionfalse" in target_ref or "conclusionisfalse" in target_ref
        uncertain_found = "answer:uncertain" in target_ref or "answeris:uncertain" in target_ref or "answerisuncertain" in target_ref or "answertothequestionisuncertain" in target_ref or "conclusionisuncertain" in target_ref or "overallconclusionuncertain" in target_ref or "conclusionasuncertain" in target_ref
        if true_found and not false_found and not uncertain_found:
            answer = "true"
        elif false_found and not true_found and not uncertain_found:
            answer = "false"
        elif uncertain_found and not true_found and not false_found:
            answer = "uncertain"

    if answer == "unknown":
        lower_case = re.sub('[!,*)@#%(&$_?.^:;"\']', '', lower_case)
        if lower_case.startswith("true ") or lower_case.startswith(" true ") or lower_case.startswith("answer true ") or lower_case.startswith(" answer true "):
            answer = "true"
        elif lower_case.startswith("false ") or lower_case.startswith(" false ") or lower_case.startswith("answer false ") or lower_case.startswith(" answer false "):
            answer = "false"
        elif lower_case.startswith("uncertain ") or lower_case.startswith(" uncertain ") or lower_case.startswith("answer uncertain ") or lower_case.startswith(" answer uncertain "):
            answer = "uncertain"

    return{
        "answer_given": answer,
        "correct": answer == gt.lower()
    }

for run_number in range(5):
    for model_name in ["gemma-2-9b","granite","llama-3.1-8b","llama-3.1-8b-prop-logic-lora","llama-3.2-1b","llama-3-8B-informalization", "llama-3-70b","ministral","phi-3.5-mini","qwen-2.5-1.5b","qwen-2.5-14b","yi-1.5-34b","gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini", "phi-3", "mistral", "llama-3-8b"]:
        with open("final_final_data/folio_results_verified/results/run%i/folio_dataset_%s.json"%(run_number,model_name), 'r') as dataset_file:
            dataset = json.load(dataset_file)
            for idx in tqdm(range(len(dataset['data']))):
                dataset['data'][idx]["verification"] = verify(dataset['data'][idx]["llm_answer"],dataset['data'][idx]["label"])
        with open("final_final_data/folio_results_verified/results/run%i/folio_dataset_%s.json"%(run_number,model_name), 'w') as output_file:
            json.dump(dataset, output_file, indent=4, ensure_ascii=False)