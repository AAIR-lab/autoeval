import evaluate
import json
from nltk.tokenize import word_tokenize
from tqdm import tqdm

bleu_evaluator = evaluate.load("bleu")
rouge_evaluator = evaluate.load('rouge')
meteor_evaluator = evaluate.load('meteor')
bertscore_evaluator = evaluate.load("bertscore")


def calculate_score(prediction,gt,model_name="microsoft/deberta-xlarge-mnli"):
    output = {}
    output['bleu'] = bleu_evaluator.compute(predictions=[prediction], references=[gt], tokenizer=word_tokenize)
    output['rouge'] = rouge_evaluator.compute(predictions=[prediction],references=[gt])
    output['meteor'] = meteor_evaluator.compute(predictions=[prediction],references=[gt])
    output['bert_score'] = bertscore_evaluator.compute(predictions=[prediction],references=[gt], model_type=model_name)
    return output

BASE_DIR = "/tmp/rkaria/codexglue"
RUNS = range(5)

import sys
try:
    MODEL_NAME = sys.argv[1]
except Exception:
    assert False
    
for run in RUNS:

    dataset_filepath = "%s/run%u/dataset_%s.json" % (BASE_DIR,
        run,
        MODEL_NAME)
    
    with open(dataset_filepath, 'r') as dataset_file:
        dataset = json.load(dataset_file)

    for idx in tqdm(range(len(dataset['data']))):
        dataset['data'][idx]["nl_evaluation"] = calculate_score(
            dataset['data'][idx]["llm_answer"],
            dataset['data'][idx]["docstring"])

    with open(dataset_filepath, 'w') as output_file:
        json.dump(dataset, output_file, indent=4, ensure_ascii=False)