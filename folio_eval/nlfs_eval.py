import evaluate
import json
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import argparse
from nlfs import alg
from nlfs.verifier import logic
import os

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

def run_nlfs_eval(dataset_file):

    with open(dataset_file, 'r') as fh:
        dataset = json.load(fh)

    for idx in tqdm(range(len(dataset['data']))):
        dataset['data'][idx]["nl_evaluation"] = calculate_score(
            dataset['data'][idx]["llm_nl"],dataset['data'][idx]["nl"])

    with open(dataset_file, 'w') as fh:
        json.dump(dataset, fh, indent=4, ensure_ascii=False)

    alg.verify(dataset_file, logic.verify, stdout=open(os.devnull, "w"))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-file", type=str, required=True,
        help="The filepath to the dataset")

    args = parser.parse_args()
    run_nlfs_eval(args.dataset_file)
    