import json

models = ["claude","gpt-3.5-turbo","gpt-4o","llama-3-8b","mistral","phi-3"]
logic_operators = ["∧","¬","∀","∃","*","+","v"]
regex_operators = ["*","+"]

def check_file_leak(path,operator_list):
    count = 0
    total = 0
    was_correct = 0
    total_correct = 0
    with open(path,"r") as input_file:
        file = json.load(input_file)
        for current in file["data"]:
            total += 1
            original = current["fs"]
            explanation = current["conversations"][1]["content"]
            verification = current["verification"]
            if original in explanation and any(op in original for op in logic_operators):
                print(original)
                print(explanation)
                print("----")
                count += 1
                if verification["has_error"] is None:
                    if verification["is_equivalent"]:
                        was_correct += 1
            if verification["has_error"] is None:
                if verification["is_equivalent"]:
                    total_correct += 1
    return count, was_correct, total, total_correct

look_up_dictionary = {}

for i in range(10):
    for dataset in ["fol", "fol_human", "ksat", "plogic", "regex"]:
        if dataset not in look_up_dictionary:
            look_up_dictionary[dataset] = {}
        for model in models:
            c, wc, t, tc = check_file_leak("results_r0-9_verified/run%i/%s/dataset_nlfs_%s_verfied.json"%(i,dataset,model),logic_operators)
            if model not in look_up_dictionary[dataset]:
                look_up_dictionary[dataset][model] = (c, wc, t, tc)
            else:
                look_up_dictionary[dataset][model] = (c + look_up_dictionary[dataset][model][0],
                                                      wc + look_up_dictionary[dataset][model][1],
                                                      t + look_up_dictionary[dataset][model][2],
                                                      tc + look_up_dictionary[dataset][model][3])

with open("results.csv", "w+") as output:
    for current_data in look_up_dictionary:
        output.write("%s\n"%(current_data))
        for current_model in look_up_dictionary[current_data]:
            output.write("%s,%i,%i,%i,%i\n"%(current_model,look_up_dictionary[current_data][current_model][0],
                                             look_up_dictionary[current_data][current_model][1],
                                             look_up_dictionary[current_data][current_model][2],
                                             look_up_dictionary[current_data][current_model][3]))


# import nltk
#
# hypothesis = ['p','and', 'b', 'and', 'c', 'and', 'd']
# reference = ['p', 'and', 'p', 'and', 'p', 'and', 'p','and', 'b', 'and', 'c', 'and', 'd']
# #there may be several references
# BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
# print(BLEUscore)
#
# from transformers import BertTokenizer, BertForMaskedLM, BertModel
# # from bert_score import BERTScorer
# #
# # # Example texts
# # reference = "p and p"
# # candidate = "p and p"
# # # BERTScore calculation
# # scorer = BERTScorer(model_type='bert-base-uncased')
# # P, R, F1 = scorer.score([candidate], [reference])
# # print(f"BERTScore Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")