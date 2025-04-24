from datasets import load_dataset
from nlfs.verifier.logic import str_to_expr, verify
from nltk.sem.logic import *
import random
import json


access_token = ""
from huggingface_hub import login
login(access_token)

random_seed = 67148797
random.seed(random_seed)
output_file = "temp.json"


def count_components(expr):
    assert(isinstance(expr,Expression))
    quantifier_count = 0
    operator_count = 0
    predicate_count = 0
    node_stack = [expr]

    while len(node_stack) > 0:
        current_expr = node_stack.pop()
        if isinstance(current_expr, ExistsExpression) or isinstance(current_expr, AllExpression):
            quantifier_count += 1
            node_stack.append(current_expr.term)
        elif isinstance(current_expr, ApplicationExpression):
            predicate_count += 1
            for current_arg in current_expr.args:
                node_stack.append(current_arg)
        elif isinstance(current_expr, AndExpression) or isinstance(current_expr, ImpExpression) or isinstance(current_expr, IffExpression) or isinstance(current_expr, OrExpression) or isinstance(current_expr, EqualityExpression):
            operator_count += 1
            node_stack.append(current_expr.first)
            node_stack.append(current_expr.second)

        elif isinstance(current_expr, NegatedExpression):
            operator_count += 1
            node_stack.append(current_expr.term)
    return operator_count, predicate_count, quantifier_count


data = load_dataset("yale-nlp/FOLIO")

output = {}
output['info'] = {}
output['info']['type'] = "FOLIO premises dataset"
output['info']['seed'] = random_seed
output['info']['filter_field'] = "num_operators"
output['data'] = []
output['metadata'] = {}
output['num_predicates'] = {}
output['num_operators'] = {}
output['num_quantifiers'] = {}


added = []

for i in range(len(data['train']['premises-FOL'])):
    if data['train']['premises'][i] in added:
        continue
    added.append(data['train']['premises'][i])
    nl_set = data['train']['premises'][i].split("\n")
    fs_set = data['train']['premises-FOL'][i].split("\n")
    if len(nl_set) != len(fs_set):
        continue
    successful_pairs = []
    for nl, fs in zip(nl_set, fs_set):
        try:
            expression = str_to_expr(fs)
            successful_pairs.append((nl,fs, expression))
        except:
            continue
    
    if len(successful_pairs) < 2:
        continue

    for current_pair in successful_pairs:
        chosen_example_pair = random.choice([x for x in successful_pairs if x[1] != current_pair[1]])
        data_dict = {}
        data_dict['idx'] = len(output['data'])
        data_dict['fs'] = current_pair[1]
        data_dict['nl'] = current_pair[0]
        data_dict['example_nl'] = chosen_example_pair[0]
        data_dict['example_fs'] = chosen_example_pair[1]
        data_dict['num_operators'], data_dict['num_predicates'], data_dict['num_quantifiers'] = count_components(current_pair[2])
        
        for metric in ['num_operators','num_predicates','num_quantifiers']:
            metric_key = str(data_dict[metric])
            if metric_key not in output[metric]:
                output[metric][metric_key] = []
            output[metric][metric_key].append(data_dict['idx'])

        output['data'].append(data_dict)

for metric in ['num_operators','num_predicates','num_quantifiers']:
    output['metadata'][metric] = {key:len(output[metric][key]) for key in output[metric]}
with open(output_file,"w+") as ofile:
    json.dump(output,ofile, indent=4, ensure_ascii=False)
