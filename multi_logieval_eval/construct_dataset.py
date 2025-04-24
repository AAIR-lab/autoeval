import json
import os

inference_rules = {
    "MP": "Modus Ponens",
    "MT": "Modus Tollens",
    "HS": "Hypothetical Syllogism",
    "DS": "Disjunctive Syllogism",
    "CD": "Constructive Dilemma",
    "DD": "Destructive Dilemma",
    "BD": "Bidirectional Dilemma",
    "CT": "Commutation",
    "DMT": "De Morgan’s Theorem",
    "CO": "Composition",
    "IM": "Importation",
    "MI": "Material Implication",
    "EG": "Existential Generalization",
    "UI": "Universal Instantiation"
}

dataset_path = "Multi-LogiEval-main/data"
save_path = "explain-llm-main/multi_logieval_eval/results/dataset.json"
output_dataset = {"info": {
        "type": "Multi-LogiEval dataset",
    },
    "data": [],
    "metadata": {},
    "predicate_logic": [],
    "first_order_logic": [],
    "depth": {
        1: [],
        2: [],
        3: [],
        4: [],
        5: []
    }}

for current_depth in range(1,6):
    for logic_type in ["fol","pl"]:
        folder_path = "%s/d%i_Data/%s"%(dataset_path,current_depth,logic_type)
        for filename in os.listdir(folder_path):
            with open(os.path.join(folder_path, filename), 'r',encoding = "ISO-8859-1") as f:
                current_file = json.load(f)
                current_rule = current_file['rule']
                for current in current_file['samples']:
                    current_point = {
                        "idx": len(output_dataset["data"]),
                        "rule": current_rule,
                        "depth": current_depth,
                        "logic": logic_type,
                        "context": current['context'],
                        "question": current['question'],
                        "answer": current['answer']
                    }
                    if logic_type == "fol":
                        output_dataset["first_order_logic"].append(current_point['idx'])
                    else:
                        output_dataset["predicate_logic"].append(current_point['idx'])
                    output_dataset["depth"][current_depth].append(current_point['idx'])
                    output_dataset["data"].append(current_point)

output_dataset["metadata"]["first_order_logic"] = len(output_dataset["first_order_logic"])
output_dataset["metadata"]["predicate_logic"] = len(output_dataset["predicate_logic"])
output_dataset["metadata"]["depth"] = {}
for i in range(1,6):
    output_dataset["metadata"]["depth"][i] = len(output_dataset["depth"][i])

with open(save_path,"w+") as output_file:
    json.dump(output_dataset,output_file, indent=4, ensure_ascii=False)


