from datasets import load_dataset
import re
import random
import json

def clean_code(code):
    code = re.sub(r"//.*|#.*", "", code)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code = re.sub(r"'''(.*?)'''|\"\"\"(.*?)\"\"\"", "", code, flags=re.DOTALL)
    return code

sample_size = 500
random_seed = 92374078
save_location = "code_x_glue_eval/results/code_x_glue_dataset.json"

output_dataset = {"info": {
        "type": "code_x_glue_eval",
        "seed": random_seed,
        "sample_size": sample_size
    },
    "data": [],
    "metadata": {},
    "go": [],
    "ruby": [],
    "java": [],
    "javascript": [],
    "php": [],
    "python": []
}

random.seed(random_seed)
for language in ["go","ruby","java","javascript","php","python"]:

    ds = load_dataset("google/code_x_glue_ct_code_to_text", language)['test']
    if len(ds) > sample_size:
        ds = ds.select(random.sample(range(len(ds)), k=sample_size))

    for x in ds:
        current_sample = {
            "idx": len(output_dataset['data']),
            "language": language,
            "code": clean_code(x['code']),
            "docstring": x['docstring']
        }
        output_dataset[language].append(current_sample['idx'])
        output_dataset['data'].append(current_sample)

    output_dataset['metadata'][language] = len(output_dataset[language])

with open(save_location,'w+') as output_file:
    json.dump(output_dataset, output_file, indent=4, ensure_ascii=False)
