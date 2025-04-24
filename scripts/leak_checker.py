import json
import argparse

operator_list = ["∧","¬","∀","∃"]

def check_file_leak(path,operator_list):
    with open(path,"r") as input_file:
        file = json.load(input_file)
        for current in file["data"]:
            current["has_fs_leak"] = any(op in current["conversations"][1]["content"] for op in operator_list)
            
    with open(path,"w") as output_file:
        json.dump(file, output_file, indent=4, ensure_ascii=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default="/tmp/results/",
                        required=False)
    parser.add_argument("--runs", type=int, nargs="+",
                        default=range(10))
    parser.add_argument("--datasets", type=str, nargs="+",
                        default= ["ksat", "plogic", "fol", "fol_human", "regex"])
    
    MODELS = ["gemma-2-9b","granite","llama-3.1-8b","llama-3.1-8b-prop-logic-lora","llama-3.2-1b","llama-3-8B-informalization", "llama-3-70b","ministral","phi-3.5-mini","qwen-2.5-1.5b","qwen-2.5-14b","yi-1.5-34b","gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini", "phi-3", "mistral", "llama-3-8b","claude","gpt-4o1"]
        #["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini", "phi-3", "mistral", "llama-3-8b","claude","gpt-4o1"]
    parser.add_argument("--models", type=str, nargs="+",
                        default=MODELS)
    parser.add_argument("--clean", default=False, action="store_true")


    args = parser.parse_args()

    for current_batch in range(10):
        for dataset_type in args.datasets:
            for model in args.models:
                current_path = "%s/batch%i/%s/%s.verified.json"%(args.base_dir,current_batch,dataset_type,model)
                print(current_path)
                try:
                    if dataset_type == "regex":
                        check_file_leak(current_path,["*"])
                    else:
                        check_file_leak(current_path,operator_list)
                except Exception as e:
                    print("Error running!")
                    continue
