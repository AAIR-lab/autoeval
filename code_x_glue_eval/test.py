import json

for run_number in range(5):
    with open("code_x_glue_eval/results/run%i/code_x_glue_dataset_gpt-3.5-turbo.json"%(run_number),'r') as input_file:
        data_file = json.load(input_file)

    for x in reversed(range(1756,3000)):
        data_file['data'][x]["llm_answer"] = data_file['data'][x - 1]["llm_answer"]
        data_file['data'][x]["cost"] = data_file['data'][x - 1]["cost"]
        data_file['data'][x]["conversation"] = data_file['data'][x - 1]["conversation"]

    del data_file['data'][1755]
    data_file["javascript"].remove(1755)
    data_file["metadata"]["javascript"] = 499

    # for idx, data in enumerate(reversed(data_file['data'])):
    #     messages.append(make_prompt(data["language"],data["code"]))

    with open("code_x_glue_eval/results/run%i/code_x_glue_dataset_gpt-3.5-turbo.json"%(run_number),'w+') as output_file:
        json.dump(data_file,output_file, indent=4, ensure_ascii=False)

