from nlfs import llm_agents
import json
import time
from tqdm import tqdm


def make_prompt(context,question):
    output = "Given the context that contains rules of logical reasoning in natural language and question, perform step-by-step reasoning to answer the question. Based on context and reasoning steps, answer the question ONLY in ‘yes’ or ‘no.’ Please use the below format:\n Context: %s\n Question: %s\n Reasoning steps: [generate step-by-step reasoning]\n Answer: Yes/No"%(context,question)

    return output

def run_multi_logieval(target_dataset, llm_name):

    path, datatype = target_dataset.rsplit('.',1)


    result_dataset = "%s_%s.%s"%(path,llm_name,datatype)

    with open(target_dataset, 'r') as dataset_file:
        dataset = json.load(dataset_file)
    dataset['llm_name'] = llm_name

    messages = []

    for idx, data in enumerate(dataset['data']):
        messages.append(make_prompt(data["context"],data["question"]))

    results = None
    costs = None

    if llm_name.startswith("gpt") and llm_name != "gpt-4o1-mini":
        agent = llm_agents.get_batch_llm_agent(llm_name)
        print("Starting batch.")
        batch_name = agent.start_batch(messages)
        print("batch name: %s"%(batch_name))

        while results is None:
            print("Sleep 60 seconds...")
            time.sleep(60)
            status = agent.check_batch_status(batch_name)
            print("status: %s"%(status))
            if status == 'completed':
                results, costs = agent.extract_batch_data(batch_name)
    else:
        agent = llm_agents.get_llm_agent(llm_name)
        results = [None] * len(messages)
        costs = [0.0] * len(messages)

        print("progress:")
        for idx in tqdm(range(len(messages))):
            results[idx], costs[idx] = agent.send_message(messages[idx])
            agent.reset()

    for current, response, cost, prompt in zip(dataset['data'], results, costs, messages):
        current["llm_answer"] = response
        current["cost"] = cost
        current["conversation"] = [{"role": "user","content":prompt},{"role":"assistant","content":response}]


    with open(result_dataset, 'w') as output_file:
        dataset = json.dump(dataset, output_file, indent=4, ensure_ascii=False)

import sys
try:
    target_dataset = sys.argv[1]
    llm_name = sys.argv[2]
except IndexError as e:

    target_dataset = "/tmp/logieval/run0/dataset.json"
    llm_name = "gpt-3.5-turbo"

run_multi_logieval(target_dataset, llm_name)