from nlfs.prompts.prompt import LogicInterpretationPrompt, LogicTranslationPrompt
from nlfs import llm_agents
import json
import time
from tqdm import tqdm
import os

def run_folio(target_dataset, llm_name):

    path, datatype = target_dataset.rsplit('.',1)

    result_dataset = "%s_%s.%s"%(path,llm_name,datatype)

    nlfs_prompt = LogicInterpretationPrompt()
    fsnl_prompt = LogicTranslationPrompt()

    with open(target_dataset, 'r') as dataset_file:
        dataset = json.load(dataset_file)
    dataset['llm_name'] = llm_name

    nlfs_messages = []
    fsnl_messages = []
    for idx, data in enumerate(dataset['data']):
        nlfs_msg = nlfs_prompt.generate(data['nl'], True, data)
        fsnl_msg = fsnl_prompt.generate(data, True)
        nlfs_messages.append(nlfs_msg)
        fsnl_messages.append(fsnl_msg)

    nlfs_results = None
    nlfs_costs = None
    fsnl_results = None
    fsnl_costs = None

    if llm_name.startswith("gpt") and llm_name != "gpt-4o1-mini":
        agent = llm_agents.get_batch_llm_agent(llm_name)
        print("Starting nl->fs batch.")
        nlfs_batch_name = agent.start_batch(nlfs_messages)
        print("nl-fs batch name: %s"%(nlfs_batch_name))
        print("Starting fs->nl batch.")
        fsnl_batch_name = agent.start_batch(fsnl_messages)
        print("fs-nl batch name: %s" % (fsnl_batch_name))

        while nlfs_results is None or fsnl_results is None:
            print("Sleep 60 seconds...")
            time.sleep(60)
            if nlfs_results is None:
                status = agent.check_batch_status(nlfs_batch_name)
                print("NL->FS status: %s"%(status))
                if status == 'completed':
                    nlfs_results, nlfs_costs = agent.extract_batch_data(nlfs_batch_name)
            if fsnl_results is None:
                status = agent.check_batch_status(fsnl_batch_name)
                print("FS->NL status: %s"%(status))
                if status == 'completed':
                    fsnl_results, fsnl_costs = agent.extract_batch_data(fsnl_batch_name)
    else:
        agent = llm_agents.get_llm_agent(llm_name)
        nlfs_results = [None] * len(nlfs_messages)
        nlfs_costs = [0.0] * len(nlfs_messages)
        fsnl_results = [None] * len(fsnl_messages)
        fsnl_costs = [0.0] * len(fsnl_messages)

        print("NL->FS progress:")
        for idx in tqdm(range(len(nlfs_messages))):
            nlfs_results[idx], nlfs_costs[idx] = agent.send_message(nlfs_messages[idx])
            agent.reset()
        print("FS->NL progress:")
        for idx in tqdm(range(len(fsnl_messages))):
            fsnl_results[idx], fsnl_costs[idx] = agent.send_message(fsnl_messages[idx])
            agent.reset()

    for current, fsnl, fsnl_cost, fsnl_prompt, nlfs, nlfs_cost, nlfs_prompt in zip(dataset['data'], fsnl_results,fsnl_costs,fsnl_messages,nlfs_results,nlfs_costs,nlfs_messages):
        current["llm_fs"] = nlfs
        current["llm_nl"] = fsnl
        current["nlfs_cost"] = nlfs_cost
        current["fsnl_cost"] = fsnl_cost
        current["nlfs_conversation"] = [{"role": "user","content":nlfs_prompt},{"role":"assistant","content":nlfs}]
        current["fsnl_conversation"] = [{"role": "user", "content": fsnl_prompt}, {"role": "assistant", "content": fsnl}]


    with open(result_dataset, 'w') as output_file:
        dataset = json.dump(dataset, output_file, indent=4, ensure_ascii=False)

import sys
try:
    target_dataset = sys.argv[1]
    llm_name = sys.argv[2]
except IndexError as e:

    target_dataset = "/tmp/folio/run0/dataset.json"
    llm_name = "gpt-3.5-turbo"

run_folio(target_dataset, llm_name)
