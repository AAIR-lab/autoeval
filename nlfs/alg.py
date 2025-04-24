from nlfs import llm_agents
import numpy as np
import json
from pathlib import Path
from nlfs.prompts.prompt import Prompt
import time
import concurrent.futures
from datetime import timedelta
from nlfs.verifier import logic
from nlfs.prompts.prompt import LogicTranslationPrompt
from nlfs.prompts.prompt import LogicInterpretationPrompt
from nlfs.prompts.regex_prompt import RegexTranslationPrompt
from nlfs.prompts.regex_prompt import RegexInterpretationPrompt
from nlfs.verifier import regex
import time
import pathlib
import sys
import pebble
import os

def nlfs(fs, llm_agent_name,
                 translation_prompt,
                 interpretation_prompt,
                 use_prompt_examples):

    start_time = time.time()
    llm_agent = llm_agents.get_llm_agent(llm_agent_name)

    fs_to_nl = translation_prompt.generate(fs,use_prompt_examples)

    conversations = []
    nl, _ = llm_agent.send_message(fs_to_nl)
    conversations += llm_agent.get_conversation_history()

    conversations.append("RESET")
    llm_agent.reset()
    nl_to_fs = interpretation_prompt.generate(nl,use_prompt_examples,fs_datum=fs)
    llm_fs, _ = llm_agent.send_message(nl_to_fs)
    conversations += llm_agent.get_conversation_history()

    return llm_fs, \
        llm_agent.get_total_cost(), \
        conversations, \
        start_time


def get_elapsed_timedelta(start_time, end_time=None):

    if start_time is None:
        return "--:--:--"

    if end_time is None:
        end_time = time.time()

    total_seconds = round(end_time - start_time)
    return timedelta(seconds=total_seconds)

def perform_nlfs(dataset_filepath,
                 gpt_agent_name,
                 translation_prompt,
                 interpretation_prompt,
                 filter_field=None,
                 store_conversation_history=False,
                 stdout=sys.stdout,
                 use_prompt_examples=False):

    if isinstance(stdout, str):
        stdout = open(stdout, "w", buffering=1)

    dataset = get_dataset(dataset_filepath)
    if filter_field is None:
        filter_field = dataset["info"]["filter_field"]

    fs_data = dataset["data"]

    results_json = {
        "info": dataset["info"],
        "verified": False,
        "filter_field": filter_field,
        "data": fs_data
    }

    thread_pool = concurrent.futures.ThreadPoolExecutor()
    futures = {}
    for idx, fs_datum in enumerate(fs_data):

        nlfs_args = (fs_datum, gpt_agent_name,
                     translation_prompt,
                     interpretation_prompt,
                     use_prompt_examples)
        future = thread_pool.submit(nlfs, *nlfs_args)
        futures[future] = idx

    total_completed = 0
    total_cost = 0
    total_errors = 0
    start_time = time.time()
    conversations = []
    for future in concurrent.futures.as_completed(futures):

        total_completed += 1
        idx = futures[future]
        future_start_time = None
        cost = 0
        error = None
        try:
            llm_fs, cost, conversations, future_start_time = future.result()
            total_cost += cost

        except Exception as e:
            cost = 0
            conversation = []
            llm_fs = None
            error = str(e)

        total_errors += error is not None
        future_elapsed_time = get_elapsed_timedelta(
            future_start_time)

        fs_data[idx]["db_idx"] = fs_data[idx]["idx"]
        fs_data[idx]["idx"] = idx
        fs_data[idx]["llm_fs"] = llm_fs
        fs_data[idx]["cost"] = cost
        fs_data[idx]["error"] = error

        if store_conversation_history:
            fs_data[idx]["conversations"] = conversations

        print("LLM Generated: %4s/%-4s | "
              "idx: %4s | "
              "cost: %6.2f | "
              "time: %9s | "
              "total_cost: %6.2f | "
              "error_rate: %6.2f%% | "
              "elapsed_time: %9s |" % (
            total_completed, len(fs_data), idx, cost,
            future_elapsed_time,
            total_cost,
            total_errors * 100 / total_completed,
            get_elapsed_timedelta(start_time)), file=stdout)

    new_filepath = save_dataset(dataset_filepath, results_json, "_nlfs_%s" % (gpt_agent_name))
    return results_json, new_filepath, total_cost

def get_dataset(dataset_filepath):

    dataset_filepath = pathlib.Path(dataset_filepath).as_posix()
    dataset = json.load(open(dataset_filepath, "r"))
    return dataset

def save_dataset(dataset_filepath, dataset, suffix=""):

    dataset_filepath = pathlib.Path(dataset_filepath)
    parent_filepath = dataset_filepath.parent.as_posix()
    dataset_filename = dataset_filepath.name.replace(".json", "")

    dataset_filepath = "%s/%s%s.json" % (parent_filepath,
                                         dataset_filename, suffix)

    with open(dataset_filepath, "w") as fh:
        json.dump(dataset, fh, indent=4, ensure_ascii=False)

    return dataset_filepath

def verify(dataset_filepath, verify_func, stdout=sys.stdout,
           timeout=600, llm_verify_model=None, dataset_type=None):

    if isinstance(stdout, str):
        stdout = open(stdout, "w", buffering=1)

    dataset = get_dataset(dataset_filepath)
    process_pool = pebble.ProcessPool()

    fs_data = dataset["data"]
    args_list = []
    if llm_verify_model is None:
        for idx, fs_datum in enumerate(fs_data):

            fs = fs_datum["fs"]
            llm_fs = fs_datum["llm_fs"]
            verify_func_args = (fs, llm_fs)
            args_list.append(verify_func_args)
    else:
        for idx, fs_datum in enumerate(fs_data):
            fs = fs_datum["fs"]
            llm_fs = fs_datum["llm_fs"]
            verify_func_args = (fs, llm_fs, llm_verify_model, dataset_type)
            args_list.append(verify_func_args)

    future = process_pool.map(verify_func,
                              args_list, timeout=timeout)
    iterator = future.result()

    start_time = time.time()
    total_completed = 0
    total_errors = 0
    total_equivalent = 0

    idx = 0
    while True:

        try:
            total_completed += 1
            err = None
            future_elapsed_time = None
            is_equivalent = None
            verify_prompt = None
            llm_response = None

            if llm_verify_model is None:
                is_equivalent, future_start_time, err = next(iterator)
            else:
                is_equivalent, verify_prompt, llm_response, future_start_time, err = next(iterator)

            total_errors += err is not None

            if is_equivalent is not None:
                total_equivalent += is_equivalent

            future_elapsed_time = get_elapsed_timedelta(future_start_time)
        except StopIteration:

           break
        except Exception as e:
           err = str(e)

        if llm_verify_model is None:
            fs_data[idx]["verification"] = {
                "has_error": err,
                "is_equivalent": is_equivalent,
                "time": "%s" % (future_elapsed_time)
            }
        else:
            if "llm_verification" not in fs_data[idx]:
                fs_data[idx]["llm_verification"] = {}
            fs_data[idx]["llm_verification"][llm_verify_model] = {
                "has_error": err,
                "is_equivalent": is_equivalent,
                "time": "%s" % (future_elapsed_time),
                "conversations": [
                {
                    "role": "user",
                    "content": verify_prompt
                },
                {
                    "role": "assistant",
                    "content": llm_response
                }
                ]
            }
        idx += 1

        print("Verified: %4s/%-4s | "
              "idx: %4s | "
              "is_equivalent: %5s | "
              "time: %9s | "
              "error_rate: %6.2f%% | " 
              "accuracy: %6.2f%% | "
              "total_time: %9s |" % (
            total_completed, len(fs_data), idx,
            is_equivalent,
            future_elapsed_time,
            total_errors * 100 / total_completed,
            total_equivalent * 100 / total_completed,
            get_elapsed_timedelta(start_time)), file=stdout, flush=True)

    # Kill any leftover prover9 processes.
    os.system("pkill prover9")

    dataset["verified"] = True

    if dataset_filepath.endswith("verified.json"):
        new_filepath = save_dataset(dataset_filepath, dataset, suffix="_verified")
    else:
        new_filepath = save_dataset(dataset_filepath, dataset)
    return dataset, new_filepath

def get_logic_params(logic_type):

    assert logic_type in ["ksat", "plogic", "fol"]

    dataset_filepath = "/tmp/results/%s/dataset.json" % (
        logic_type)
    translation_prompt = LogicTranslationPrompt()
    interpretation_prompt = LogicInterpretationPrompt(
        is_first_order=logic_type == "fol")
    verify_func = logic.verify
    filter_field = "num_operators"

    return dataset_filepath, translation_prompt, \
        interpretation_prompt, verify_func, \
        filter_field

def get_regex_params():

    dataset_filepath = "/tmp/results/regex/dataset.json"
    translation_prompt = RegexTranslationPrompt()
    interpretation_prompt = RegexInterpretationPrompt()
    verify_func = regex.verify
    filter_field = "depth"

    return dataset_filepath, translation_prompt, \
        interpretation_prompt, verify_func, \
        filter_field

if __name__ == "__main__":

    dataset_type = "plogic"
    llm_agent_name = "gpt-3.5-turbo"

    if len(sys.argv) == 3:
        dataset_type = sys.argv[1]
        llm_agent_name = sys.argv[2]

    if dataset_type in ["ksat", "plogic", "fol"]:

        dataset_filepath, translation_prompt, \
            interpretation_prompt, verify_func, \
            filter_field = get_logic_params(dataset_type)
    else:

        dataset_filepath, translation_prompt, \
            interpretation_prompt, verify_func, \
            filter_field = get_regex_params()

    start_time = time.time()

    _, dataset_filepath = perform_nlfs(dataset_filepath,
                 llm_agent_name,
                 translation_prompt,
                 interpretation_prompt,
                 filter_field,
                 True)

    _, _ = verify(dataset_filepath, verify_func)

    print("Total time: %s" % get_elapsed_timedelta(start_time))


def perform_batch_nlfs(dataset_filepath,
                 gpt_agent_name,
                 translation_prompt,
                 interpretation_prompt,
                 filter_field=None,
                 store_conversation_history=False,
                 stdout=sys.stdout,
                 use_prompt_examples=False,
                 batch_log_path="results/log_batch.json"):

    llm_agent = llm_agents.get_llm_agent(llm_agent_name)

    if isinstance(stdout, str):
        stdout = open(stdout, "w")

    dataset = get_dataset(dataset_filepath)
    if filter_field is None:
        filter_field = dataset["info"]["filter_field"]

    fs_data = dataset["data"]

    if os.path.exists(batch_log_path):
        with open(batch_log_path, "r") as batch_file:
            batch_log = json.load(batch_file)
    else:
        Path(batch_log_path).mkdir(parents=True)
        batch_log = {}

    # Log format: (dataset path, llm name): (status,batch_name,dataset_file_path)
    # Statuses: NLFS, FSNL, Complete, Error

    lookup_key = (dataset_filepath,gpt_agent_name)
    if lookup_key in batch_log:
        if batch_log[lookup_key][0] == "FSNL":
            batch_status = llm_agent.check_batch_status(batch_log[lookup_key][1])
            if batch_status == 'complete':
                # FS->NL completed meaning that it is time to collect that data and start NL->FS.
                fsnl_responses, costs = llm_agent.extract_batch_data(batch_log[lookup_key][1])
                with open(batch_log[lookup_key][2],"r") as results_file:
                    results_json = json.load(results_file)
                messages = []
                for current_response, current_result, current_cost in zip(fsnl_responses,results_json["data"], costs):
                    messages.append(interpretation_prompt.generate(current_response,use_prompt_examples,fs_datum=current_result))
                    if store_conversation_history:
                        current_result["conversation"].append({
                            "role":"assistant",
                            "content":current_response
                        })
                        current_result["conversation"].append("RESET")
                        current_result["conversation"].append({
                            "role":"user",
                            "content":messages[-1]
                        })
                    current_result["cost"] = current_cost

                batch_name = llm_agent.start_batch(messages)    
                with open(batch_log[lookup_key][2],"w") as results_file:
                    results_json = json.dump(results_file,results_json)
                batch_log[lookup_key] = ("NLFS",batch_name,batch_log[lookup_key][2])
                

            elif batch_status in ['failed','expired','cancelling','cancelled']:
                # TODO: add some catch for this case.
                batch_log[lookup_key] = ("Error",None,None)
        elif batch_log[lookup_key][0] == "NLFS":
            batch_status = llm_agent.check_batch_status(batch_log[lookup_key][1])
            if batch_status == 'complete':
                # NL->FS completed meaning that it is time to collect that data and finish the dataset file.
                nlfs_responses, costs = llm_agent.extract_batch_data(batch_log[lookup_key][1])
                with open(batch_log[lookup_key][2],"r") as results_file:
                    results_json = json.load(results_file)
                for current_response, current_result, current_cost in zip(nlfs_responses,results_json["data"], costs):
                    if store_conversation_history:
                        current_result["conversation"].append({
                            "role":"assistant",
                            "content":current_response
                        })
                    current_result["db_idx"] = current_result["idx"]
                    current_result["llm_fs"] = current_response
                    current_result["cost"] += current_cost
                    current_response["error"] = None
                    #TODO: handle cases the LLM has an error.
                with open(batch_log[lookup_key][2],"w") as results_file:
                    results_json = json.dump(results_file,results_json)
                batch_log[lookup_key] = ("Complete",None,None)

            elif batch_status in ['failed','expired','cancelling','cancelled']:
                batch_log[lookup_key] = ("Error",None,None)
    else:
        # Means brand new dataset so starts FS->NL.
        messages = []
        results_json = {
            "info": dataset["info"],
            "verified": False,
            "filter_field": filter_field,
            "data": fs_data
        }

        for idx, fs_datum in enumerate(fs_data):
            messages.append(translation_prompt.generate(fs_datum,use_prompt_examples))
            if store_conversation_history:
                results_json["data"][idx]["conversations"] = [{"role":"user",
                                                            "content":messages[-1]}]
        batch_name = llm_agent.start_batch(messages)
        new_filepath = save_dataset(dataset_filepath, results_json, "_nlfs_%s" % (gpt_agent_name))
        batch_log[lookup_key] = ("FSNL",batch_name,new_filepath)

    with open(batch_log_path, "w") as batch_file:
        json.dump(batch_log,batch_file)    