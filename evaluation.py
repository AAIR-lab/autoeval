import argparse

from nlfs.prompts.prompt import LogicTranslationPrompt
from nlfs.prompts.prompt import LogicInterpretationPrompt
from nlfs.verifier import logic

from nlfs.prompts.regex_prompt import RegexTranslationPrompt
from nlfs.prompts.regex_prompt import RegexInterpretationPrompt
from nlfs.verifier import regex
from nlfs import alg
from nlfs.verifier import llm_verifier

import concurrent.futures
import pathlib
import os
import tqdm
import time
from datetime import timedelta
import sys

SUPPORTED_DATASETS = ["ksat", "plogic", "fol", "fol_human", "regex"]
SUPPORTED_MODELS = ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini", "claude", "phi-3", "mistral", "llama-3-8b",
    "gpt-4o1", "gpt-4o1-mini","ministral", "llama-3.1-8b", "qwen-2.5-14b", "granite", "deepseek-v2-lite", "gemma-2-9b","llama-3-8B-informalization",
    "vicuna-13b", "falcon-40b", "llama-3-70b", "yi-1.5-34b", "phi-3.5-mini",
    "llama-3.2-1b", "llama-3.1-8b-prop-logic-lora", "qwen-2.5-1.5b", "deepseek-r1"
]
SUPPORTED_PROMPTING = ["zero-shot", "few-shot"]


def get_prompts(dataset_type):

    assert dataset_type in SUPPORTED_DATASETS
    if dataset_type in ["ksat", "plogic", "fol", "fol_human"]:

        translation_prompt = LogicTranslationPrompt()
        interpretation_prompt = LogicInterpretationPrompt(
            is_first_order=dataset_type in ["fol", "fol_human"])
    else:
        translation_prompt = RegexTranslationPrompt()
        interpretation_prompt = RegexInterpretationPrompt()

    return translation_prompt, interpretation_prompt


def get_verify_func(dataset_type):

    assert dataset_type in SUPPORTED_DATASETS
    if dataset_type in ["ksat", "plogic", "fol", "fol_human"]:

        verify_func = logic.verify
    else:
        verify_func = regex.verify

    return verify_func


def get_elapsed_timedelta(start_time, end_time=None):

    if start_time is None:
        return "--:--:--"

    if end_time is None:
        end_time = time.time()

    total_seconds = round(end_time - start_time)
    return timedelta(seconds=total_seconds)


def _run_auto_nlfs(args, batch_no, dataset_type, models, total_cost,
                   prog_start_time):

    print("======= NLFS | dataset: %7s | batch: %2d | total_models: %2d | ======= "
          % (dataset_type, batch_no, len(models)))
    process_pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=args.max_nlfs_workers)

    dataset_filepath = "%s/batch%d/%s/dataset.json" % (args.base_dir,
                                                     batch_no,
                                                     dataset_type)
    dataset_filepath = os.path.abspath(os.path.expanduser(dataset_filepath))
    dataset_directory = pathlib.Path(dataset_filepath).parent

    t_prompt, i_prompt = get_prompts(dataset_type)


    futures = {}
    for model_name in models:

        stdout_filepath ="%s/nlfs.%s.log" % (dataset_directory, model_name)
        use_examples = args.prompting in ['few-shot']
        nlfs_args = (dataset_filepath,
            model_name,
            t_prompt,
            i_prompt,
            args.filter_field,
            args.store_conversation_history,
            stdout_filepath,
            use_examples)

        future = process_pool.submit(alg.perform_nlfs, *nlfs_args)
        futures[future] = model_name

    start_time = time.time()

    total_completed = 0
    total_count = len(futures)
    for future in concurrent.futures.as_completed(futures):

        total_completed += 1
        model_name = futures[future]
        elapsed_time = get_elapsed_timedelta(start_time)
        total_time = get_elapsed_timedelta(prog_start_time)
        try:
            _, _, cost = future.result()
            total_cost += cost
            print("[%2d/%-2d] NLFS SUCCESS |"
                " batch: %2d |"
                " dataset: %7s |"
                " time: %9s |"
                " cost: %6.2f |"
                " model: %-17s |" 
                " total_cost: %6.2f |"
                " total_time: %9s |"
                % (
                total_completed, total_count,
                batch_no, dataset_type, elapsed_time,
                cost, model_name,
                total_cost, total_time))
        except Exception as e:

            cost = 0
            print("[%2d/%-2d] NLFS FAILURE |"
                " batch: %2d |"
                " dataset: %7s |"
                " time: %9s |"
                " cost: %6.2f |"
                " model: %-17s |"
                " total_cost: %6.2f |"
                " total_time: %9s |"
                " exception: %s |" % (
                total_completed, total_count,
                batch_no, dataset_type, elapsed_time,
                cost, model_name,
                total_cost, total_time, str(e)))

    return total_cost


def _run_auto_verify(args, batch_no, dataset_type, models, prog_start_time):

    print("******* VERIFY | dataset: %7s | batch: %2d | total_models: %2d | ******* "
          % (dataset_type, batch_no, len(models)))

    temp_llm_verify_name = args.llm_verify
    temp_dataset_type = dataset_type
    if args.llm_verify is None:
        verify_func = get_verify_func(dataset_type)
    else:
        assert args.llm_verify in SUPPORTED_MODELS
        verify_func = llm_verifier.verify

    for i, model_name in enumerate(models):

        dataset_filepath = "%s/batch%d/%s/dataset_nlfs_%s%s.json" % (args.base_dir,
                                                                 batch_no,
                                                                 dataset_type,
                                                                 model_name,
                                                                 "_verified" if args.use_verified else "")
        dataset_filepath = os.path.abspath(os.path.expanduser(dataset_filepath))
        dataset_directory = pathlib.Path(dataset_filepath).parent

        stdout_filepath ="%s/verify.%s.log" % (dataset_directory, model_name)

        start_time = time.time()
        try:
            _, _ = alg.verify(dataset_filepath,
                              verify_func,
                              stdout_filepath,llm_verify_model=temp_llm_verify_name,dataset_type=temp_dataset_type)
            print("[%2d/%-2d] VERIFY SUCCESS |"
                " batch: %2d |"
                " dataset: %7s |"
                " time: %9s |"
                " model: %-17s |"
                " total_time: %9s |" % (
                i + 1, len(models),
                batch_no, dataset_type,
                get_elapsed_timedelta(start_time), model_name,
                get_elapsed_timedelta(prog_start_time)))
        except Exception as e:

            print("[%2d/%-2d] VERIFY FAILURE |"
                " batch: %2d |"
                " dataset: %7s |"
                " time: %9s |"
                " model: %-17s |"
                " total_time: %9s |"
                " exception: %s |" % (
                i + 1, len(models),
                batch_no, dataset_type,
                get_elapsed_timedelta(start_time), model_name,
                get_elapsed_timedelta(prog_start_time),
                str(e)))


def run_auto(args, prog_start_time):

    total_cost = 0
    if not args.skip_nlfs:
        for batch_no in range(args.start_batch, args.total_batches + args.start_batch):
            for dataset_type in args.datasets:
                total_cost = _run_auto_nlfs(args, batch_no, dataset_type,
                                            args.models,
                                            total_cost, prog_start_time)

    if not args.skip_verify:
        for batch_no in range(args.start_batch, args.total_batches + args.start_batch):
            for dataset_type in args.datasets:
                _run_auto_verify(args, batch_no, dataset_type, args.models,
                                 prog_start_time)


def run_single(args):

    t_prompt, i_prompt = get_prompts(args.dataset_type)

    if args.llm_verify is None:
        verify_func = get_verify_func(args.dataset_type)
    else:
        assert args.llm_verify in SUPPORTED_MODELS
        verify_func = llm_verifier.verify

    use_examples = args.prompting in ['few-shot']
    dataset_filepath = args.dataset_filepath
    if not args.skip_nlfs:

        _, dataset_filepath, _ = alg.perform_nlfs(args.dataset_filepath,
                                               args.model_name,
                                               t_prompt,
                                               i_prompt,
                                               args.filter_field,
                                               args.store_conversation_history,
                                               use_examples)

    if not args.skip_verify:

        _, _ = alg.verify(dataset_filepath, verify_func, llm_verify_model=args.llm_verify,
                dataset_type=args.dataset_type)

def verify_auto_args(args):

    assert args.base_dir is not None
    assert args.total_batches > 0
    assert args.dataset_filepath is None
    assert args.model_name is None
    assert args.dataset_type is None
    assert args.filter_field is None
    assert all([dataset in SUPPORTED_DATASETS for dataset in args.datasets])
    assert all([model in SUPPORTED_MODELS for model in args.models])
    assert args.prompting in SUPPORTED_PROMPTING

def verify_single_args(args):

    assert args.base_dir is None
    assert args.dataset_filepath is not None
    assert args.model_name is not None
    assert args.dataset_type is not None
    assert args.dataset_type in SUPPORTED_DATASETS
    assert args.model_name in SUPPORTED_MODELS
    assert args.prompting in SUPPORTED_PROMPTING

if __name__ == "__main__":

    try:
        pythonhashseed = int(os.getenv("PYTHONHASHSEED"))
    except Exception:

        print("PYTHONHASHSEED not set or is not a valid integer")
        sys.exit(1)


    parser = argparse.ArgumentParser()

    parser.add_argument("--auto", default=False, action="store_true")
    parser.add_argument("--total-batches", type=int, default=1)
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--max-nlfs-workers", type=int, default=None)
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=SUPPORTED_DATASETS)
    parser.add_argument("--models", type=str, nargs="+",
                        default=SUPPORTED_MODELS)

    parser.add_argument("--dataset-filepath", type=str, default=None)

    parser.add_argument("--model-name", type=str, default=None,
                        choices=SUPPORTED_MODELS)

    parser.add_argument("--dataset-type", type=str, default=None,
                        choices=SUPPORTED_DATASETS)

    parser.add_argument("--skip-nlfs", default=False, action="store_true")
    parser.add_argument("--filter-field", type=str, default=None)
    parser.add_argument("--store-conversation-history", default=False,
                        action="store_true")

    parser.add_argument("--skip-verify", default=False, action="store_true")

    parser.add_argument("--prompting", type=str, default="zero-shot",choices=SUPPORTED_PROMPTING)
    parser.add_argument("--llm-verify", type=str, default=None)
    parser.add_argument("--use-verified", default=False, action="store_true")

    args = parser.parse_args()
    prog_start_time = time.time()
    if args.auto:
        args.store_conversation_history = True
        verify_auto_args(args)
        run_auto(args, prog_start_time)
    else:
        verify_single_args(args)
        run_single(args)

    print("Total time: %s" % (get_elapsed_timedelta(prog_start_time)))



