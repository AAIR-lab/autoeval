import argparse
import os
import shutil
import json

def get_path(base_dir, run_no, dataset_type):

    return "%s/run%d/%s/" % (base_dir, run_no,
                             dataset_type)

def get_dataset_path(base_dir, run_no, dataset_type):

    return "%s/dataset.json" % (
        get_path(base_dir, run_no, dataset_type))

def get_dataset(base_dir, run_no, dataset_type):

    with open(get_dataset_path(base_dir, run_no, dataset_type), "r") as fh:
        return json.load(fh)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default=None,
                        required=True)
    parser.add_argument("--total-runs", type=int, default=10)
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument("--end-batch", type=int, default=9)
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["ksat", "plogic", "fol", "fol_human", "regex"])
    parser.add_argument("--output-dir", type=str, default=None,
                        required=True)

    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    else:
        os.makedirs(args.output_dir)

    datasets = {}
    for dataset_type in args.datasets:
        datasets[dataset_type] = get_dataset(args.base_dir,
                                       args.start_batch,
                                       dataset_type)
        for run_no in range(args.start_batch + 1, args.start_batch + args.end_batch + 1):

            batch_data = get_dataset(args.base_dir,
                                       run_no,
                                       dataset_type)

            for datum in batch_data["data"]:

                datum["idx"] = len(datasets[dataset_type]["data"])
                datasets[dataset_type]["data"].append(datum)

    for dataset_type in args.datasets:
        for run_no in range(args.total_runs):

            path = get_path(args.output_dir, run_no, dataset_type)
            os.makedirs(path, exist_ok=True)

            dataset_path = get_dataset_path(args.output_dir, run_no,
                                            dataset_type)
            with open(dataset_path, "w") as fh:
                json.dump(datasets[dataset_type], fh, indent=4,
                          ensure_ascii=False)