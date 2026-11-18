import sys
import pathlib

_ROOT = (pathlib.Path(__file__).parent / "../../../").as_posix()
sys.path.append(_ROOT)

import argparse
import json
import evaluation
import tqdm

import os
import pickle
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.lines as lines

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
})


def bin(data, width=1):

    # https://stackoverflow.com/questions/21921178/binning-a-numpy-array
    results = data[:(data.size // width) * width].reshape(-1, width).mean(axis=1)
    return results

def read_dataset(data_dict, dataset_filepath, filter_field=None):

    with open(dataset_filepath, "r") as fh:
        dataset = json.load(fh)

    if filter_field is None:
        filter_field = dataset["info"]["filter_field"]

    for idx, datum in enumerate(dataset["data"]):

        if "has_fs_leak" in datum:
            if datum["has_fs_leak"]:
                continue

        filter_value = datum[filter_field]
        error = datum["verification"]["has_error"]
        is_equivalent = datum["verification"]["is_equivalent"]
        if is_equivalent is None:
            is_equivalent = False

        if error is not None \
            and ("timeout" in error.lower() \
                or "seconds" in error.lower()):
            continue

        compl_list, equiv_list, verify_list = data_dict.setdefault(
            filter_value, ([], [], []))

        compl_list.append(int(error is None))
        equiv_list.append(int(is_equivalent))

        try:
            verify_list.append(int(datum["llm_verification"]["is_equivalent"]))
            pass
        except Exception:
            verify_list.append(None)
            pass

def get_dataset_filepath(base_dir, run_no, dataset_type, model_name):

    return "%s/batch%d/%s/%s.verified.json" % (
        base_dir, run_no, dataset_type, model_name)

def get_model_data_dict(data_dict, model_name):

    return data_dict.setdefault(model_name, {})

def get_data_dict_dir(base_dir):

    return "%s/parsed_data.pkl" % (base_dir)

def get_data_dict(base_dir):

    try:
        fh = open(get_data_dict_dir(base_dir), "rb")
        return pickle.load(fh)
    except Exception:

        return None

def save_data_dict(base_dir, data_dict):

    fh = open(get_data_dict_dir(base_dir), "wb")
    pickle.dump(data_dict, fh)

def clean_data_dict(base_dir):

    try:
        os.remove(get_data_dict_dir(base_dir))
    except Exception:

        pass

def parse_data(args):

    data_dict = {}

    progress_bar = tqdm.tqdm(
        total=len(args.models) * len(args.datasets) * len(args.runs),
        unit=" experiments",
        leave=False)

    for dataset_type in args.datasets:

        data_dict[dataset_type] = {}
        for model_name in args.models:

            data_dict[dataset_type][model_name] = {}
            for run_no in args.runs:

                dataset_filepath = get_dataset_filepath(
                    args.base_dir,
                    run_no,
                    dataset_type,
                    model_name)
                read_dataset(data_dict[dataset_type][model_name],
                             dataset_filepath)
                progress_bar.update(1)
    progress_bar.close()

    return data_dict

def parse_few_shot_data(args):

    data_dict = {}

    progress_bar = tqdm.tqdm(
        total=len(args.models) * len(args.datasets) * len(args.runs),
        unit=" experiments",
        leave=False)

    for dataset_type in args.datasets:

        data_dict[dataset_type] = {}
        for model_name in args.models:

            data_dict[dataset_type][model_name] = {}
            for run_no in args.runs:

                dataset_filepath = get_dataset_filepath(
                    args.few_shot_dir,
                    run_no,
                    dataset_type,
                    model_name)
                read_dataset(data_dict[dataset_type][model_name],
                             dataset_filepath)
                progress_bar.update(1)
    progress_bar.close()

    return data_dict


def plot(ax, data, few_shot_data, idx, plot_props):

    x = []
    y = []
    fy = []
    y_err = []
    for x_value in sorted(data.keys()):

        y_values = data[x_value][idx]
        few_shot_y_values = few_shot_data[x_value][idx]

        mean = np.mean(y_values)
        few_shot_mean = np.mean(few_shot_y_values)
        std = np.std(y_values)

        if mean > 0.0:
            x.append(x_value)
            y.append(mean)
            fy.append(few_shot_mean)
            y_err.append(std)

    x = bin(np.asarray(x))
    y = bin(np.asarray(y))
    fy = bin(np.asarray(fy))

    non_zero_bins = np.argwhere(y>0)
    x = x[non_zero_bins]

    y = fy[non_zero_bins] - y[non_zero_bins]

    ax.plot(x, y, color=plot_props["color"],
            linestyle=plot_props["linestyle"])

    # ax.fill_between(x, y - y_err, y + y_err, color=plot_props["color"],
    #                 alpha=0.05)


PLOT_PROPS = {

    "gpt-4o": {

        "color": "tab:blue",
        "linestyle": "-",
        "label": "GPT-4o",
    },

    "gpt-4o-mini": {

        "color": "tab:red",
        "linestyle": "dashdot",
        "label": "GPT-4o-mini",
    },

    "gpt-3.5-turbo": {

        "color": "tab:green",
        "linestyle": "dashed",
        "label": "ChatGPT",
    },

    "claude": {

        "color": "tab:red",
        "linestyle": "dashdot",
        "label": "Sonnet",
    },

    "mistral": {

        "color": "tab:orange",
        "linestyle": (0, (6, 1, 1, 6)),
        "label": "Mistral",
    },

    "phi-3": {

        "color": "tab:pink",
        "linestyle": (0, (5, 6)),
        "label": "Phi",
    },

    "llama-3-8b": {

        "color": "tab:grey",
        "linestyle": "dotted",
        "label": "LLama3",
    },

    "ksat": {
        "title": "$3$--SAT$(12)$",
        "xticks": [0, 10, 20, 30, 40, 50, 60],
        "scatter_y_offset": 0,
        "xlim": [-1, 42],
    },

    "plogic": {
        "title": "Propositional Logic$(12)$",
        "xticks": [0, 10, 20, 30, 40],
        "scatter_y_offset": 0,
        "xlim": [-1, 42],
    },

    "fol": {
        "title": "FOL$(8, 12)-$S",
        "xticks": [0, 10, 20, 30, 40],
        "scatter_y_offset": 0.03,
        "xlim": [-1, 42],
    },

    "fol_human": {
        "title": "FOL$(8, 12)-$E",
        "xticks": [0, 10, 20, 30, 40],
        "scatter_y_offset": 0,
        "xlim": [-1, 42],
    },

    "regex": {
        "title": "Regular Expression$(2)$",
        "xticks": [0, 10, 20, 30, 40],
        "scatter_y_offset": 0,
        "xlim": [-1, 42],
    },

    "figsize": (18, 9),

    "title_fontsize": 20,
    "legend_fontsize": 20,
    "xticklabel_fontsize": 18,
    "yticklabel_fontsize": 18,
    "legend_handlelength": 2,
    "legend_bbox_anchor": (0.88, 0.98),

    "precision_bbox_anchor": (0.57, 0.25),
    "recall_bbox_anchor": (0.60, 0.39),
    "precision_legend_fontsize": 16,

    "xlabel_fontsize": 20,
    "ylabel_fontsize": 20,
    "yticks": [-0.25, 0, 0.25, 0.5, 0.75, 1.0]
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default="/tmp/results/",
                        required=False)
    parser.add_argument("--few-shot-dir", type=str, default="/tmp/results/",
                        required=True)
    parser.add_argument("--save-location", type=str, default="")  
    parser.add_argument("--runs", type=int, nargs="+",
                        default=range(10))
    parser.add_argument("--datasets", type=str, nargs="+",
                        default= ["plogic", "fol", "fol_human", "regex"])
    
    MODELS = ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini", "phi-3", 
        "mistral", "llama-3-8b",]
    parser.add_argument("--models", type=str, nargs="+",
                        default=MODELS)
    parser.add_argument("--clean", default=False, action="store_true")


    args = parser.parse_args()

    if args.save_location == "":
        args.save_location = args.base_dir

    if args.clean:
        clean_data_dict(args.base_dir)

    data_dict = get_data_dict(args.base_dir)
    if data_dict is None:

        data_dict = (parse_data(args),parse_few_shot_data(args))
        save_data_dict(args.base_dir, data_dict)

    fig = plt.Figure(figsize=PLOT_PROPS["figsize"])
    gs = GridSpec(nrows=3, ncols=len(args.datasets), figure=fig,
                  hspace=0.07, wspace=0.05)

    for i, dataset_type in enumerate(args.datasets):

        ax0 = fig.add_subplot(gs[0, i])
        ax1 = fig.add_subplot(gs[1, i])
    

        ax0.set_title(PLOT_PROPS[dataset_type]["title"],
                      fontsize=PLOT_PROPS["title_fontsize"])
        for model_name in args.models:

            plot(ax0, data_dict[0][dataset_type][model_name],data_dict[1][dataset_type][model_name], 0,
                 PLOT_PROPS[model_name])
            ax0.set_ylim([-0.26, 1.1])
            ax0.set_xticklabels([])

            plot(ax1, data_dict[0][dataset_type][model_name],data_dict[1][dataset_type][model_name], 1,
                 PLOT_PROPS[model_name])
            ax1.set_ylim([-0.26, 1.1])

            if i != 0:
                ax0.set_yticklabels([])
                ax1.set_yticklabels([])
            else:


                ax0.set_ylabel("Syntactic\nCompliance",
                               fontsize=PLOT_PROPS["ylabel_fontsize"])
                ax1.set_ylabel("$\\forall$uto$\exists$$\lor\!\land$L\nScore",
                               fontsize=PLOT_PROPS["ylabel_fontsize"])

            for ax in [ax0, ax1]:

                ax.set_xlim(PLOT_PROPS[dataset_type]["xlim"])
                ax.set_yticks(PLOT_PROPS["yticks"])
                ax.set_xticks(PLOT_PROPS[dataset_type]["xticks"])
                ax.tick_params(
                    axis="y", labelsize=PLOT_PROPS["yticklabel_fontsize"])
                ax.tick_params(
                    axis="x", labelsize=PLOT_PROPS["xticklabel_fontsize"])
                ax.yaxis.grid(True, linestyle="--", alpha=0.45)

    fig.add_artist(lines.Line2D([0.13, 0.70], [0.32, 0.32],
                                color="black"))
    fig.text(0.16, 0.285, "\\# of Operators: $\land, \lor, \\neg$ ($\\neg$ is counted as an operator iff not succeeded by a terminal)",
             fontsize=PLOT_PROPS["xlabel_fontsize"])


    fig.add_artist(lines.Line2D([0.715, 0.9], [0.32, 0.32],
                                color="black"))
    fig.text(0.74, 0.285, "CFG Parse Tree Depth",
             fontsize=PLOT_PROPS["xlabel_fontsize"])

    labels = [PLOT_PROPS[model]["label"] for model in args.models]
    fig.legend(labels=labels, frameon=False,
               ncols=6,
               bbox_to_anchor=PLOT_PROPS["legend_bbox_anchor"],
               fontsize=PLOT_PROPS["legend_fontsize"],
               handlelength=PLOT_PROPS["legend_handlelength"])

    #fig.savefig("%s/plot.png" % (args.base_dir), bbox_inches="tight")
    fig.savefig("%s/figure_16.pdf" % (args.save_location), bbox_inches="tight")
    pass
