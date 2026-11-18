import sys
import pathlib

_ROOT = (pathlib.Path(__file__).parent / "../../").as_posix()
sys.path.append(_ROOT)

import argparse
import json
import evaluate
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


def read_dataset(data_dict, dataset_filepath, filter_field=None, equiv_model=None):

    with open(dataset_filepath, "r") as fh:
        dataset = json.load(fh)

    if filter_field is None:
        filter_field = dataset["info"]["filter_field"]

    for idx, datum in enumerate(dataset["data"]):

        filter_value = datum[filter_field]
        error = datum["verification"]["has_error"]
        is_equivalent = datum["verification"]["is_equivalent"]
        if is_equivalent is None:
            is_equivalent = False

        if error is not None \
            and ("timeout" in error.lower() \
                or "seconds" in error.lower()):
            continue

        if equiv_model is None:
            if datum["has_fs_leak"]:
                continue

        compl_list, equiv_list, verify_list = data_dict.setdefault(
            filter_value, ([], [], []))

        compl_list.append(int(error is None))
        equiv_list.append(int(is_equivalent))

        try:
            verify_list.append(int(datum["llm_verification"][equiv_model]["is_equivalent"]))
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

def parse_equiv_data(bdir):
    data_dict = {}
    for dataset_type in ['fol','fol_human','ksat','plogic','regex']:
        data_dict[dataset_type] = {}
        for model_name in EQUIVMODELS:
            data_dict[dataset_type][model_name] = {}
            for run_no in range(10):
                # print("%s/equiv_results/%s_equiv_results/run%i/%s/dataset_nlfs_gpt-4o_verfied.json"%(bdir,model_name,run_no,dataset_type))
                read_dataset(data_dict[dataset_type][model_name],"%s/compiled_results/run%i/%s/dataset_nlfs_gpt-4o_verfied.json"%(bdir,run_no,dataset_type),equiv_model=model_name)
    return data_dict

def plot_precision_recall(ax, twinx, data, plot_props, scatter_y_offset=0):


    x = []
    y_precision = []
    y_specificity = []
    for x_value in sorted(data.keys()):

        tp = 0
        tn = 0
        fp = 0
        fn = 0

        compl_list, verifier_equiv, llm_equiv = data[x_value]
        assert len(llm_equiv) > 0

        for compl, gt, v in zip(compl_list, verifier_equiv, llm_equiv):

            if not compl or v is None:
                continue

            if gt:
                if v:
                    tp += 1
                else:
                    fn += 1
            else:
                if not v:
                    tn += 1
                else:
                    fp += 1

        if (tp + fp) == 0:
            continue
        else:
            precision = tp / (tp + fp)

        if (tn + fp) == 0:
            continue
        else:
            specificity = tn / (tn + fp)

        x.append(x_value)
        y_precision.append(precision)
        y_specificity.append(specificity)

    ax.plot(x, y_precision, color=plot_props["color"])

    scatter_point = len(x) // 2
    ax.plot([x[scatter_point]], [y_precision[scatter_point]
                                 + scatter_y_offset],
               marker="o",
               markersize=8,
               color=plot_props["color"])
    twinx.plot(x, y_specificity, alpha=0.45, linestyle="dashed")

def plot_precision(ax, data, plot_props, scatter_y_offset=0):


    x = []
    y_precision = []
    for x_value in sorted(data.keys()):

        tp = 0
        tn = 0
        fp = 0
        fn = 0

        compl_list, verifier_equiv, llm_equiv = data[x_value]
        assert len(llm_equiv) > 0

        for compl, gt, v in zip(compl_list, verifier_equiv, llm_equiv):

            if not compl or v is None:
                continue

            if gt:
                if v:
                    tp += 1
                else:
                    fn += 1
            else:
                if not v:
                    tn += 1
                else:get_data_dict

        compl_list, verifier_equiv, llm_equiv = data[x_value]
        assert len(llm_equiv) > 0

        for compl, gt, v in zip(compl_list, verifier_equiv, llm_equiv):

            if not compl or v is None:
                continue

            if gt:
                if v:
                    tp += 1
                else:
                    fn += 1
            else:
                if not v:
                    tn += 1
                else:
                    fp += 1

        if (tn + fp) == 0:
            continue
        else:
            specificity = tn / (tn + fp)

        x.append(x_value)
        y_specificity.append(specificity)

    ax.plot(x, y_specificity, color=plot_props["color"],linestyle=plot_props["linestyle"])

def plot_f1(ax, data, plot_props, scatter_y_offset=0):


    x = []
    y_f1 = []
    for x_value in sorted(data.keys()):

        tp = 0
        tn = 0
        fp = 0
        fn = 0

        compl_list, verifier_equiv, llm_equiv = data[x_value]
        assert len(llm_equiv) > 0

        for compl, gt, v in zip(compl_list, verifier_equiv, llm_equiv):

            if not compl or v is None:
                continue

            if gt:
                if v:
                    tp += 1
                else:
                    fn += 1
            else:
                if not v:
                    tn += 1
                else:
                    fp += 1

        f1 = tp + (0.5 * (fp + fn))
        if f1 == 0:
            continue
        f1 = tp / f1

        x.append(x_value)
        y_f1.append(f1)

    center = int(len(x) // 2)

    line = ax.plot(x, y_f1, label=plot_props["label"], 
        markevery=[center],
        marker=plot_props["marker"],
        linestyle=plot_props["linestyle"],
        color=plot_props["color"])
    plot_props["handle"] = line[0]


def plot(ax, data, idx, plot_props):

    x = []
    y = []
    y_err = []
    for x_value in sorted(data.keys()):

        x.append(x_value)
        y_values = data[x_value][idx]

        mean = np.mean(y_values)
        std = np.std(y_values)

        y.append(mean)
        y_err.append(std)

    x = np.asarray(x)
    y = np.asarray(y)

    center = int(len(x) // 2)

    line = ax.plot(x, y, label=plot_props["label"], 
        markevery=[center],
        marker=plot_props["marker"],
        linestyle=plot_props["linestyle"],
        color=plot_props["color"])
    plot_props["handle"] = line[0]

    # ax.fill_between(x, y - y_err, y + y_err, color=plot_props["color"],
    #                 alpha=0.05)


PLOT_PROPS = {

    "gpt-4o": {

        "color": "tab:blue",
        "marker": "o",
        "linestyle": "-",
        "label": "GPT-4o",
    },

    "gpt-4o-mini": {

        "color": "tab:orange",
        "marker": "s",
        "linestyle": "-",
        "label": "GPT-4o-mini",
    },

    "gpt-3.5-turbo": {

        "color": "tab:green",
        "marker": "D",
        "linestyle": "-",
        "label": "ChatGPT",
    },


    "mistral": {

        "color": "tab:red",
        "marker": "x",
        "linestyle": "-",
        "label": "Mistral-7B-Instruct-v0.2",
    },

    "phi-3": {

        "color": "tab:blue",
        "marker": "o",
        "linestyle": "--",
        "label": "Phi-3-Medium-4k-Instruct",
    },

    "llama-3-8b": {

        "color": "tab:orange",
        "marker": "s",
        "linestyle": "--",
        "label": "LLama-3-8B-Instruct",
    },

    "gemma-2-9b": {

        "color": "tab:green",
        "marker": "D",
        "linestyle": "--",
        "label": "Gemma-2-9B-IT",
    },

    "granite": {

        "color": "tab:red",
        "marker": "x",
        "linestyle": "--",
        "label": "Granite-3.0-8B-Instruct",
    },

    "llama-3.1-8b": {

        "color": "tab:blue",
        "marker": "o",
        "linestyle": "-.",
        "label": "Llama-3.1-8B-Instruct",
    },

    "llama-3.2-1b": {

        "color": "tab:orange",
        "marker": "s",
        "linestyle": "-.",
        "label": "LLama-3.2-1B-Instruct",
    },   

    "llama-3-70b": {

        "color": "tab:green",
        "marker": "D",
        "linestyle": "-.",
        "label": "LLama-3-70B-Instruct",
    },

    "ministral": {

        "color": "tab:red",
        "marker": "x",
        "linestyle": "-.",
        "label": "Ministral-8B-Instruct-2410",
    },

    "phi-3.5-mini": {

        "color": "tab:blue",
        "marker": "o",
        "linestyle": ":",
        "label": "Phi-3.5-Mini-Instruct",
    },

    "qwen-2.5-1.5b": {

        "color": "tab:orange",
        "marker": "s",
        "linestyle": ":",
        "label": "Qwen-2.5-1.5B-Instruct",
    },

    "qwen-2.5-14b": {

        "color": "tab:green",
        "marker": "D",
        "linestyle": ":",
        "label": "Qwen-2.5-14B-Instruct",
    },
    "yi-1.5-34b": {

        "color": "tab:red",
        "marker": "x",
        "linestyle": ":",
        "label": "Yi-1.5-34B",
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

    "figsize": (18, 25),

    "title_fontsize": 20,
    "legend_fontsize": 14,
    "xticklabel_fontsize": 18,
    "yticklabel_fontsize": 18,
    "legend_handlelength": 2.5,
    "legend_bbox_anchor": (0.83, 0.95),
    "other_bbox_anchor": (0.6, 1.02),

    "precision_bbox_anchor": (0.57, 0.25),
    "recall_bbox_anchor": (0.60, 0.39),
    "precision_legend_fontsize": 16,

    "xlabel_fontsize": 20,
    "ylabel_fontsize": 20,
    "yticks": [0, 0.25, 0.5, 0.75, 1.0],
    "yticklabels": ["$0$", "$.25$", "$.5$", "$.75$", "$1$"],
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default="/tmp/rebuttal_data/zero-shot",
                        required=False)
    parser.add_argument("--save-location", type=str, default="")  
    parser.add_argument("--runs", type=int, nargs="+",
                        default=range(10))
    parser.add_argument("--datasets", type=str, nargs="+",
                        default= ["plogic", "fol", "fol_human", "regex"])

    parser.add_argument("--plot-type", type=int, required=True)

    MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-3.5-turbo",
        "mistral",
        "phi-3",
        "llama-3-8b",
        "gemma-2-9b",
        "granite",
        "llama-3.1-8b",
        "llama-3.2-1b",
        "llama-3-70b",
        "ministral",
        "phi-3.5-mini",
        "qwen-2.5-1.5b",
        "qwen-2.5-14b",
        "yi-1.5-34b",
    ]

    EQUIVMODELS = [
        "gemma-2-9b",
        "granite",
        "llama-3.1-8b",
        "llama-3.2-1b",
        "ministral",
        "phi-3.5-mini",
        "qwen-2.5-1.5b",
        "qwen-2.5-14b",
        "gpt-3.5-turbo",
        "gpt-4o",
        "phi-3",
        "mistral",
        "llama-3-70b",
        "yi-1.5-34b",
        "llama-3-8b"   
    ]
    
     # 0 for syntactic compliance, 1 for accuracy, 2 for llm as verifier.
    # PLOT_TYPE = 2

    # if PLOT_TYPE == 2:
    #     MODELS = EQUIVMODELS

    parser.add_argument("--models", type=str, nargs="+",
                        default=MODELS)

    args = parser.parse_args()

    if args.save_location == "":
        args.save_location = args.base_dir

    data_dict = get_data_dict(args.base_dir)
    assert data_dict is not None

    PLOT_TYPE = args.plot_type
    if PLOT_TYPE == 2 and args.models == MODELS:
        args.models = EQUIVMODELS

    if PLOT_TYPE == 2:
        equiv_models = parse_equiv_data(args.base_dir)

    fig = plt.Figure(figsize=PLOT_PROPS["figsize"])
    gs = GridSpec(nrows=len(MODELS), ncols=len(args.datasets), figure=fig,
                  hspace=0.07, wspace=0.05)

    for i, dataset_type in enumerate(args.datasets):
        for j, model_name in enumerate(args.models):
            
            ax0 = fig.add_subplot(gs[j, i])
            

            if j == 0:
                ax0.set_title(PLOT_PROPS[dataset_type]["title"],
                            fontsize=PLOT_PROPS["title_fontsize"])

            if PLOT_TYPE == 2:
                plot_f1(ax0,equiv_models[dataset_type][model_name],PLOT_PROPS[model_name],scatter_y_offset=PLOT_PROPS[dataset_type]["scatter_y_offset"])
            else:
                plot(ax0, data_dict[dataset_type][model_name], PLOT_TYPE, PLOT_PROPS[model_name])
            ax0.set_ylim([-0.1, 1.1])

            if j != len(args.models) - 1:
                ax0.set_xticklabels([])


            if i != 0:
                ax0.set_yticklabels([])
            else:
                pass
                # ax0.set_yticklabels(PLOT_PROPS["yticklabels"])

                # ax0.set_ylabel("\S A2: Syntactic\nCompliance",
                #             fontsize=PLOT_PROPS["ylabel_fontsize"])

            for ax in [ax0]:

                ax.set_xlim(PLOT_PROPS[dataset_type]["xlim"])
                # ax.set_yticks(PLOT_PROPS["yticks"])
                # ax.set_xticks(PLOT_PROPS[dataset_type]["xticks"])
                ax.tick_params(
                    axis="y", labelsize=PLOT_PROPS["yticklabel_fontsize"])
                ax.tick_params(
                    axis="x", labelsize=PLOT_PROPS["xticklabel_fontsize"])
                # ax.yaxis.grid(True, linestyle="--", alpha=0.45)

            if i == 0:
                ax0.legend(fontsize=PLOT_PROPS["legend_fontsize"],)

    fig.add_artist(lines.Line2D([0.13, 0.70], [0.095, 0.095],
                                color="black"))
    fig.text(0.16, 0.085, "\\# of Operators: $\land, \lor, \\neg$ ($\\neg$ is counted as an operator iff not succeeded by a terminal)",
             fontsize=PLOT_PROPS["xlabel_fontsize"])


    fig.add_artist(lines.Line2D([0.715, 0.9], [0.095, 0.095],
                                color="black"))
    fig.text(0.74, 0.085, "CFG Parse Tree Depth",
             fontsize=PLOT_PROPS["xlabel_fontsize"])

    if PLOT_TYPE == 0:
        fig.text(0.09, 0.45, "Syntactic Compliance",
                rotation=90,
                fontsize=PLOT_PROPS["xlabel_fontsize"])
    elif PLOT_TYPE == 1:
        fig.text(0.09, 0.45, "$\\forall$uto$\exists$$\lor\!\land$L Score",
                rotation=90,
                fontsize=PLOT_PROPS["xlabel_fontsize"])
    else:
        fig.text(0.09, 0.45, "LLMs as Verifiers $F_1$ Score",
                rotation=90,
                fontsize=PLOT_PROPS["xlabel_fontsize"])        
    labels = [PLOT_PROPS[model]["label"] for model in args.models if PLOT_PROPS[model]["color"] != "lightgray"]
    # handles = [PLOT_PROPS[model]["handle"] for model in args.models if PLOT_PROPS[model]["color"] != "lightgray"]

    # gray_line = [PLOT_PROPS[model]["handle"] for model in args.models if PLOT_PROPS[model]["color"] == "lightgray"][0]
    # labels += ["Other LLMs $(n = 10)$"]
    # handles += [gray_line]

    fig.legend(labels=labels,
               frameon=False,
               ncols=4,
               columnspacing=0.75,
               bbox_to_anchor=PLOT_PROPS["legend_bbox_anchor"],
               fontsize=PLOT_PROPS["legend_fontsize"],
               handlelength=PLOT_PROPS["legend_handlelength"])

    if PLOT_TYPE == 0:
        #fig.savefig("%s/individual_models_sc.png" % (args.base_dir), bbox_inches="tight")
        fig.savefig("%s/figure_18.pdf" % (args.save_location), bbox_inches="tight")
    elif PLOT_TYPE == 1:
        #fig.savefig("%s/individual_models_a.png" % (args.base_dir), bbox_inches="tight")
        fig.savefig("%s/figure_19.pdf" % (args.save_location), bbox_inches="tight")
    else:
        fig.savefig("%s/figure_20.pdf" % (args.save_location), bbox_inches="tight")
    pass
