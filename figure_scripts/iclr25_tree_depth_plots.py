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
                             dataset_filepath,filter_field="depth")
                progress_bar.update(1)
    progress_bar.close()

    return data_dict

def parse_equiv_data(bdir, filter_field=None):
    data_dict = {}
    for dataset_type in ['fol','fol_human','ksat','plogic','regex']:
        data_dict[dataset_type] = {}
        for model_name in EQUIVMODELS:
            data_dict[dataset_type][model_name] = {}
            for run_no in range(10):
                # print("%s/equiv_results/%s_equiv_results/run%i/%s/dataset_nlfs_gpt-4o_verfied.json"%(bdir,model_name,run_no,dataset_type))
                read_dataset(data_dict[dataset_type][model_name],"%s/compiled_results/run%i/%s/dataset_nlfs_gpt-4o_verfied.json"%(bdir,run_no,dataset_type),equiv_model=model_name, filter_field=filter_field)
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
                else:
                    fp += 1

        if (tp + fp) == 0:
            continue
        else:
            precision = tp / (tp + fp)

        x.append(x_value)
        y_precision.append(precision)

    ax.plot(x, y_precision, color=plot_props["color"],linestyle=plot_props["linestyle"])

def plot_specificity(ax, data, plot_props, scatter_y_offset=0):


    x = []
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

    ax.plot(x, y_f1, color=plot_props["color"],linestyle=plot_props["linestyle"])


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

    line = ax.plot(x, y, color=plot_props["color"],
            linestyle=plot_props["linestyle"],label="_nolegend_" if plot_props["color"] == "lightgray" else plot_props["label"])
    plot_props["handle"] = line[0]

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
        "label": "Phi3",
    },

    "llama-3-8b": {

        "color": "black",
        "linestyle": "dotted",
        "label": "LLama3",
    },

    "gemma-2-9b": {

        "color": "lightgray",
        "linestyle": (0, (1, 1)),
        "label": "Gemma2",
    },

    "granite": {

        "color": "lightgray",
        "linestyle": (0, (1, 1)),
        "label": "Granite",
    },

    "llama-3.1-8b": {

        "color": "lightgray",
        "linestyle": (0, (1, 1)),
        "label": "Llama3.1",
    },

    "llama-3.1-8b-prop-logic-lora": {

        "color": "lightgray",
        "linestyle": (0, (1, 1)),
        "label": "Llama3.1(pl)",
    },

    "llama-3.2-1b": {

        "color": "lightgray",
        "linestyle": (0, (1, 1)),
        "label": "Llama3.2",
    },   

    "llama-3-8B-informalization": {

        "color": "lightgray",
        "linestyle": (0, (1, 1)),
        "label": "Llama3(informalization)",
    },

    "llama-3-70b": {

        "color": "lightgray",
        "linestyle": (0, (1, 1)),
        "label": "Llama3(70B)",
    },

    "ministral": {

        "color": "lightgray",
        "linestyle": (0, (1, 1)),
        "label": "Ministral",
    },

    "phi-3.5-mini": {

        "color": "lightgray",
        "linestyle": (0, (1, 1)),
        "label": "Phi3.5-mini",
    },

    "qwen-2.5-1.5b": {

        "color": "lightgray",
        "linestyle": (0, (1, 1)),
        "label": "Qwen2.5(1.5B)",
    },

    "qwen-2.5-14b": {

        "color": "lightgray",
        "linestyle": (0, (1, 1)),
        "label": "Qwen2.5(14B)",
    },
    "yi-1.5-34b": {

        "color": "lightgray",
        "linestyle": (0, (1, 1)),
        "label": "Yi-1.5",
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
    "legend_bbox_anchor": (0.9, 0.98),

    "precision_bbox_anchor": (0.57, 0.25),
    "recall_bbox_anchor": (0.60, 0.39),
    "precision_legend_fontsize": 16,

    "xlabel_fontsize": 20,
    "ylabel_fontsize": 20,
    "yticks": [0, 0.25, 0.5, 0.75, 1.0]
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default="/tmp/results/",
                        required=False)
    parser.add_argument("--save-location", type=str, default="")  
    parser.add_argument("--runs", type=int, nargs="+",
                        default=range(10))
    parser.add_argument("--datasets", type=str, nargs="+",
                        default= ["plogic", "fol", "fol_human", "regex"])
    
    MODELS = [
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
        "gpt-3.5-turbo",
        "gpt-4o",
        "gpt-4o-mini",
        "phi-3",
        "mistral",
        "llama-3-8b"   
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

        data_dict = parse_data(args)
        save_data_dict(args.base_dir, data_dict)

    fig = plt.Figure(figsize=PLOT_PROPS["figsize"])
    gs = GridSpec(nrows=4, ncols=len(args.datasets), figure=fig,
                  hspace=0.07, wspace=0.05)

    equiv_models = parse_equiv_data(args.base_dir,"depth")
    for i, dataset_type in enumerate(args.datasets):

        ax0 = fig.add_subplot(gs[0, i])
        ax1 = fig.add_subplot(gs[1, i])
        ax2 = fig.add_subplot(gs[2, i])
        
        # plot_f1(ax2,
        #       data_dict[dataset_type]["gpt-4o"],
        #       PLOT_PROPS["gpt-4o"],
        #       scatter_y_offset=PLOT_PROPS[dataset_type]["scatter_y_offset"])
        
        for model_name in EQUIVMODELS:
            plot_f1(ax2,equiv_models[dataset_type][model_name],PLOT_PROPS[model_name],scatter_y_offset=PLOT_PROPS[dataset_type]["scatter_y_offset"])

        ax0.set_title(PLOT_PROPS[dataset_type]["title"],
                      fontsize=PLOT_PROPS["title_fontsize"])
        for model_name in args.models:

            plot(ax0, data_dict[dataset_type][model_name], 0,
                 PLOT_PROPS[model_name])
            ax0.set_ylim([-0.1, 1.1])
            ax0.set_xticklabels([])

            plot(ax1, data_dict[dataset_type][model_name], 1,
                 PLOT_PROPS[model_name])
            ax1.set_xticklabels([])
            ax1.set_ylim([-0.1, 1.1])
            ax2.set_ylim([-0.1, 1.1])

            if i != 0:
                ax0.set_yticklabels([])
                ax1.set_yticklabels([])
                ax2.set_yticklabels([])
            else:


                ax0.set_ylabel("Syntactic\nCompliance",
                               fontsize=PLOT_PROPS["ylabel_fontsize"])
                ax1.set_ylabel("$\\forall$uto$\exists$$\lor\!\land$L\nScore",
                               fontsize=PLOT_PROPS["ylabel_fontsize"])
                ax2.set_ylabel("LLMs as Verifiers\n$F_1$ Score",
                               fontsize=PLOT_PROPS["ylabel_fontsize"])

            for ax in [ax0, ax1, ax2]:

                ax.set_xlim(PLOT_PROPS[dataset_type]["xlim"])
                ax.set_yticks(PLOT_PROPS["yticks"])
                ax.set_xticks(PLOT_PROPS[dataset_type]["xticks"])
                ax.tick_params(
                    axis="y", labelsize=PLOT_PROPS["yticklabel_fontsize"])
                ax.tick_params(
                    axis="x", labelsize=PLOT_PROPS["xticklabel_fontsize"])
                ax.yaxis.grid(True, linestyle="--", alpha=0.45)


    fig.add_artist(lines.Line2D([0.12, 0.9], [0.26, 0.26],
                                color="black"))
    fig.text(0.4, 0.225, "CFG Parse Tree Depth",
             fontsize=PLOT_PROPS["xlabel_fontsize"])

    labels = [PLOT_PROPS[model]["label"] for model in args.models if PLOT_PROPS[model]["color"] != "lightgray"]
    handles = [PLOT_PROPS[model]["handle"] for model in args.models if PLOT_PROPS[model]["color"] != "lightgray"]

    gray_line = [PLOT_PROPS[model]["handle"] for model in args.models if PLOT_PROPS[model]["color"] == "lightgray"][0]
    labels += ["Other LLMs $(n = 10)$"]
    handles += [gray_line]

    fig.legend(labels=labels, handles=handles,
               frameon=False,
               ncols=67,
               columnspacing=0.75,
               bbox_to_anchor=PLOT_PROPS["legend_bbox_anchor"],
               fontsize=PLOT_PROPS["legend_fontsize"],
               handlelength=PLOT_PROPS["legend_handlelength"])

    #fig.savefig("%s/plot.png" % (args.base_dir), bbox_inches="tight")
    fig.savefig("%s/figure_11.pdf" % (args.save_location), bbox_inches="tight")
    pass
