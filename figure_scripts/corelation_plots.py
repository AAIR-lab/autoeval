import sys
import pathlib
import statistics

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
from scipy.stats import linregress, pearsonr


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
})

def get_data_dict(base_dir):

    try:
        fh = open(base_dir, "rb")
        return pickle.load(fh)
    except Exception:

        return None

PLOT_PROPS = {

    "gpt-4o": {

        "color": "tab:blue",
        "linestyle": "-",
        "label": "GPT-4o",
        "marker": "o",
    },

    "gpt-4o-mini": {

        "color": "tab:red",
        "linestyle": "dashdot",
        "label": "GPT-4o-mini",
        "marker": "s",
    },

    "gpt-3.5-turbo": {

        "color": "tab:green",
        "linestyle": "dashed",
        "label": "ChatGPT",
        "marker": "$\Delta$",
    },

    "mistral": {

        "color": "tab:orange",
        "linestyle": (0, (6, 1, 1, 6)),
        "label": "Mistral",
        "marker": "^",
    },

    "phi-3": {

        "color": "tab:pink",
        "linestyle": (0, (5, 6)),
        "label": "Phi",
        "marker": "h",
    },

    "llama-3-8b": {

        "color": "tab:grey",
        "linestyle": "dotted",
        "label": "LLama3",
        "marker": "D",
    },

    "gemma-2-9b": {

        "color": "lightgray",
        "linestyle": "solid",
        "label": "Gemma2",
        "marker": "H",
    },

    "granite": {

        "color": "lightgray",
        "linestyle": "solid",
        "label": "Granite",
        "marker": "H",
    },

    "llama-3.1-8b": {

        "color": "lightgray",
        "linestyle": "solid",
        "label": "Llama3.1",
        "marker": "H",
    },

    "llama-3.1-8b-prop-logic-lora": {

        "color": "lightgray",
        "linestyle": "solid",
        "label": "Llama3.1(pl)",
        "marker": "H",
    },

    "llama-3.2-1b": {

        "color": "lightgray",
        "linestyle": "solid",
        "label": "Llama3.2",
        "marker": "H",
    },   

    "llama-3-8B-informalization": {

        "color": "lightgray",
        "linestyle": "solid",
        "label": "Llama3(informalization)",
        "marker": "H",
    },

    "llama-3-70b": {

        "color": "lightgray",
        "linestyle": "solid",
        "label": "Llama3(70B)",
        "marker": "H",
    },

    "ministral": {

        "color": "lightgray",
        "linestyle": "solid",
        "label": "Ministral",
        "marker": "H",
    },

    "phi-3.5-mini": {

        "color": "lightgray",
        "linestyle": "solid",
        "label": "Phi3.5-mini",
        "marker": "H",
    },

    "qwen-2.5-1.5b": {

        "color": "lightgray",
        "linestyle": "solid",
        "label": "Qwen2.5(1.5B)",
        "marker": "H",
    },

    "qwen-2.5-14b": {

        "color": "lightgray",
        "linestyle": "solid",
        "label": "Qwen2.5(14B)",
        "marker": "H",
    },
    "yi-1.5-34b": {

        "color": "lightgray",
        "linestyle": "solid",
        "label": "Yi-1.5",
        "marker": "H",
    },   

    "ksat": {
        "title": "$3$--SAT$(12)$",
        "xticks": [0, 10, 20, 30, 40, 50, 60],
        "scatter_y_offset": 0,
        "xlim": [-1, 42],
    },

    "plogic": {
        "title": "PL$(12)$",
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

    "figsize": (18, 3),

    "markersize": 7,
    "title_fontsize": 20,
    "legend_fontsize": 20,
    "xticklabel_fontsize": 18,
    "yticklabel_fontsize": 18,
    "annotation_size": 14,
    "legend_handlelength": 2,
    "legend_bbox_anchor": (0.80, 1.15),
    "others_bbox_anchor": (0.67, 1.25),
    "dataset_annotation_size": 16,

    "precision_bbox_anchor": (0.57, 0.25),
    "recall_bbox_anchor": (0.60, 0.39),
    "precision_legend_fontsize": 16,

    "xlabel_fontsize": 20,
    "ylabel_fontsize": 20,
    "yticks": [0, 0.25, 0.5, 0.75, 1.0],
    "xticks": [0, 0.25, 0.5, 0.75, 1.0],

}

def compute_pearson_coefficient(args, benchmark_data):

    X = []
    Y = []
    for model in args.models:

        if "X" not in benchmark_data[model]:
            continue

        x = np.mean(benchmark_data[model]["X"])
        y = np.mean(benchmark_data[model]["Y"])

        X.append(x)
        Y.append(y)

    correlation, pvalue = pearsonr(X, Y)

    x_bounds = [-0.05, 1.05]
    slope, intercept, r_value, p_value, std_err = linregress(X, Y)
    trend_line = [(slope * current_x) + intercept for current_x in x_bounds]
    print(trend_line)
    return correlation, pvalue, [x_bounds, trend_line]

def plot_benchmark(args, ax, benchmark_data, title="Test", verbose=True):

    print(title,":")
    for model in args.models:
        print("%s & %0.2f & %0.2f \\"%(model,round(np.mean(benchmark_data[model]["X"]),2),round(np.mean(benchmark_data[model]["Y"]),2)))

    print("============")

    ax.set_title(title, fontsize=PLOT_PROPS["title_fontsize"])

    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])

    if "HumanEval" in title:
        print(benchmark_data.keys())
        pass

    all_x = []
    all_y = []
    for model in args.models:

        if "X" not in benchmark_data[model]:
            continue

        handle = ax.plot(np.mean(benchmark_data[model]["X"]), 
            np.mean(benchmark_data[model]["Y"]),
            color=PLOT_PROPS[model]["color"],
            linestyle="",
            marker=PLOT_PROPS[model]["marker"],
            markersize=PLOT_PROPS["markersize"],
            label= PLOT_PROPS[model]["label"] if PLOT_PROPS[model]["color"] != "lightgray" else "_nolegend_")
        PLOT_PROPS[model]["handle"] = handle
        
        head_width = .04
        X = benchmark_data[model]["X"]
        Y = benchmark_data[model]["Y"]
        all_x.append((model, np.mean(benchmark_data[model]["X"])))
        all_y.append((model, np.mean(benchmark_data[model]["Y"])))
        # if "FOLIO ($\math" in title:

        #     for k in range(len(X) - 1):
        #         middle_x = (X[k] + X[k + 1]) / 2.0
        #         middle_y = (Y[k] + Y[k + 1]) / 2.0
        #         slope = ((Y[k] - Y[k + 1])+0.00000001) / ((X[k] - X[k + 1])+0.00000001)

        #         ax.arrow(middle_x, middle_y, 
        #                  -0.0000000001, 
        #                  -(slope * 0.0000000001), 
        #                  shape='full', 
        #                  overhang=0.2,
        #                  lw=0, length_includes_head=True,
        #                 head_width=head_width, 
        #                 color=PLOT_PROPS[model]["color"],
        #                 label="_nolegend_")

    
    print(all_x)
    print(all_y)
    pearson = benchmark_data["pearson_correlation"]
    if "p_value" in benchmark_data:
        p_value = benchmark_data["p_value"]
    else:
        p_value = -1.0
    trend_line = benchmark_data["trend_line"]

    pearson, p_value, trend_line = compute_pearson_coefficient(args, benchmark_data)

    # Determine the correct markers for the intercepts on the trendline.
    slope, intercept = np.polyfit(trend_line[0], 
        trend_line[1], 
        1)
    X = [-0.1]
    Y = [intercept]

    x_at_ymax = (1.1 - intercept) / slope
    y_at_xmax = slope * 1.1 + intercept
    
    if x_at_ymax < 1.1:
        X.append(x_at_ymax)
        Y.append(1.1)
    else:
        X.append(1.1)
        Y.append(y_at_xmax)

    trend_line = ax.plot(X,
        Y,
        linestyle=(0, (1, 4)),
        marker="x",
        clip_on=False,
        markersize=PLOT_PROPS["markersize"],
        zorder=10,
        color="tab:gray")
    PLOT_PROPS["trend_line"] = trend_line

    ax.annotate("$\\rho = %.2f$\n$p=%.2f$" % (pearson, p_value), 
        xy=(-0.03, 0.9),
        fontsize=PLOT_PROPS["annotation_size"])

    total_greater_equal = 0.0
    correct = 0.0
    for i, (current_x, current_y) in enumerate(zip(all_x,all_y)):

        m1 = current_x[0]
        assert current_y[0] == m1
        x1 = current_x[1]
        y1 = current_y[1]

        for j, (second_x, second_y) in enumerate(zip(all_x,all_y)):
            if i == j:
                continue

            m2 = second_x[0]
            assert second_y[0] == m2
            x2 = second_x[1]
            y2 = second_y[1]
            if x1 >= x2:
                total_greater_equal += 1
                if y1 >= y2:
                    correct += 1
                elif verbose:
                    print("Failed for m1:%s (x1:%0.5f, x2:%0.2f) and m2:%s (y1:%0.2f, y2:%0.2f)" % (m1, x1, x2, m2, y1, y2))

    print(len(all_x),len(all_y),total_greater_equal, correct, correct/total_greater_equal)


def process_human_eval(args, human_eval, dataset, limit=float("inf")):

    human_eval_scores = {
        "gpt-3.5-turbo": .743,
        "gpt-4o": .918,
        "gpt-4o-mini": .883,
        "llama-3-8b": .628,
        "mistral": .445,
        "phi-3": .75,
        "gemma-2-9b": .683,
        "granite": .622,
        "llama-3.1-8b": .634,
        "llama-3.2-1b": .346,
        "llama-3-70b": .799,
        "ministral": .768,
        "phi-3.5-mini": .713,
        "qwen-2.5-1.5b": .567,
        "qwen-2.5-14b": .805,
        "yi-1.5-34b": .726
    }

    for model in args.models:

        if model not in human_eval_scores:
            continue

        data = [(k, v) 
                for k, v in human_eval[dataset][model].items() 
                if k < limit]
        data = sorted(data)

        Y = np.mean([x[1] for x in data])

        human_eval[dataset][model]["X"] = [Y]
        human_eval[dataset][model]["Y"] = [human_eval_scores[model]]

    X = [human_eval[dataset][model]["X"][0] for model in args.models if "X" in human_eval[dataset][model]]
    Y = [human_eval_scores[model] for model in args.models if model in human_eval_scores]

    x_bounds = [-0.05, 0.75]
    from scipy.stats import linregress, pearsonr
    slope, intercept, r_value, p_value, std_err = linregress(X, Y)
    line = [(slope * current_x) + intercept for current_x in x_bounds]


    correlation, pvalue = pearsonr(X, Y)
    human_eval[dataset]["trend_line"] = (x_bounds, line)
    human_eval[dataset]["pearson_correlation"] = correlation
    human_eval[dataset]["p_value"] = pvalue


if __name__ == "__main__":

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default="/tmp/rebuttal_data/",
                        required=False)
    parser.add_argument("--save-location", type=str, default="")  
    parser.add_argument("--models", type=str, nargs="+", default=MODELS)
    args = parser.parse_args()

    if args.save_location == "":
        args.save_location = args.base_dir

    logieval_figures = get_data_dict("%s/logicbench_figures.pk" % (args.base_dir))
    folio_figures = get_data_dict("%s/folio_figures.pk" % (args.base_dir))
    folio_premises_figures = get_data_dict("%s/folio_premises_figures.pk" % (args.base_dir))
    human_eval = get_data_dict("%s/humaneval.pk" % (args.base_dir))

    assert logieval_figures is not None
    assert folio_figures is not None
    assert folio_premises_figures is not None

    human_eval_dataset = "regex"
    human_eval_limit = 11
    process_human_eval(args, human_eval, human_eval_dataset,
                       limit=human_eval_limit)

    pass

    fig = plt.Figure(figsize=PLOT_PROPS["figsize"])
    gs = GridSpec(nrows=1, ncols=6, figure=fig,
                  hspace=0.07, wspace=0.05)

    axs = []
    for i in range(6):
        ax = fig.add_subplot(gs[0, i])
        axs.append(ax)

    # for model in args.models:
    #     X = np.mean(folio_premises_figures[5][model]["X"])
    #     folio_figures[0][model]["X"] = X
    #     folio_figures[1][model]["X"] = X


    plot_benchmark(args, axs[0], folio_figures[0], title="FOLIO (R; NL)")
    axs[0].annotate("$S_{cal}($FOL$(8,12)$--E$,6)$", 
        color="tab:brown",
        fontweight="bold",
        xy=(.03, -0.06),
        fontsize=PLOT_PROPS["dataset_annotation_size"])

    plot_benchmark(args, axs[1], folio_figures[1], title="FOLIO (R; FOL)")
    axs[1].annotate("$S_{cal}($FOL$(8,12)$--E$,6)$", 
        color="tab:brown",
        fontweight="bold",
        xy=(.03, -0.06),
        fontsize=PLOT_PROPS["dataset_annotation_size"])

    plot_benchmark(args, axs[2], logieval_figures[4], title="LogiEval (R; PL)")
    axs[2].annotate("$S_{cal}($PL$(12),30)$", 
        color="tab:brown",
        fontweight="bold",
        xy=(.28, -0.06),
        fontsize=PLOT_PROPS["dataset_annotation_size"])

    plot_benchmark(args, axs[3], logieval_figures[5], title="LogiEval (R; FOL)")
    axs[3].annotate("$S_{cal}($FOL$(8,12)$--E$,30)$", 
        color="tab:brown",
        fontweight="bold",
        xy=(-.03, -0.06),
        fontsize=PLOT_PROPS["dataset_annotation_size"])

    plot_benchmark(args, axs[4], folio_premises_figures[5], title="FOLIO ($\mathcal{A}$)")
    axs[4].annotate("$S_{cal}($FOL$(8,12)$--E$,6)$", 
        color="tab:brown",
        fontweight="bold",
        xy=(.03, -0.06),
        fontsize=PLOT_PROPS["dataset_annotation_size"])

    plot_benchmark(args, axs[5], human_eval[human_eval_dataset], title="HumanEval ($\mathcal{A}$)")
    axs[5].annotate("$S_{cal}(RE(2),7)$", 
        color="tab:brown",
        fontweight="bold",
        xy=(.35, -0.06),
        fontsize=PLOT_PROPS["dataset_annotation_size"])

    for i, ax in enumerate(axs):

        ticklabels = ["$0$", "$.25$", "$.5$",
            "$.75$", "$1$"]

        ax.set_xticklabels(ticklabels)
        
        if i != 0:
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels(ticklabels)
            ax.set_ylabel("Benchmark Score",
                fontsize=PLOT_PROPS["ylabel_fontsize"])

        # ax.set_xlim(PLOT_PROPS[dataset_type]["xlim"])
        ax.set_yticks(PLOT_PROPS["yticks"])
        ax.set_xticks(PLOT_PROPS["xticks"])
        ax.tick_params(
            axis="y", labelsize=PLOT_PROPS["yticklabel_fontsize"])
        ax.tick_params(
            axis="x", labelsize=PLOT_PROPS["xticklabel_fontsize"])
        ax.yaxis.grid(True, linestyle="--", alpha=0.45)

    labels = [PLOT_PROPS[model]["label"] for model in args.models if PLOT_PROPS[model]["color"] != "lightgray"]
    fig.legend(labels=labels, frameon=False,
               ncols=6,
               bbox_to_anchor=PLOT_PROPS["legend_bbox_anchor"],
               fontsize=PLOT_PROPS["legend_fontsize"],
               handlelength=PLOT_PROPS["legend_handlelength"],
               columnspacing=1,
               handletextpad=0.2)

    line = [model for model in args.models if PLOT_PROPS[model]["color"] == "lightgray"][0]
    fig.legend(labels=["Trend Line", "Other Models $(n = 10)$"], 
                handles=[PLOT_PROPS["trend_line"][0], PLOT_PROPS[line]["handle"][0]],
               frameon=False,
               ncols=2,
               bbox_to_anchor=PLOT_PROPS["others_bbox_anchor"],
               fontsize=PLOT_PROPS["legend_fontsize"],
               handlelength=PLOT_PROPS["legend_handlelength"],
               columnspacing=1,
               handletextpad=0.2)

    fig.text(0.41, -0.1, "Calibrated $\\forall$uto$\exists$$\lor\!\land$L Score",
             fontsize=PLOT_PROPS["xlabel_fontsize"])

    #fig.savefig("%s/corelation.png" % (args.base_dir), bbox_inches="tight")
    fig.savefig("%s/figure_4.pdf" % (args.save_location), bbox_inches="tight")
