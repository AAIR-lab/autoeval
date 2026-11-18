import sys
import pathlib

_ROOT = (pathlib.Path(__file__).parent / "../../").as_posix()
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
        "label": "GPT-4",
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

    "markersize": 6,
    "title_fontsize": 20,
    "legend_fontsize": 20,
    "xticklabel_fontsize": 18,
    "yticklabel_fontsize": 18,
    "annotation_size": 14,
    "legend_handlelength": 2,
    "legend_bbox_anchor": (0.875, 1.15),

    "precision_bbox_anchor": (0.57, 0.25),
    "recall_bbox_anchor": (0.60, 0.39),
    "precision_legend_fontsize": 16,

    "xlabel_fontsize": 20,
    "ylabel_fontsize": 20,
    "yticks": [0, 0.25, 0.5, 0.75, 1.0],
    "xticks": [0, 0.25, 0.5, 0.75, 1.0],

}

def plot_benchmark(args, ax, benchmark_data, title="Test"):

    ax.set_title(title, fontsize=PLOT_PROPS["title_fontsize"])

    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])

    if "HumanEval" in title:
        pass

    for model in args.models:

        ax.plot(benchmark_data[model]["X"], 
            benchmark_data[model]["Y"],
            color=PLOT_PROPS[model]["color"],
            linestyle=PLOT_PROPS[model]["linestyle"],
            marker=PLOT_PROPS[model]["marker"],
            markersize=PLOT_PROPS["markersize"])
        
        head_width = .04
        X = benchmark_data[model]["X"]
        Y = benchmark_data[model]["Y"]
        if "FOLIO ($\math" in title:

            for k in range(len(X) - 1):
                middle_x = (X[k] + X[k + 1]) / 2.0
                middle_y = (Y[k] + Y[k + 1]) / 2.0
                slope = ((Y[k] - Y[k + 1])+0.00000001) / ((X[k] - X[k + 1])+0.00000001)

                ax.arrow(middle_x, middle_y, 
                         -0.0000000001, 
                         -(slope * 0.0000000001), 
                         shape='full', 
                         overhang=0.2,
                         lw=0, length_includes_head=True,
                        head_width=head_width, 
                        color=PLOT_PROPS[model]["color"],
                        label="_nolegend_")

    pearson = benchmark_data["pearson_correlation"]
    
    if "p_value" not in benchmark_data:
        benchmark_data["p_value"] = -1.0
        for x in benchmark_data:
            if x in ['gpt-4o','gpt-4o-mini','gpt-3.5-turbo','llama-3-8b','phi-3','mistral','trend_line','pearson_correlation']:
                continue
            benchmark_data["p_value"] = benchmark_data[x]
        assert benchmark_data["p_value"] != -1.0
    p_value = benchmark_data["p_value"]

    # Determine the correct markers for the intercepts on the trendline.
    slope, intercept = np.polyfit(benchmark_data["trend_line"][0], 
        benchmark_data["trend_line"][1], 
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

    ax.plot(X,
        Y,
        linestyle=(0, (1, 4)),
        marker="x",
        clip_on=False,
        zorder=10,
        color="tab:gray")

    ax.annotate("$\\rho = %.2f$\n$p=%.2f$" % (pearson, p_value), 
        xy=(-0.03, 0.9),
        fontsize=PLOT_PROPS["annotation_size"])
 
def process_human_eval(args, human_eval, dataset, limit=float("inf")):

    human_eval_scores = {
        "gpt-3.5-turbo": .68,
        "gpt-4o": .902,
        "gpt-4o-mini": .872,
        "llama-3-8b": .616,
        "mistral": .421,
        "phi-3": .622,
    }

    for model in args.models:

        data = [(k, v) 
                for k, v in human_eval[dataset][model].items() 
                if k < limit]
        data = sorted(data)

        Y = np.mean([x[1] for x in data])

        human_eval[dataset][model]["X"] = [Y]
        human_eval[dataset][model]["Y"] = [human_eval_scores[model]]

    X = [human_eval[dataset][model]["X"][0] for model in args.models]
    Y = [human_eval_scores[model] for model in args.models]

    x_bounds = [-0.05, 0.75]
    from scipy.stats import linregress, pearsonr
    slope, intercept, r_value, p_value, std_err = linregress(X, Y)
    line = [(slope * current_x) + intercept for current_x in x_bounds]


    correlation, pvalue = pearsonr(X, Y)
    human_eval[dataset]["trend_line"] = (x_bounds, line)
    human_eval[dataset]["pearson_correlation"] = correlation
    human_eval[dataset]["p_value"] = pvalue


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default="/tmp/corelation",
                        required=False)
    parser.add_argument("--save-location", type=str, default="")    
    parser.add_argument("--runs", type=int, nargs="+",
                        default=range(2))
    parser.add_argument("--datasets", type=str, nargs="+",
                        default= ["plogic", "fol_human"])
    parser.add_argument("--models", type=str, nargs="+",
                        default=["gpt-3.5-turbo", "gpt-4o", "llama-3-8b",
                        "mistral", "phi-3", "gpt-4o-mini"])
    args = parser.parse_args()

    if args.save_location == "":
        args.save_location = args.base_dir

    logieval_figures = get_data_dict("%s/logicbench_figures.pk" % (args.base_dir))
    folio_figures = get_data_dict("%s/folio_figures.pk" % (args.base_dir))
    folio_premises_figures = get_data_dict("%s/folio_premises_figures.pk" % (args.base_dir))
    human_eval = get_data_dict("%s/humaneval.pk" % (args.base_dir))

    print(logieval_figures)

    assert logieval_figures is not None
    assert folio_figures is not None
    assert folio_premises_figures is not None

    human_eval_dataset = "regex"
    human_eval_limit = 11
    process_human_eval(args, human_eval, human_eval_dataset,
                       limit=human_eval_limit)

    pass

    fig = plt.Figure(figsize=PLOT_PROPS["figsize"])
    gs = GridSpec(nrows=1, ncols=5, figure=fig,
                  hspace=0.07, wspace=0.05)

    axs = []
    for i in range(5):
        ax = fig.add_subplot(gs[0, i])
        axs.append(ax)


    plot_benchmark(args, axs[0], folio_premises_figures[5], title="FOLIO ($\mathcal{A}$)")
    plot_benchmark(args, axs[1], folio_premises_figures[1], title="FOLIO ($\mathcal{I}$; BLEU)")
    plot_benchmark(args, axs[2], folio_premises_figures[2], title="FOLIO ($\mathcal{I}$; ROUGE)")
    plot_benchmark(args, axs[3], folio_premises_figures[3], title="FOLIO ($\mathcal{I}$; METEOR)")
    plot_benchmark(args, axs[4], folio_premises_figures[4], title="FOLIO ($\mathcal{I}$; BERT)")
    #plot_benchmark(args, axs[5], human_eval[human_eval_dataset], title="HumanEval ($\mathcal{A}$)")

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

    labels = [PLOT_PROPS[model]["label"] for model in args.models]
    fig.legend(labels=labels, frameon=False,
               ncols=6,
               bbox_to_anchor=PLOT_PROPS["legend_bbox_anchor"],
               fontsize=PLOT_PROPS["legend_fontsize"],
               handlelength=PLOT_PROPS["legend_handlelength"])

    fig.text(0.46, -0.1, "Parameterized $\\forall$uto$\exists$$\lor\!\land$L Score",
             fontsize=PLOT_PROPS["xlabel_fontsize"])

    #fig.savefig("%s/corelation.png" % (args.base_dir), bbox_inches="tight")
    fig.savefig("%s/figure_17.pdf" % (args.save_location), bbox_inches="tight")
