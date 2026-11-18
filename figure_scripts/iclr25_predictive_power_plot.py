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
    'text.latex.preamble': "\\usepackage{amsmath}", 
    "font.family": "sans-serif",
})


PLOT_PROPS = {

    "gpt-4o": {

        "color": "tab:blue",
        "linestyle": "-",
        "label": "GPT-4",
    },

    "gpt-4o1": {

        "color": "tab:blue",
        "linestyle": "-",
        "label": "OpenAI o1",
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
        "linestyle": (0, (3, 10, 1, 10)),
        "label": "Mistral",
    },

    "phi-3": {

        "color": "tab:pink",
        "linestyle": (0, (5, 10)),
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

    "figsize": (8, 4),

    "title_fontsize": 20,
    "legend_fontsize": 20,
    "xticklabel_fontsize": 18,
    "yticklabel_fontsize": 18,
    "legend_handlelength": 3,
    "legend_bbox_anchor": (0.9, 0.98),

    "precision_bbox_anchor": (0.57, 0.25),
    "recall_bbox_anchor": (0.60, 0.39),
    "precision_legend_fontsize": 16,

    "xlabel_fontsize": 20,
    "ylabel_fontsize": 20,
    "yticks": [0, 0.25, 0.5, 0.75, 1.0]
}

power_values = {
    "FOLIO(NL) $:$ Accuracy": 89,
    "FOLIO($\mathcal{A}$) $:$ Accuracy": 87,
    "BBH(R) $:$ Accuracy": 86,
    "FOLIO(FOL) $:$ Accuracy": 85,
    "LogiEval(PL) $:$ Accuracy": 81,
    "HumanEval($\mathcal{A}$) $:$ Accuracy": 78,
    "LogiEval(FOL) $:$ Accuracy": 73,
    "FOLIO($\mathcal{I}$) $:$ METEOR": 73,
    "FOLIO($\mathcal{I}$) $:$ BLEU": 72,
    "FOLIO($\mathcal{I}$) $:$ ROUGE": 71,
    "FOLIO($\mathcal{I}$) $:$ BERT": 68,   
}

# color values for fl-based vs NL based metrics based on order
# in power_values {} above.
fl_metric_color = "tab:blue"
nl_metric_color = "#FF4500"
bar_label_colors = [fl_metric_color] * 7 + [nl_metric_color] * 4


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default="/tmp/rebuttal_data",
                        required=False)

    args = parser.parse_args()

    import os
    os.makedirs(args.base_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=PLOT_PROPS["figsize"])


    metric_names = []
    predictive_power = []
    color = []

    for current in power_values:
        metric_names.append(current)
        predictive_power.append(power_values[current] / 100.0)
        # color.append(power_values[current][1])

    rects = ax.bar(metric_names, predictive_power, width=0.75, label=metric_names, color="tab:blue")
    ax.set_xticklabels(metric_names, rotation = 50)

    bar_labels = [str(round(v, 2)) if v >= 0.05 else "" 
                  for v in rects.datavalues]
    
    annotations = ax.bar_label(rects, labels=bar_labels, padding=3, fontsize=14)
    
    assert len(annotations) == len(bar_label_colors)
    for annotation, bar_label_color in zip(annotations, bar_label_colors):
        annotation.set(color=bar_label_color)

    for rect, label, bar_label_color in zip(rects, power_values, bar_label_colors):

        import matplotlib.colors as mcolors
        rect.set_facecolor(mcolors.to_rgba("#A3A3A3", alpha= 0.20))
        rect.set_edgecolor("black")

        x, y = rect.get_xy()
        cx = x + rect.get_width() / 2 - 0.11
        ax.annotate(label, 
            (cx, y + 0.04), color=bar_label_color, fontweight="bold",
            rotation=90, fontsize=14)

    ax.set_ylabel("$\mathcal{P}_{\\forall\\text{uto}\exists\lor\!\land\\text{L}}$(X)", 
    fontsize=PLOT_PROPS["ylabel_fontsize"])
    ax.set_xlabel('Benchmark X (Annotated bars)', fontsize=PLOT_PROPS["xlabel_fontsize"])
    
    ax.tick_params(
        axis="y", labelsize=PLOT_PROPS["yticklabel_fontsize"])
    ax.tick_params(
        axis="x", labelsize=PLOT_PROPS["xticklabel_fontsize"])
    ax.set_xticklabels([])
    ax.set_ylim([0, 1.00])

    ticklabels = ["$0$", "$.25$", "$.5$",
        "$.75$", "$1$"]
    ax.set_yticks(PLOT_PROPS["yticks"])
    ax.set_yticklabels(ticklabels)

    ax.spines[['right', 'top']].set_visible(False)
    
    ax.margins(x=0.02)

    # https://matplotlib.org/stable/users/explain/text/annotations.html#annotations
    import matplotlib.patches as mpatches

    fl_an_y = 0.98
    fl_an_x = 1
    ax.annotate("FL-based metrics", 
                xy=(3.0, fl_an_y - 0.018),
                horizontalalignment="center",
                color=fl_metric_color,
                fontsize=14)

    arr = mpatches.FancyArrowPatch((-0.4, fl_an_y), (1.7, fl_an_y),
                                arrowstyle='<-,head_width=.15', 
                                mutation_scale=20,
                                color=fl_metric_color)
    ax.add_patch(arr)
    arr = mpatches.FancyArrowPatch((4.3, fl_an_y), (6.4, fl_an_y),
                                arrowstyle='->,head_width=.15', 
                                mutation_scale=20,
                                color=fl_metric_color)
    ax.add_patch(arr)


    fl_an_y = 0.87
    ax.annotate("Length Dependent\nNL Metrics", 
                xy=(8.6, fl_an_y - 0.035),
                horizontalalignment="center",
                color=nl_metric_color,
                fontsize=14)

    arr = mpatches.FancyArrowPatch((9.5, fl_an_y), (10.5, fl_an_y),
                                arrowstyle='->,head_width=.15', 
                                mutation_scale=20,
                                color=nl_metric_color)
    ax.add_patch(arr)
    arr = mpatches.FancyArrowPatch((6.6, fl_an_y), (7.6, fl_an_y),
                                arrowstyle='<-,head_width=.15', 
                                mutation_scale=20,
                                color=nl_metric_color)
    ax.add_patch(arr)


    #fig.savefig("%s/plot.png" % (args.base_dir), bbox_inches="tight")
    fig.savefig("%s/figure_5.pdf" % (args.base_dir), bbox_inches="tight")
    pass
