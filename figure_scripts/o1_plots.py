import sys
import pathlib

#_ROOT = (pathlib.Path(__file__).parent / "../../").as_posix()
#sys.path.append(_ROOT)

import argparse
import json
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

from svgpathtools import svg2paths
from svgpath2mpl import parse_path

def generate_marker_from_svg(svg_path):
    image_path, attributes = svg2paths(svg_path)

    image_marker = parse_path(attributes[0]['d'])

    image_marker.vertices -= image_marker.vertices.mean(axis=0)

    image_marker = image_marker.transformed(mpl.transforms.Affine2D().rotate_deg(180))
    image_marker = image_marker.transformed(mpl.transforms.Affine2D().scale(-1,1))

    return image_marker

# https://stackoverflow.com/questions/51802684/multi-lineswith-different-markers-in-legend
# (with modifications to make it work with fig.legend())
from matplotlib.legend_handler import HandlerTuple
class HandlerLinesVertical(HandlerTuple):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):
        ndivide = len(orig_handle) + 1
        a_list = []
        for i, handle in enumerate(orig_handle):
            # Position lines vertically based on the number of handles
            y = (height / float(ndivide)) * i - ydescent
            
            # Create a vertical line and a point for the legend
            line = plt.Line2D(np.array([0, 1]) * width, [-y, -y])
            
            # line.update_from(handle)
            # Don't update the line from the handle. Just copy the necessary
            # params. Copying more stuff leads to transform issues when moving
            # from ax.legend() to fig.legend()
            line.set_color(handle.get_color())
            line.set_linestyle(handle.get_linestyle())
            line.set_marker("None")
            
            point = plt.Line2D(np.array([0.5]) * width, [-y])
            
            # Same for the point.
            # point.update_from(handle)
            point.set_marker(handle.get_marker())
            point.set_markersize(PLOT_PROPS["legend_markersize"])
            point.set_color(handle.get_color())

            # No need to apply transformation.            
            # Apply transformation to make the legend work with the figure
            # for artist in [line, point]:
            #     artist.set_transform(trans)
            
            a_list.extend([line, point])
        return a_list

def read_dataset(data_dict, dataset_filepath, filter_field=None):

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

        compl_list, equiv_list, verify_list = data_dict.setdefault(
            filter_value, ([], [], []))

        compl_list.append(int(error is None))
        equiv_list.append(is_equivalent)

        try:
            verify_list.append(int(datum["llm_verification"]["is_equivalent"]))
            pass
        except Exception:
            verify_list.append(None)
            pass

    pass

def get_dataset_filepath(base_dir, run_no, dataset_type, model_name):

    return "%s/run%d/%s/%s.verified.json" % (
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

def parse_equiv_data(bdir):
    data_dict = {}
    for dataset_type in ['fol','fol_human','ksat','plogic','regex']:
        data_dict[dataset_type] = {}
        for model_name in ['gpt-3.5-turbo','llama-3-8b','mistral','phi-3']:
            data_dict[dataset_type][model_name] = {}
            for run_no in range(10):
                # print("%s/equiv_results/%s_equiv_results/run%i/%s/dataset_nlfs_gpt-4o_verfied.json"%(bdir,model_name,run_no,dataset_type))
                read_dataset(data_dict[dataset_type][model_name],"%s/equiv_results/%s_equiv_results/run%i/%s/dataset_nlfs_gpt-4o_verfied.json"%(bdir,model_name,run_no,dataset_type))
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

def bin(data, width=3):

    # https://stackoverflow.com/questions/21921178/binning-a-numpy-array
    results = data[:(data.size // width) * width].reshape(-1, width).mean(axis=1)
    return results

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

    x = bin(np.asarray(x))
    y = bin(np.asarray(y))

    line = ax.plot(x, y, color=plot_props["color"],
            marker=plot_props["marker"],
            markersize=PLOT_PROPS["markersize"],
            markevery=[0, len(x)//2, len(x) - 1],
            linestyle=plot_props["linestyle"],
            alpha=plot_props["alpha"],)

    # ax.fill_between(x, y - y_err, y + y_err, color=plot_props["color"],
    #                 alpha=0.05)
    return line[0]


PLOT_PROPS = {

    "gpt-4o": {

        "color": "tab:blue",
        "linestyle": "-",
        "label": "GPT-4",
    },

    "gpt-4o1": {

        "color": "tab:blue",
        "linestyle": "-",
        "marker": "None",
        "label": "OpenAI o1",
        "alpha": 1.0,
    },

    "deepseek-r1": {

        "color": "tab:orange",
        "linestyle": "-",
        "marker": "None",
        "label": "DeepSeek R1",
        "alpha": 0.5,
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

    "figsize": (6, 9),
    "markersize": 4,
    "legend_markersize": 5,
    "title_fontsize": 20,
    "legend_fontsize": 14,
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default="/tmp/o1",
                        required=False)
    parser.add_argument("--save-location", type=str, default="")
    parser.add_argument("--runs", type=int, nargs="+",
                        default=range(2))
    parser.add_argument("--datasets", type=str, nargs="+",
                        default= ["plogic", "fol_human"])
    parser.add_argument("--models", type=str, nargs="+",
                        default=["gpt-4o1", "deepseek-r1"])
    parser.add_argument("--clean", default=True, action="store_true")


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
    handles = {}
    sc_handles = []
    autoscore_handles = []
    for i, dataset_type in enumerate(args.datasets):

        ax0 = fig.add_subplot(gs[0, i])

        ticklabels = ["$0$", "$.25$", "$.5$",
            "$.75$", "$1$"]
        
        ax1 = ax0.twinx()

        ax0.set_title(PLOT_PROPS[dataset_type]["title"],
                      fontsize=PLOT_PROPS["title_fontsize"])
        for model_name in args.models:

            l1 = plot(ax0, data_dict[dataset_type][model_name], 0,
                 {"color": PLOT_PROPS[model_name]["color"],
                  "marker": PLOT_PROPS[model_name]["marker"],
                  "alpha": PLOT_PROPS[model_name]["alpha"],
                    "linestyle": "--",
                    "label": "Syntactic Compliance"})
            ax0.set_ylim([-0.1, 1.1])

            l2 = plot(ax1, data_dict[dataset_type][model_name], 1,
                 {"color": PLOT_PROPS[model_name]["color"],
                  "marker": PLOT_PROPS[model_name]["marker"],
                  "alpha": PLOT_PROPS[model_name]["alpha"],
                    "linestyle": "-",
                    "label": "Accuracy"})
            ax1.set_ylim([-0.1, 1.1])

            if i == 0:
                ax1.set_yticklabels([])
                ax0.set_ylabel("Syntactic Compliance",
                               fontsize=PLOT_PROPS["ylabel_fontsize"],
                               color="k")
                ax0.tick_params(axis='y', labelcolor="k")
                ax0.set_yticklabels(ticklabels)
            else:
                ax0.set_yticklabels([])
                ax1.set_ylabel("$\\forall$uto$\exists$$\lor\!\land$L Score",
                               fontsize=PLOT_PROPS["ylabel_fontsize"],
                               color="k")
                ax1.tick_params(axis='y', labelcolor="k")
                ax1.set_yticklabels(ticklabels)

            for ax in [ax0, ax1]:
                ax.set_yticks(PLOT_PROPS["yticks"])
                ax.set_xticks(PLOT_PROPS[dataset_type]["xticks"])
                ax.tick_params(
                    axis="y", labelsize=PLOT_PROPS["yticklabel_fontsize"])
                ax.tick_params(
                    axis="x", labelsize=PLOT_PROPS["xticklabel_fontsize"])
                ax.yaxis.grid(True, linestyle="--", alpha=0.45)
                ax.set_xlim(PLOT_PROPS[dataset_type]["xlim"])

            handles[model_name] = [l1, l2]
            sc_handles.append(l1)
            autoscore_handles.append(l2)

    fig.text(0.28, 0.63, "\\# of Operators: $\land, \lor, \\neg$",
             fontsize=PLOT_PROPS["xlabel_fontsize"])
    legend_handles = [tuple(handles[model_name]) for model_name in args.models]
    legend_labels = [PLOT_PROPS[model_name]["label"] for model_name in args.models]
    
    # create the legend
    # fig.legend(legend_handles, legend_labels, 
    #            ncols=2,
    #            frameon=False,
    #            handler_map={tuple: HandlerLinesVertical()},
    #            fontsize=PLOT_PROPS["legend_fontsize"],
    #            bbox_to_anchor=(0.89, 0.98),
    #             handlelength=2.5, handleheight=3)

    offset = 0.02
    sc_line = lines.Line2D([-0.018, -0.018], [0.65 - offset, 0.685 - offset], 
        linewidth=1.5,
        linestyle="--",
        color= PLOT_PROPS["gpt-4o1"]["color"])
    fig.add_artist(sc_line)
    fig.text(-0.03, 0.7 - offset, "OpenAI o1",
             fontsize=PLOT_PROPS["legend_fontsize"],
             rotation=90)

    sc_line = lines.Line2D([-0.018, -0.018], [0.80 - offset, 0.835 - offset], 
        linewidth=1.5,
        linestyle="--",
        color= PLOT_PROPS["deepseek-r1"]["color"],
        alpha=PLOT_PROPS["deepseek-r1"]["alpha"])
    fig.add_artist(sc_line)
    fig.text(-0.03, 0.85 - offset, "DeepSeek R1",
             fontsize=PLOT_PROPS["legend_fontsize"],
             rotation=90)

    x_offset = 1.05
    sc_line = lines.Line2D([-0.018 + x_offset, -0.018 + x_offset], [0.65 - offset, 0.685 - offset], 
        linewidth=1.5,
        linestyle="-",
        color= PLOT_PROPS["gpt-4o1"]["color"],
        alpha=PLOT_PROPS["gpt-4o1"]["alpha"])
    fig.add_artist(sc_line)
    fig.text(-0.03 + x_offset, 0.7 - offset, "OpenAI o1",
             fontsize=PLOT_PROPS["legend_fontsize"],
             rotation=90)

    sc_line = lines.Line2D([-0.018 + x_offset, -0.018 + x_offset], [0.80 - offset, 0.835 - offset], 
        linewidth=1.5,
        linestyle="-",
        color= PLOT_PROPS["deepseek-r1"]["color"],
        alpha=PLOT_PROPS["deepseek-r1"]["alpha"])
    fig.add_artist(sc_line)
    fig.text(-0.03 + x_offset, 0.85 - offset, "DeepSeek R1",
             fontsize=PLOT_PROPS["legend_fontsize"],
             rotation=90)

    
    # fig.legend(sc_handles, legend_labels, 
    #            ncols=1,
    #            frameon=False,
    #            handler_map={tuple: HandlerLinesVertical()},
    #            fontsize=PLOT_PROPS["legend_fontsize"],
    #            bbox_to_anchor=(0.4, 0.99),
    #             handlelength=1.5)

    #fig.savefig("%s/plot.png" % (args.base_dir), bbox_inches="tight")
    fig.savefig("%s/figure_6.pdf" % (args.save_location), bbox_inches="tight")
    pass
