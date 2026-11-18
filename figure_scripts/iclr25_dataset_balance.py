import argparse
import json
import re
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import math

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
})

def depth(target):
    return target['depth']

def density(target):
    return target['density']

def edges(target):
    return target['num_edges']

def nodes(target):
    return target['num_nodes']

def or_count(target):
    return target["fs"].count("v")

def and_count(target):
    return target["fs"].count("∧")

def not_count(target):
    filter_value = 0
    for current_index in [m.start() for m in re.finditer("¬", target["fs"])]:
        if target["fs"][current_index+1].lower() not in ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']:
            filter_value += 1
    return filter_value

NOT_REGEX_DATASETS = ['plogic','fol','fol_human']

dataset_label = {
    "plogic": "Propositional Logic$(12)$",
    "fol": "FOL$(8, 12)-$S",
    "fol_human": "FOL$(8, 12)-$E",
    "regex": "Regular Expression$(2)$"
}

PLOT_PROPS = {
    "not_regex" : {
        "depth" : {
            "function": depth,
            "bin_size": 3,
            "min_value": 1,
            "max_value": 40,
            "min_change_rate": 1,
            "Title": "CFG Parse Tree Depth"
        },
        "and" : {
            "function": and_count,
            "bin_size": 1,
            "min_value": 0,
            "max_value": 17,
            "min_change_rate": 1,
            "Title": "\\# of $\\land$ Operators"
        },
        "or" : {
            "function": or_count,
            "bin_size": 1,
            "min_value": 0,
            "max_value": 18,
            "min_change_rate": 1,
            "Title": "\\# of $\\lor$ Operators"
        },
        "not" : {
            "function": not_count,
            "bin_size": 1,
            "min_value": 0,
            "max_value": 6,
            "min_change_rate": 1,
            "Title": "\\# of $\\neg$ Operators"
        }
    },
    "regex" : {
        "density" : {
            "function": density,
            "bin_size": 0.1,
            "min_value": 0,
            "max_value": 0.5,
            "min_change_rate": 0.1,
            "Title": "Density"
        },
        "edges" : {
            "function": edges,
            "bin_size": 1,
            "min_value": 1,
            "max_value": 7,
            "min_change_rate": 1,
            "Title": "\\# of Edges"
        },
        "nodes" : {
            "function": nodes,
            "bin_size": 1,
            "min_value": 1,
            "max_value": 7,
            "min_change_rate": 1,
            "Title": "\\# of Nodes"
        }
    },
    "figsize": (18, 9),
    "title_fontsize": 20,
    "legend_fontsize": 20,
    "ylabel_fontsize": 20,
    "ylabel_padding": 40,
    "xlabel_fontsize": 14
}

def read_data(path,target_dict,regex=False):

    plot_name = 'regex' if regex else 'not_regex'
    
    for current in PLOT_PROPS[plot_name]:
        if current not in target_dict:
            target_dict[current] = {}

    with open(path) as input_file:
        dataset = json.load(input_file)
        for idx, datum in enumerate(dataset["data"]):
            for current in PLOT_PROPS[plot_name]:
                filter_value = PLOT_PROPS[plot_name][current]["function"](datum)
                if filter_value not in target_dict[current]:
                    target_dict[current][filter_value] = 1
                else:
                    target_dict[current][filter_value] += 1

def generate_graph(ax, bin_info, counts):
    last_bin_plus = False

    set_of_bins = []
    bin_labels = []
    current = bin_info['min_value']
    while current <= bin_info['max_value']:
        set_of_bins.append([current, current + bin_info['bin_size'],0])
        current += bin_info['bin_size']
        if math.isclose(set_of_bins[-1][1] - set_of_bins[-1][0],bin_info['min_change_rate']):
            if isinstance(set_of_bins[-1][0], float): 
                bin_labels.append("%.1f"%(set_of_bins[-1][0]))
            else:
                bin_labels.append("%i"%(set_of_bins[-1][0]))
        else:
            if isinstance(set_of_bins[-1][0], float):
                bin_labels.append("%.1f-%.1f"%(set_of_bins[-1][0],set_of_bins[-1][1] - bin_info['min_change_rate'])) 
            else:
                bin_labels.append("%i-%i"%(set_of_bins[-1][0],set_of_bins[-1][1] - bin_info['min_change_rate'])) 
    
    for current_count in counts:
        added = False
        for current_bin in set_of_bins:
            if (math.isclose(current_bin[0],current_count) or current_bin[0] < current_count) and (current_count < current_bin[1] and not math.isclose(current_bin[1],current_count)):
                current_bin[2] += counts[current_count]
                added = True
                break
        if not added:
            set_of_bins[-1][2] += counts[current_count]
            last_bin_plus = True

    for c_bin, c_label in zip(set_of_bins,bin_labels):
        if c_bin[2] == 0:
            set_of_bins.remove(c_bin)
            bin_labels.remove(c_label)

    if last_bin_plus:
        bin_labels[-1] = str(set_of_bins[-1][0]) + "+"

    ax.pie([bin[2] for bin in set_of_bins],labels=bin_labels,wedgeprops={'linewidth':1,'edgecolor':'k'},textprops={'fontsize':PLOT_PROPS["xlabel_fontsize"]},colors=plt.get_cmap("Blues")(np.linspace(0, 1, len(set_of_bins))),labeldistance=1.11)

def create_non_regex_figure(count_data,args):
    fig, axes = plt.subplots(nrows=len(NOT_REGEX_DATASETS), ncols=len(PLOT_PROPS['not_regex']), figsize=PLOT_PROPS["figsize"])
    for type_idx, current_type in enumerate(NOT_REGEX_DATASETS):
        axes[type_idx,0].set_ylabel(dataset_label[current_type], fontsize=PLOT_PROPS["ylabel_fontsize"],labelpad=PLOT_PROPS["ylabel_padding"])
        for field_idx, current_field in enumerate(PLOT_PROPS['not_regex']):
            generate_graph(axes[type_idx,field_idx],PLOT_PROPS['not_regex'][current_field],count_data[current_type][current_field])
            if type_idx == 0:
                axes[type_idx,field_idx].set_title(PLOT_PROPS['not_regex'][current_field]["Title"], size=PLOT_PROPS["title_fontsize"])

    #fig.savefig("%s/non_regex_count_plot.png" % (args.base_dir), bbox_inches="tight")
    fig.savefig("%s/figure_10.pdf" % (args.save_location), bbox_inches="tight")


def create_regex_figure(count_data, args):
    fig, axes = plt.subplots(nrows=1, ncols=len(PLOT_PROPS['regex']), figsize=PLOT_PROPS["figsize"])
    axes[0].set_ylabel(dataset_label["regex"], fontsize=PLOT_PROPS["ylabel_fontsize"],labelpad=PLOT_PROPS["ylabel_padding"])
    for field_idx, current_field in enumerate(PLOT_PROPS['regex']):
        print(count_data[current_type][current_field])
        generate_graph(axes[field_idx],PLOT_PROPS['regex'][current_field],count_data[current_type][current_field])
        axes[field_idx].set_title(PLOT_PROPS['regex'][current_field]["Title"], size=PLOT_PROPS["title_fontsize"])

    #fig.savefig("%s/regex_count_plot.png" % (args.base_dir), bbox_inches="tight")
    fig.savefig("%s/figure_9.pdf" % (args.save_location), bbox_inches="tight")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default="/tmp/folio_results_verified/results",
                    required=False)
    parser.add_argument("--save-location", type=str, default="") 
    args = parser.parse_args()

    if args.save_location == "":
        args.save_location = args.base_dir

    counted_fields = {}

    for current_type in NOT_REGEX_DATASETS:
        counted_fields[current_type] = {}
        for current_batch in range(10):
            read_data("%s/batch%i/%s/dataset.json"%(args.base_dir,current_batch,current_type),counted_fields[current_type])
        
    counted_fields["regex"] = {}
    for current_batch in range(10):
        read_data("%s/batch%i/regex/dataset.json"%(args.base_dir,current_batch),counted_fields[current_type],True)
    
    create_non_regex_figure(counted_fields,args)
    create_regex_figure(counted_fields,args)