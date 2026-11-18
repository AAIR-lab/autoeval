import matplotlib.pyplot as plt
import json
import argparse
import os
import pickle

models = ["gpt-3.5-turbo", "gpt-4o", "phi-3", "mistral", "llama-3-8b"]
datasets = ["ksat", "plogic", "fol", "fol_human", "regex"]

inner_labels = ['True Labels', 'False Labels']
#inner_colors = ['#1f77b4', '#d62728', ]
inner_colors = ['tab:green', 'tab:red', ]

outer_labels = ['True Positive','False Negative', 'Non-compliant (True Label)', 'True Negative', 'False Positive', 'Non-compliant (False Label)']
outer_figure_labels = ['TP','FN', '', 'TN', 'FP', '']
#outer_colors = ['#0d3d66', 'cyan', '#aec7e8', '#8c1a1a', 'tomato', '#ff9896']
#outer_colors = ['darkgreen', 'greenyellow', 'palegreen', 'darkred', 'orange', 'lightcoral']
outer_colors = ['darkgreen', 'limegreen', 'darkgray', 'firebrick', 'lightcoral', 'lightgray']
start_angle = 90
inner_radius = 0.8
inner_width = 0.4
outer_width = 0.4

PLOT_PROPS = {
    "figsize": (18, 16),

    "title_fontsize": 20,
    "legend_fontsize": 18,
    "xticklabel_fontsize": 18,
    "yticklabel_fontsize": 18,
    "legend_handlelength": 2,
    "legend_bbox_anchor": (0.9, 0.94),

}

dataset_metrics = {
    "ksat": ("num_operators", 0, 20),
    "plogic": ("num_operators", 0, 20),
    "fol": ("num_operators", 0, 20),
    "fol_human": ("num_operators", 0, 20),
    "regex": ("depth", 0, 20)
}



model_title = {
    "gpt-3.5-turbo": "ChatGPT",
    "gpt-4o": "GPT-4o",
    "phi-3": "Phi3",
    "mistral": "Mistral",
    "llama-3-8b": "Llama3",
    "yi-1.5-34b": "Yi-1.5",
    "qwen-2.5-14b": "Qwen2.5(14B)",
    "qwen-2.5-1.5b": "Qwen2.5(1.5B)",
    "phi-3.5-mini": "Phi3.5-mini",
    "ministral": "Ministral",
    "llama-3-70b": "Llama3(70B)",
    "llama-3.2-1b": "Llama3.2",
    "llama-3.1-8b": "Llama3.1",
    "granite": "Granite",
    "gemma-2-9b": "Gemma2"
}

def read_data(path, model, dataset, metric, lower_bound, upper_bound):
    tp = tn = fp = fn = tincompliant = fincompliant = total = 0
    for current_batch in range(10):
        file_path = "%s/compiled_results/run%i/%s/dataset_nlfs_gpt-4o_verfied.json"%(path,current_batch,dataset)
        #file_path = "%s/%s_equiv_results/run%i/%s/dataset_nlfs_gpt-4o_verfied.json" % (path, model, current_batch, dataset)
        with open(file_path,"r") as current_file:
            data = json.load(current_file)
            for current_entry in data["data"]:
                if metric not in current_entry:
                    raise Exception("The metric %s is not measured in %s." % (metric, dataset))
                if lower_bound <= current_entry[metric] <= upper_bound:
                    label = False if current_entry["verification"]["is_equivalent"] is None else current_entry["verification"]["is_equivalent"]
                    prediction = current_entry["llm_verification"][model]["is_equivalent"]
                    total += 1
                    if prediction is None:
                        if label:
                            tincompliant += 1
                        else:
                            fincompliant += 1
                    elif label:
                        if prediction:
                            tp += 1
                        else:
                            fn += 1
                    elif prediction:
                        fp += 1
                    else:
                        tn += 1
    return tp, fp, tn, fn, tincompliant, fincompliant, total


def create_figure(ax, path, model, dataset, metric, lower_bound, upper_bound, data_dict=None, update_dict=False):
    if update_dict:
        tpos, fpos, tneg, fneg, tincompliant, fincompliant, total = read_data(path, model, dataset, metric, lower_bound, upper_bound)
        data_dict[(model,dataset)] = (tpos, fpos, tneg, fneg, tincompliant, fincompliant, total)
    else:
        tpos, fpos, tneg, fneg, tincompliant, fincompliant, total = data_dict[(model,dataset)]
    true_labels = tpos + fneg + tincompliant
    false_labels = tneg + fpos + fincompliant

    # Inner ring of true and false labels
    ax.pie(
        [true_labels,false_labels],
        colors=inner_colors,
        startangle=start_angle,
        radius=inner_radius,
        wedgeprops={'width': inner_width, 'edgecolor': 'w'}
    )
    # Outer ring
    ax.pie(
        [tpos, fneg, tincompliant, tneg, fpos, fincompliant],
        #labels=outer_figure_labels,
        colors=outer_colors,
        startangle=start_angle,
        radius=inner_radius + inner_width,  # Slightly larger radius than inner ring
        wedgeprops={'width': outer_width, 'edgecolor': 'w'}
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default="rebuttal_data/zero-shot",
                        required=False)
    parser.add_argument("--save-location", type=str, default="")
    parser.add_argument("--save-name", type=str, default="results")
    parser.add_argument("--runs", type=int, nargs="+",
                        default=range(10))
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["ksat","plogic", "fol", "fol_human", "regex"])

    #MODELS = ["gpt-4o", "gpt-3.5-turbo", "mistral", "phi-3", "llama-3-8b", "gemma-2-9b", "granite", "llama-3.1-8b"]
    MODELS = ["llama-3.2-1b","llama-3-70b","ministral","phi-3.5-mini","qwen-2.5-1.5b","qwen-2.5-14b","yi-1.5-34b"]
    parser.add_argument("--models", type=str, nargs="+",
                        default=MODELS)
    
    parser.add_argument("--clean", default=False, action="store_true")

    args = parser.parse_args()

    if args.save_location == "":
        args.save_location = args.base_dir

    if args.clean:
        try:
            os.remove("%s/parsed_data.pkl" % (args.base_dir))
        except Exception:
            pass
        data_dict = {}
        update_dict = True

    else:
        try:
            with open("%s/parsed_data.pkl" % (args.base_dir), "rb") as fh:
                data_dict = pickle.load(fh)
                update_dict = False
        except Exception:
            data_dict = {}
            update_dict = True


    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=len(args.models), ncols=len(args.datasets), figsize=(18, 3 * len(args.models)))

    for model_idx, model in enumerate(args.models):
        axes[model_idx, 0].set_ylabel(model_title[model], size=PLOT_PROPS["yticklabel_fontsize"])
        for dataset_idx, dataset in enumerate(args.datasets):
            create_figure(axes[model_idx, dataset_idx], args.base_dir, model, dataset, *dataset_metrics[dataset], data_dict, update_dict)

    for dataset_idx, dataset in enumerate(args.datasets):
        axes[0, dataset_idx].set_title(dataset, size=PLOT_PROPS["title_fontsize"])

    fig.legend(labels=inner_labels + outer_labels, frameon=False,
               ncols=4,
               bbox_to_anchor=PLOT_PROPS["legend_bbox_anchor"],
               fontsize=PLOT_PROPS["legend_fontsize"],
               handlelength=PLOT_PROPS["legend_handlelength"])

    #fig.savefig("%s/plot.png" % (args.base_dir), bbox_inches="tight")
    fig.savefig("%s/%s.pdf" % (args.save_location,args.save_name), bbox_inches="tight")

    with open("%s/parsed_data.pkl" % (args.base_dir), "wb") as fh:
        pickle.dump(data_dict, fh)