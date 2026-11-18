#!/bin/bash

data_path="/home/asurite.ad.asu.edu/drbrambl/Desktop/explain/final_dataset"
save_dir="figures"

mkdir -p $save_dir

# Figure 3:
python3 figure_scripts/iclr25_plots.py --base-dir $data_path/zero-shot --save-location $save_dir --clean

# Figure 18:
python3 figure_scripts/individual_model_plots.py --base-dir $data_path/zero-shot --save-location $save_dir --plot-type 0

# Figure 19:
python3 figure_scripts/individual_model_plots.py --base-dir $data_path/zero-shot --save-location $save_dir --plot-type 1

# Figure 4:
python3 figure_scripts/corelation_plots.py --base-dir $data_path --save-location $save_dir

# Figure 5:
python3 figure_scripts/iclr25_predictive_power_plot.py --base-dir $save_dir

# Figure 6:
python3 figure_scripts/o1_plots.py --base-dir $data_path/o1 --save-location $save_dir --clean

# Figure 7:
python3 figure_scripts/iclr25_ksat_plot.py --base-dir $data_path/zero-shot --save-location $save_dir --clean

# Figure 8:
python3 figure_scripts/iclr25_batch_plot.py --base-dir $data_path/merged_batches --save-location $save_dir --clean

# Figures 9-10:
python3 figure_scripts/iclr25_dataset_balance.py --base-dir $data_path/zero-shot --save-location $save_dir

# Figure 11:
python3 figure_scripts/iclr25_tree_depth_plots.py --base-dir $data_path/zero-shot --save-location $save_dir --clean

# Figure 12:
python3 figure_scripts/iclr25_operator_plots.py --base-dir $data_path/zero-shot --save-location $save_dir --operator-type and --clean

# Figure 13:
python3 figure_scripts/iclr25_operator_plots.py --base-dir $data_path/zero-shot --save-location $save_dir --operator-type or --clean

# Figure 14:
python3 figure_scripts/iclr25_operator_plots.py --base-dir $data_path/zero-shot --save-location $save_dir --operator-type not --clean

# Figure 15:
python3 figure_scripts/iclr25_regex_other_plots.py --base-dir $data_path/zero-shot --save-location $save_dir --clean

# Figure 16:
python3 figure_scripts/iclr25_few_shot_plots.py --base-dir $data_path/zero-shot --few-shot-dir $data_path/few-shot --save-location $save_dir --clean

# Figure 17:
python3 figure_scripts/iclr_appendix_correlation_plots.py --base-dir $data_path --save-location $save_dir

# Figure 20:
python3 figure_scripts/individual_model_plots.py --base-dir $data_path/zero-shot --save-location $save_dir --plot-type 2

# Figure 21:
python3 figure_scripts/plot_confusion_matrices.py --base-dir $data_path/zero-shot --save-location $save_dir --clean --save-name figure_21 --models gpt-4o gpt-3.5-turbo mistral phi-3 llama-3-8b gemma-2-9b granite llama-3.1-8b

#Figure 22:
python3 figure_scripts/plot_confusion_matrices.py --base-dir $data_path/zero-shot --save-location $save_dir --clean --save-name figure_22 --models llama-3.2-1b llama-3-70b ministral phi-3.5-mini qwen-2.5-1.5b qwen-2.5-14b yi-1.5-34b

#Figure 23:
python3 figure_scripts/claude_plots.py --base-dir $data_path/zero-shot --save-location $save_dir --clean