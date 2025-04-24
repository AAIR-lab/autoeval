# AutoEval

This repository includes the code and datasets for the paper:


[Autonomous Assessment of LLM Truth Maintenance in Formal Translation Tasks without Human Labeling: Dynamic Datasets, Assessment Paradigms, and End-to-End Benchmarks](https://arxiv.org/abs/2403.18327)


## Installation
You can either choose to install this software on your system, build a docker container yourself, or use the pre-built docker container.

### Environment Setup
First, setup some required environment variables. For api keys, only export the variables that you expect to use.
```bash
export PYTHONHASHSEED=0
export OPENAI_API_KEY=<your_openai_api_key>
export ANTHROPIC_API_KEY=<your_anthropic_api_key>
export HUGGINGFACE_API_KEY=<your_huggingface_api_key>
```

### System Installation
This code has been tested on Ubuntu 22.04. To install the code, create a virtual environment in python as follows.
```bash
apt update
apt install -y python3-venv
python3 -m venv venv
source venv/bin/activate
```
Next, install the required software by executing `./scripts/install.sh`

### Docker Image Creation
A docker image may be built by executing `docker build . -t autoeval:latest`

### Use a Prebuilt Docker Image (Recommended)
We recommend using a pre-built docker image to run AutoEval.

First, pull the container.
```bash
docker pull rushangkaria/autoeval:latest
```
Next, login to the container.
```
docker run -it --name autoeval rushangkaria/autoeval:latest
```

## Datasets
Possible dataset types are
```
fol: First-order logic (Synthetic)
fol_human: First-order logic (English)
ksat: 3-SAT
plogic: Propositional Logic
regex: Regular Expressions
```

### Packaged datasets
The packaged datasets and their prompts can be found in the `dataset.zip`
file. Please unzip the zip file to view the contents. We release our datasets
under the Creative Commons v4 license.

The file hierarchy is as follows: `prompt_type -- batch_no -- dataset_type`.
Each dataset type has a corresponding `dataset.json` which contains the vanilla
dataset and contains several `<model>.verified.json` files which contain
results from FS -> NL -> FS that may be used as a dataset for computing external
metrics that are not provided.

Each dataset is a `json` with an `info` key that contains the seed and other
parameters used to generate it. The `data` field contains a list of the data.
Each data item is a dictionary with `fs` indicating the ground-truth formal syntax
and some keys like `depth` indicating the CFG tree depth for the expression.

The verified datsets contain additional fields like `llm_fs` to indicate the
llm generated formal syntax, the complete LLM response in `conversation_history`,
and the verification results in `verification`.

### Generating your own datasets
Use the following script
```
python dataset_generator.py \
    --base-dir <base_dir> \
    --dataset-type <dataset_type> \
    --depth <depth> \
    --n <branching_factor_to_sample> \
    --seed <random_seed> \
```
to generate datasets of different types. The `-h` command can give more
information about dataset specific parameters. (eg. `--num-propositions`) for
propositional logic.

### Generating new datasets
Please provide new grammars in either `nlfs.dataset.propositional_dataset` or
`nlfs.dataset.regex_dataset` to create any new dataset based on these types.
The pipeline should work out-of-the-box in that case.

For new types of formal syntax, the appropriate verifier must also be included
in `nlfs.verifier`.

## Reproducing NeurIPS-24 results

To reproduce our results reported in the main paper, use the following scripts.
You can skip the LLM or the verifier by using the `--skip-nlfs` and `--skip-verify`
options to the command below. The results and log files are stored in the same
directory. For example, batch 0, fol saves its results in 
`./dataset/zero-shot/batch0/fol`.
```bash
python3 evaluation.py --auto \
  --total-batches 10 \
  --base-dir ./dataset/zero-shot/
  --datasets fol ksat ... \
  --models gpt-3.5-turbo claude ...
```

Possible models are:
```
gpt-3.5-turbo: ChatGPT
gpt-4o: GPT-4
claude: Anthropic Claude Sonnet
phi-3: Microsoft Phi
llama-3-8b: Meta Llama-3
mistral: Mistral 7B v0.2
```
Note that you need to setup the api keys via environment variables as described
above.

For running open-source models, please setup a vLLM server using instructions
[here](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html).