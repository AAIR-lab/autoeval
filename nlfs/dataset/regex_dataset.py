import tempfile

import nltk
from nlfs.dataset.dataset import Dataset
import time
import random
import nltk
from nlfs.grammar import sentence_generator
from nlfs.vocabulary.constant_vocabulary import ConstantVocabulary
from nlfs.verifier import regex as reg
import json
import concurrent.futures
import math
import tqdm

def get_min_dfa_info(regex):

    temp_dir = tempfile.TemporaryDirectory()

    try:
       g = reg.get_graph(regex, "", temp_dir.name)
    except Exception as e:

        raise e
    finally:

        temp_dir.cleanup()

    v = len(g.nodes())
    e = len(g.edges())

    if v > 2:
        density = e / (v * (v - 1))
        density = round(density, 1)
    else:
        density = 0

    return v, e, density

class RegexDataset(Dataset):

    REGEX_GRAMMAR = [
        "s -> '(' s ')' STAR",
        "STAR -> '*'",
        "STAR -> ' '",
        "s -> s 'V' STAR",
        "s -> 'V' STAR",
    ]

    def __init__(self, grammar, constant_vocabulary,
                 filter_field="depth"):

        self.constant_vocabulary = constant_vocabulary
        self.grammar = nltk.grammar.CFG.fromstring(grammar)
        self.filter_field = filter_field
        self.sentence_fields = ["depth", "num_nodes", "num_edges", "density"]

    def _update_info(self, json_data, name,
                     seed, depth, n,
                     sample_count):

        info_dict = {"info": {
            "type": name,
            "grammar": str(self.grammar.productions()),
            "seed": seed,
            "max_depth": depth,
            "n": n,
            "sample_count": sample_count,
            "filter_field": self.filter_field,
            "variable_info": self.constant_vocabulary.get_info(),
        }}

        json_data.update(info_dict)

    def _update_metadata(self, json_data):

        metadata_dict = {"metadata": {}}
        for field in self.sentence_fields:
            metadata_dict["metadata"][field] = {"total": 0}

        json_data.update(metadata_dict)

    def _update_field_idxs(self, json_data):

        for field in self.sentence_fields:
            json_data[field] = {}

    def process_sentence(self, sentence):

        regex = self.sentence_to_regex(sentence)
        num_nodes, num_edges, density = get_min_dfa_info(regex)

        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "density": density
        }

    def add_metadata(self, json_data, metadata_key, value_key, idx):

        data_list = json_data[metadata_key].setdefault(value_key, [])
        data_list.append(idx)

        metadata = json_data["metadata"][metadata_key]
        metadata["total"] = metadata["total"] + 1
        metadata[value_key] = metadata.get(value_key, 0) + 1

    def sentence_to_regex(self, sentence):

        regex = "".join(sentence)
        regex = regex.replace(" ", "")
        return regex

    def _generate_regex(self, sentence, sentence_data, json_data,
                        regex_set):

        regex = sentence[:]
        for i, c in enumerate(regex):

            if self.constant_vocabulary.is_match(c):
                regex[i] = self.constant_vocabulary.generate()

        regex = self.sentence_to_regex(regex)

        if regex in regex_set:
            return False
        else:
            regex_set.add(regex)

        idx = len(json_data["data"])
        expr_data = {
            "idx": idx,
            "fs": regex
        }

        for field in self.sentence_fields:
            expr_data[field] = sentence_data[field]

        json_data["data"].append(expr_data)
        for field in self.sentence_fields:
            self.add_metadata(json_data, field,
                              sentence_data[field], idx)

        return True

    def generate(self, depth=2, n=200,
                 seed=None, sample_count=50,
                 name="propositional_logic", **kwargs):

        if seed is None:
            seed = int(time.time())

        random.seed(seed)
        depth_dict = sentence_generator.generate(
            self.grammar, depth=depth, n=n)

        data = []
        json_data = {}
        self._update_info(json_data, name, seed, depth,
                          n, sample_count)
        self._update_metadata(json_data)
        json_data["data"] = data
        self._update_field_idxs(json_data)

        json_data, total_count = self.populate_dataset(
            json_data, depth_dict, sample_count)

        return json_data, total_count

class RegexFilteredDataset(RegexDataset):

    def __init__(self, grammar, constant_vocabulary,
                 filter_field="depth"):

        super(RegexFilteredDataset, self).__init__(grammar,
                                                   constant_vocabulary,
                                                   filter_field)

    def _create_filtered_dataset(self, depth_dict):

        filtered_dict = {}
        for d, sentences in depth_dict.items():

            for sentence in sentences:

                sentence_data = {
                    "depth": d
                }
                op_list = filtered_dict.setdefault(sentence_data[self.filter_field], [])
                op_list.append((sentence_data, sentence))

        return filtered_dict

    def populate_dataset(self, json_data, depth_dict, sample_count):

        filtered_dict = self._create_filtered_dataset(depth_dict)
        max_filter = max(filtered_dict.keys())
        total_count = 0
        regex_set = set()
        progress_bar1 = tqdm.tqdm(total=len(filtered_dict), unit=" items",
                                 leave=False,
                                  position=0,
                                  desc="Generating Regular Expressions")
        for filter_key in sorted(filtered_dict.keys()):

            # print("Processing %s=%s (max: %s)" % (self.filter_field,
            #                                       filter_key,
            #                                       max_filter))
            data = filtered_dict[filter_key]

            data_subset = random.sample(data, min(len(data), sample_count))
            total_count += len(data_subset)
            futures = {}
            process_pool = concurrent.futures.ProcessPoolExecutor()
            for sentence_data, sentence in data_subset:
                future = process_pool.submit(self.process_sentence, *(sentence,))
                futures[future] = (sentence_data, sentence)

            progress_bar2 = tqdm.tqdm(total=len(futures), unit=" regex",
                                     leave=False, desc="Sampling",
                                      position=1)
            for future in concurrent.futures.as_completed(futures):
                progress_bar2.update(1)
                sentence_data, sentence = futures[future]
                sentence_data.update(future.result())
                self._generate_regex(sentence, sentence_data, json_data,
                                     regex_set)

            progress_bar2.close()
            progress_bar1.update(1)

        progress_bar1.close()
        return json_data, total_count

if __name__ == "__main__":

    depth = 30
    n = 200
    sample_count = 50

    # Regex params
    filter_field = "depth"
    grammar = RegexDataset.REGEX_GRAMMAR
    output_dir = "/tmp/results/regex"
    constant_vocabulary = ConstantVocabulary(num_variables=2, prefix="", start=0)

    dataset = RegexFilteredDataset(grammar,
                                   constant_vocabulary,
                                   filter_field=filter_field)

    # Output dir
    import os
    output_dir = os.path.abspath(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        assert os.path.isdir(output_dir)
    dataset_filepath = "%s/dataset.json" % (output_dir)

    json_data, total_count = dataset.generate(name="dataset", depth=depth,
                                 n=n,
                                 sample_count=sample_count)
    print("Dataset: | Size: %6d | Path: %s" %  (
        total_count,
        dataset_filepath))

    fh = open(dataset_filepath, "w")
    json.dump(json_data, fh, indent=4, ensure_ascii=False)
    fh.close()