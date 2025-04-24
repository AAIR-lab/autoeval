import argparse
import os
import shutil
import json
import sys

from nlfs.dataset.propositional_dataset import LogicDataset
from nlfs.dataset.propositional_dataset import FreeVariableSubstitutor
from nlfs.dataset.propositional_dataset import LogicFilteredDataset
from nlfs.vocabulary.constant_vocabulary import ConstantVocabulary
from nlfs.vocabulary.predicate_vocabulary import PredicateVocabulary
from nlfs.vocabulary.dummy_vocabulary import DummyVocabulary
from nlfs.vocabulary.name_vocabulary import NameVocabulary
from nlfs.vocabulary.verb_vocabulary import VerbVocabulary

from nlfs.dataset.regex_dataset import RegexDataset
from nlfs.dataset.regex_dataset import RegexFilteredDataset


def setup_directory(directory, clean=False):

    assert not os.path.exists(directory) or os.path.isdir(directory)
    if os.path.exists(directory):

        assert os.path.isdir(directory)
        if clean:
            shutil.rmtree(directory)
            os.makedirs(directory)
    else:
        os.makedirs(directory)


def get_propositional_args(args):

    constant_vocab = ConstantVocabulary(
        num_variables=args.num_propositions)
    predicate_vocab = DummyVocabulary()

    if args.dataset_type == "ksat":
        grammar = LogicDataset.KSAT_GRAMMAR
    else:
        assert args.dataset_type == "plogic"
        grammar = LogicDataset.PROPOSITIONAL_GRAMMAR

    return grammar, constant_vocab, predicate_vocab


def get_fol_args(args):

    free_variable_vocab = ConstantVocabulary(prefix="x", suffix=".")
    object_vocab = ConstantVocabulary(num_variables=args.num_objects)
    predicate_vocab = PredicateVocabulary(
        args.num_predicates,
        object_vocab,
        min_arity=args.min_predicate_arity,
        max_arity=args.max_predicate_arity)
    grammar = LogicDataset.PRENEX_NORMAL_FORM_GRAMMAR

    return grammar, free_variable_vocab, predicate_vocab


def get_fol_human_args(args):

    free_variable_vocab = ConstantVocabulary(prefix="x", suffix=".")
    object_vocab = NameVocabulary(num_variables=args.num_objects)
    predicate_vocab = VerbVocabulary(
        args.verbnet_path,
        args.num_predicates,
        object_vocab,
        min_arity=args.min_predicate_arity,
        max_arity=args.max_predicate_arity)
    grammar = LogicDataset.PRENEX_NORMAL_FORM_GRAMMAR

    return grammar, free_variable_vocab, predicate_vocab

def get_logic_dataset(args):

    if args.dataset_type in ["ksat", "plogic"]:

        grammar, constant_vocab, predicate_vocab = \
            get_propositional_args(args)
    elif args.dataset_type == "fol":
        grammar, constant_vocab, predicate_vocab = \
            get_fol_args(args)
    else:
        assert args.dataset_type == "fol_human"
        grammar, constant_vocab, predicate_vocab = \
            get_fol_human_args(args)

    free_variable_substitutor = FreeVariableSubstitutor(
        max_substitutions=args.max_free_variables,
        substitution_threshold=args.free_variable_prob)
    ds = LogicFilteredDataset(
        grammar,
        constant_vocab, predicate_vocab,
        free_variable_substitutor,
        filter_field=args.filter_field)

    return ds


def get_regex_dataset(args):

    alphabet_vocab = ConstantVocabulary(num_variables=args.alphabet_size,
                                        prefix="", start=0)
    ds = RegexFilteredDataset(
        RegexDataset.REGEX_GRAMMAR,
        alphabet_vocab,
        filter_field=args.filter_field)

    return ds


if __name__ == "__main__":

    try:
        pythonhashseed = int(os.getenv("PYTHONHASHSEED"))
    except Exception:

        print("PYTHONHASHSEED not set or is not a valid integer")
        sys.exit(1)

    parser = argparse.ArgumentParser()

    parser.add_argument("--base-dir", type=str, default=None,
                        required=True)
    parser.add_argument("--clean", default=False, action="store_true")

    parser.add_argument("--dataset-type", type=str, default=None,
                        required=True,
                        choices=["ksat", "plogic", "fol", "regex", "fol_human"])
    parser.add_argument("--filter-field", type=str, default=None,
                        required=True)

    parser.add_argument("--num-objects", type=int, default=12)
    parser.add_argument("--num-propositions", type=int, default=12)
    parser.add_argument("--num-predicates", type=int, default=8)
    parser.add_argument("--min-predicate-arity", type=int,
                        default=1)
    parser.add_argument("--max-predicate-arity", type=int,
                        default=2)

    parser.add_argument("--max-free-variables", type=int,
                        default=float("inf"))
    parser.add_argument("--free-variable-prob", type=float,
                        default=0.25)

    parser.add_argument("--alphabet-size", type=int, default=2)

    parser.add_argument("--depth", type=int, default=40)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample-count", type=int, default=50)
    parser.add_argument("--verbnet-path", type=str, default= "dependencies/verbnet")

    args = parser.parse_args()

    if args.dataset_type == "regex":
        assert args.filter_field == "depth"

    args.base_dir = os.path.abspath(args.base_dir)
    setup_directory(args.base_dir, clean=args.clean)

    if args.dataset_type in ["ksat", "plogic", "fol", "fol_human"]:

        dataset = get_logic_dataset(args)
    else:

        assert args.dataset_type == "regex"
        dataset = get_regex_dataset(args)

    json_data, total_count = dataset.generate(
        name="dataset",
        seed=args.seed,
        depth=args.depth,
        n=args.n,
        sample_count = args.sample_count)

    dataset_filepath = "%s/dataset.json" % (args.base_dir)
    with open(dataset_filepath, "w") as fh:
        json.dump(json_data, fh, indent=4, ensure_ascii=False)

    print("Dataset: | Size: %6d | Path: %s" %  (
        total_count,
        dataset_filepath))