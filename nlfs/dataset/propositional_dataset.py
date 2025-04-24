
import nltk
from nlfs.dataset.dataset import Dataset
import time
import random
import nltk
from nlfs.grammar import sentence_generator
from nlfs.vocabulary.constant_vocabulary import ConstantVocabulary
from nlfs.vocabulary.dummy_vocabulary import DummyVocabulary
from nlfs.vocabulary.predicate_vocabulary import PredicateVocabulary
from nlfs.vocabulary.name_vocabulary import NameVocabulary
from nlfs.vocabulary.verb_vocabulary import VerbVocabulary
from nltk.sem.logic import read_logic
import json

from nlfs.verifier import logic
import tqdm

class LogicExpressionStats:

    def __init__(self):

        self.is_first_order = False
        self.predicate_arity_dict = {}
        self.objects = set()
        self.free_variables = set()
        self.predicates = set()

    def _add_predicate(self, predicate_name, arity):

        args_string = ",".join(["?p%u" % (i) for i in range(arity)])
        predicate_string = "%s(%s)" % (predicate_name,
                                       args_string)
        self.predicates.add(predicate_string)
    def parse(self, expression):

        self.objects.update([v.name for v in expression.variables()])
        if isinstance(expression, nltk.sem.logic.IndividualVariableExpression):
            return expression
        elif isinstance(expression, nltk.sem.logic.ApplicationExpression):
            assert len(expression.predicates()) == 1
            predicate_name = expression.pred.variable.name
            arity = len(expression.args)
            self._add_predicate(predicate_name, arity)
            self.predicate_arity_dict[predicate_name] = arity
            self.is_first_order = True
            for current_const in expression.args:
                self.objects.add(str(current_const))
        elif isinstance(expression, nltk.sem.logic.ConstantExpression):
            self.objects.add(str(expression))
        else:
            if isinstance(expression, nltk.sem.logic.AllExpression) \
                or isinstance(expression, nltk.sem.logic.ExistsExpression):

                self.free_variables.add(expression.variable.name)
                self.is_first_order = True

            # All other kinds of expressions (And, Or etc)
            expression.visit_structured(self.parse,
                                        expression.__class__)
        return expression

    def finalize(self):

        self.objects.difference_update(self.free_variables)

class FreeVariableSubstitutor:

    def __init__(self, max_substitutions=float("inf"), substitution_threshold=0.2):

        import collections
        self.free_variables = collections.deque()
        self.max_substitutions = max_substitutions
        self.substitution_threshold = substitution_threshold

    def push_free_variables(self, expression):

        if isinstance(expression, nltk.sem.logic.AllExpression) \
            or isinstance(expression, nltk.sem.logic.ExistsExpression):

            self.free_variables.appendleft(expression.variable)

    def pop_free_variables(self, expression):

        if isinstance(expression, nltk.sem.logic.AllExpression) \
            or isinstance(expression, nltk.sem.logic.ExistsExpression):

            v = self.free_variables.popleft()
            assert expression.variable == v

    def _substitute_bindings(self, expression):
        if len(self.free_variables) > 0:
            arguments = str(expression)
            open_para = arguments.find('(')
            predicate_name = arguments[:open_para]
            arguments = arguments[open_para + 1:-1].split(',')
            new_arguments = ""
            substitution_count = 0

            for c_arg in arguments:
                if random.random() < self.substitution_threshold and substitution_count < self.max_substitutions:
                    new_arguments += str(random.choice(self.free_variables)) + ","
                else:
                    new_arguments += c_arg + ","

            return read_logic("%s(%s)" % (predicate_name, new_arguments[:-1]))[0]

        return expression

    def substitute_free_variables(self, expression):

        self.push_free_variables(expression)
        if isinstance(expression, nltk.sem.logic.ApplicationExpression):

            expression = self._substitute_bindings(expression)
        elif not isinstance(expression, nltk.sem.logic.IndividualVariableExpression):
            expression = expression.visit_structured(self.substitute_free_variables,
                                                     expression.__class__)

        self.pop_free_variables(expression)
        return expression

class LogicDataset(Dataset):

    PROPOSITIONAL_GRAMMAR = [
        "s -> '(' s 'and' s ')'",
        "s -> '(' s 'or' s ')'",
        "s -> '(' 'not' s ')'",
        "s -> 'not' 'V'",
        "s -> 'V'",
    ]

    KSAT_GRAMMAR = [
        "s -> s 'and' s",
        "s -> '(' p 'or' p 'or' p ')'",
        "p -> 'not' 'V'",
        "p -> 'V'",
    ]

    PRENEX_NORMAL_FORM_GRAMMAR = [
        "s -> q",
        "q -> '(' 'forall' 'V' q ')'",
        "q -> '(' 'exists' 'V' q ')'",
        "q -> f",
        "f -> '(' f 'and' f ')'",
        "f -> '(' f 'or' f ')'",
        "f -> '(' 'not' f ')'",
        "f -> 'not' 'P'",
        "f -> 'P'",
    ]

    def __init__(self, grammar, constant_vocabulary,
                 predicate_vocabulary,
                 free_variable_substitutor,
                 filter_field="depth"):

        self.constant_vocabulary = constant_vocabulary
        self.grammar = nltk.grammar.CFG.fromstring(grammar)
        self.predicate_vocabulary = predicate_vocabulary
        self.free_variable_substitutor = free_variable_substitutor
        self.filter_field = filter_field
        self.sentence_fields = ["depth", "num_operators", "num_propositions",
                           "num_quantifiers", "num_predicates"]

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
            "predicate_info": self.predicate_vocabulary.get_info(),
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
        self.constant_vocabulary.reset_free()
        self.predicate_vocabulary.reset_free()

        num_operators = 0
        num_propositions = 0
        num_quantifiers = 0
        num_predicates = 0
        for i, s in enumerate(sentence):

            if s in ["not", "or", "and"]:

                is_next_constant = self.constant_vocabulary.is_match(
                    sentence[i + 1])
                is_next_predicate = self.predicate_vocabulary.is_match(
                    sentence[i + 1])
                if s != "not" or \
                    (not is_next_constant and not is_next_predicate):
                    num_operators += 1
            elif s in ["forall", "exists"]:

                assert self.constant_vocabulary.is_match(sentence[i + 1])
                num_quantifiers += 1
            elif self.constant_vocabulary.is_match(sentence[i]):
                sentence[i] = self.constant_vocabulary.generate_free()
                num_propositions += 1
            elif self.predicate_vocabulary.is_match(sentence[i]):
                sentence[i] = self.predicate_vocabulary.generate_free()
                num_predicates += 1
            else:
                assert s in ["(", ")"]

        return {
            "num_operators": num_operators,
            "num_propositions": num_propositions,
            "num_quantifiers": num_quantifiers,
            "num_predicates": num_predicates }

    def add_metadata(self, json_data, metadata_key, value_key, idx):

        data_list = json_data[metadata_key].setdefault(value_key, [])
        data_list.append(idx)

        metadata = json_data["metadata"][metadata_key]
        metadata["total"] = metadata["total"] + 1
        metadata[value_key] = metadata.get(value_key, 0) + 1

    def substitute_bindings(self, expr, sentence_data, json_data, expr_set):
        self.predicate_vocabulary.reset_free()

        variables = expr.variables()
        bindings = {}
        bindings.update(
            self.constant_vocabulary.generate_bindings(variables))
        bindings.update(
            self.predicate_vocabulary.generate_bindings(variables))

        ground_expr = expr.substitute_bindings(bindings)
        ground_expr = self.free_variable_substitutor.substitute_free_variables(
            ground_expr)

        if ground_expr in expr_set:
            return False
        else:
            expr_set.add(ground_expr)

        idx = len(json_data["data"])
        expr_data = {
            "idx": idx,
            "fs": logic.expr_to_str(ground_expr)
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
                 max_tries_per_sample=100,
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

        json_data, total_count = self.populate_dataset(json_data,
                                                       depth_dict,
                                                       sample_count,
                                                       max_tries_per_sample)
        return json_data, total_count

class LogicFilteredDataset(LogicDataset):

    def __init__(self, grammar, constant_vocabulary, predicate_vocabulary,
                 free_variable_substitutor, filter_field="num_operators"):

        super(LogicFilteredDataset, self).__init__(grammar,
                                                   constant_vocabulary,
                                                   predicate_vocabulary,
                                                   free_variable_substitutor,
                                                   filter_field)

    def _create_filtered_dataset(self, depth_dict):

        filtered_dict = {}
        for d, sentences in depth_dict.items():

            for sentence in sentences:

                sentence_data = {
                    "depth": d
                }
                sentence_data.update(self.process_sentence(sentence))
                op_list = filtered_dict.setdefault(sentence_data[self.filter_field], [])
                op_list.append((sentence_data, sentence))

        return filtered_dict
    def populate_dataset(self, json_data, depth_dict, sample_count,
                         max_tries_per_sample):

        expr_set = set()
        filtered_dict = self._create_filtered_dataset(depth_dict)
        progress_bar1 = tqdm.tqdm(total=len(filtered_dict), unit=" items",
                                 leave=False,
                                  position=0,
                                  desc="Generating Logic Expressions")
        for filter_key in sorted(filtered_dict.keys()):

            # print("Processing %s=%s" % (self.filter_field, filter_key))
            data = filtered_dict[filter_key]

            progress_bar2 = tqdm.tqdm(total=sample_count, unit=" regex",
                                     leave=False, desc="Sampling",
                                      position=1)
            for _ in range(sample_count):
                for _ in range(max_tries_per_sample):

                    sentence_data, sentence = random.choice(data)
                    expr = read_logic(" ".join(sentence))[0]

                    success = self.substitute_bindings(expr, sentence_data,
                                                       json_data, expr_set)

                    if success:
                        break

                progress_bar2.update(1)

            progress_bar2.close()
            progress_bar1.update(1)

        progress_bar1.close()
        return json_data, len(filtered_dict) * sample_count

def get_ksat_args():

    output_dir = "/tmp/results/ksat"
    constant_vocabulary = ConstantVocabulary(num_variables=3)
    predicate_vocabulary = DummyVocabulary()
    grammar = LogicDataset.KSAT_GRAMMAR
    filter_field = "num_operators"

    return output_dir, constant_vocabulary, predicate_vocabulary, grammar, \
        filter_field

def get_propositional_logic_args():

    output_dir = "/tmp/results/plogic"
    constant_vocabulary = ConstantVocabulary(num_variables=3)
    predicate_vocabulary = DummyVocabulary()
    grammar = LogicDataset.PROPOSITIONAL_GRAMMAR
    filter_field = "num_operators"

    return output_dir, constant_vocabulary, predicate_vocabulary, grammar, \
        filter_field

def get_fol_args():

    # First-order Logic
    output_dir = "/tmp/results/fol"
    constant_vocabulary = ConstantVocabulary(prefix="x", suffix=".")
    object_vocabulary = ConstantVocabulary(num_variables=4)
    predicate_vocabulary = PredicateVocabulary(3, object_vocabulary)
    grammar = LogicDataset.PRENEX_NORMAL_FORM_GRAMMAR
    filter_field = "num_operators"

    return output_dir, constant_vocabulary, predicate_vocabulary, grammar, \
        filter_field

def get_fol_human_args():

    # First-order Logic
    output_dir = "/tmp/results/fol_human"
    constant_vocabulary = ConstantVocabulary(prefix="x", suffix=".")
    object_vocabulary = NameVocabulary(num_variables=4)
    predicate_vocabulary = VerbVocabularyVocabulary('../../verbnet',3, object_vocabulary)
    grammar = LogicDataset.PRENEX_NORMAL_FORM_GRAMMAR
    filter_field = "num_operators"

    return output_dir, constant_vocabulary, predicate_vocabulary, grammar, \
        filter_field

if __name__ == "__main__":

    dataset_type = "fol"
    depth = 40
    n = 200
    sample_count = 50
    free_variable_substitutor = FreeVariableSubstitutor()

    if dataset_type == "ksat":
        (output_dir, constant_vocabulary,
         predicate_vocabulary, grammar, filter_field) = \
            get_ksat_args()
    elif dataset_type == "propositional":

        (output_dir, constant_vocabulary,
         predicate_vocabulary, grammar, filter_field) = \
            get_propositional_logic_args()
    elif dataset_type == "fol":
        (output_dir, constant_vocabulary,
         predicate_vocabulary, grammar, filter_field) = \
            get_fol_args()
    else:
        assert dataset_type == "fol_human"
        (output_dir, constant_vocabulary,
         predicate_vocabulary, grammar, filter_field) = \
            get_fol_human_args()


    dataset = LogicFilteredDataset(grammar,
                                   constant_vocabulary, predicate_vocabulary,
                                   free_variable_substitutor,
                                   filter_field=filter_field)

    # Output dir
    import os
    output_dir = os.path.abspath(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        assert os.path.isdir(output_dir)
    dataset_filepath = "%s/dataset.json" % (output_dir)

    json_data, total_count = dataset.generate(name="dataset", depth=depth, n=n,
                                 sample_count=sample_count)
    fh = open(dataset_filepath, "w")
    json.dump(json_data, fh, indent=4, ensure_ascii=False)
    fh.close()

    print("Dataset: | Size: %6d | Path: %s" %  (
        total_count,
        dataset_filepath))