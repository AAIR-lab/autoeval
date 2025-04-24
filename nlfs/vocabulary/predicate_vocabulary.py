
import random
import itertools
from nltk.sem.logic import Variable
from nltk.sem.logic import read_logic

class PredicateVocabulary:

    def __init__(self, num_predicates,
                 constant_vocabulary,
                 min_arity=1,
                 max_arity=2,
                 prefix="pred",
                 suffix="",
                 grammar_string="P"):

        self.num_predicates = num_predicates

        self.constant_vocabulary = constant_vocabulary
        assert isinstance(self.constant_vocabulary.num_variables, int)
        assert self.constant_vocabulary.num_variables > 0

        self.prefix = prefix
        self.suffix = suffix
        self.grammar_string = grammar_string
        self.free_count = None
        self.min_arity = min_arity
        self.max_arity = max_arity

        self.predicate_arity = {}
        if self.num_predicates is not None:

            self.predicates = ["%s%u%s" % (self.prefix, i, self.suffix)
                                for i in range(1, self.num_predicates + 1)]
            self.reset_arities()
        else:
            assert False

        self.reset_free()

    def is_match(self, c):

        return c == self.grammar_string

    def generate_free(self):

        return "%s%s" % (self.grammar_string,
                         next(self.free_count))

    def reset_arities(self):

        for predicate_name in self.predicates:
            arity = random.randint(self.min_arity, self.max_arity)
            self.predicate_arity[predicate_name] = arity

    def reset_free(self):

        self.free_count = itertools.count(1)
        self.reset_arities()

    def generate(self):

        predicate_name = random.choice(self.predicates)

        arity = self.predicate_arity[predicate_name]
        constants = []
        for _ in range(arity):
            constants.append(self.constant_vocabulary.generate())
        constants = ", ".join(constants)

        predicate_string = "%s(%s)" % (predicate_name, constants)
        predicate = read_logic(predicate_string)[0]
        return predicate

    def generate_bindings(self, variables):

        bindings = {}
        for v in variables:

            if v.name.startswith(self.grammar_string):

                bindings[v] = self.generate()

        return bindings

    def get_info(self):

        return {"num_predicates": self.num_predicates,
                "constants": self.constant_vocabulary.get_info(),
                "min_arity": self.min_arity,
                "max_arity": self.max_arity}