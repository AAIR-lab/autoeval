
import random
import itertools
from nltk.sem.logic import Variable
class ConstantVocabulary:

    def __init__(self, num_variables=None,
                 prefix="p",
                 suffix="",
                 free_suffix="",
                 grammar_string="V",
                 start=1):

        self.num_variables = num_variables
        self.prefix = prefix
        self.suffix = suffix
        self.free_suffix = free_suffix
        self.grammar_string = grammar_string
        self.free_count = None
        self.start = start
        self.reset_free()

        if self.num_variables is not None:

            self.variables = ["%s%u%s" % (self.prefix, i, self.suffix)
                                for i in range(start, self.num_variables + start)]
        else:
            self.variables = None

    def is_match(self, c):

        return c == self.grammar_string

    def generate_free(self):

        if self.variables is None:

            return "%s%s%s" % (self.prefix,
                               next(self.free_count),
                               self.suffix)
        else:
            return "%s%s" % (self.grammar_string,
                             next(self.free_count))

    def reset_free(self):

        self.free_count = itertools.count(self.start)

    def generate(self):

        return random.choice(self.variables)

    def generate_bindings(self, variables):

        bindings = {}
        for v in variables:

            if v.name.startswith(self.grammar_string):
                bindings[v] = Variable(self.generate())

        return bindings

    def get_info(self):

        return {"num_variables": self.num_variables}