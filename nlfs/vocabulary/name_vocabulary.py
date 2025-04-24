
import random
import itertools
from nltk.sem.logic import Variable
from faker.providers.person.en import Provider

class NameVocabulary:
    def __init__(self, grammar_string="V",num_variables=None):

        self.num_variables = num_variables
        self.grammar_string = grammar_string
        self.banned_substr = ["All", "Any", "all", "any", "'",'_','-',' ','v','or','and','V']
        self.variables = [x.lower() for x in list(set(Provider.first_names)) if not any(bsub in x for
                                                                                        bsub in self.banned_substr)]
        self.selected_variables = []
        self.reset_free()

    def is_match(self, c):
        return c == self.grammar_string

    def generate_free(self):
        raise Exception("generate_free() should never be called in NameVocabulary!")

    def reset_free(self):
        if self.num_variables is not None:
            self.selected_variables = random.sample(self.variables,self.num_variables)

    def generate(self):
        if self.num_variables is None:
            return random.choice(self.variables)
        return random.choice(self.selected_variables)

    def generate_bindings(self, variables):
        bindings = {}
        for v in variables:
            if v.name.startswith(self.grammar_string):
                bindings[v] = Variable(self.generate())

        return bindings

    def get_info(self):

        return {"num_variables": self.num_variables}

if __name__ == "__main__":

    random.seed(1234)
    name_vocab = NameVocabulary(num_variables=5)
    for _ in range(25):
        print(name_vocab.generate())

