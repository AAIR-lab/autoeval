import random
import itertools
import os
import xml.etree.ElementTree as ET
from nltk.sem.logic import Variable
from nltk.sem.logic import read_logic


class VerbVocabulary:

    def __init__(self, verbnet_path:str,
                 num_predicates:int,
                 constant_vocabulary,
                 min_arity:int=1,
                 max_arity:int=2,
                 grammar_string:str="P"):

        self.num_predicates = num_predicates
        self.banned_components = [None, 'from', "'s", 'of', 'with']
        self.allowed_cases = ['Experiencer', 'Agent', 'Recipient']
        self.banned_substr = ["all","any","v","or","and",'V']

        self.constant_vocabulary = constant_vocabulary

        self.grammar_string = grammar_string
        self.min_arity = min_arity
        self.max_arity = max_arity
        self.free_count = None
        self.predicate_arity = {}

        try:
            self._parse_verbnet(verbnet_path)
        except:
            raise Exception("Failed parsing verbnet with path: %s"%(verbnet_path))

        self.reset_free()

    def _extract_verbs(self,location, verb_list):
        verb_list.extend([verb.get('name').lower() for members in location.findall('MEMBERS') for verb in members if
                          verb.get('name') not in verb_list and verb.get('name').isalnum() and
                          not any(bsub in verb.get('name') for bsub in self.banned_substr)])

    def _extract_usages(self,location, usage_list):
        for frame in location.find('FRAMES').findall('FRAME'):
            use_case = [component.get('value') for component in frame.find('SYNTAX') if component.get('value') not
                        in self.banned_components and ' ' not in component.get('value')]

            if use_case not in usage_list and all(case in self.allowed_cases for case in use_case):
                usage_list.append(use_case)

    def _parse_verbnet(self,file_path):
        self.verb_dict = {}
        for file in os.listdir(file_path):
            if file.endswith(".xml"):
                verb_names, verb_use_cases = self._parse_xml_verb_file(os.path.join(file_path, file))
                for current_case in verb_use_cases:
                    if len(current_case) not in self.verb_dict:
                        self.verb_dict[len(current_case)] = []
                    for current_name in verb_names:
                        self.verb_dict[len(current_case)].append((current_name, current_case))
    def _parse_xml_verb_file(self,path):
        root = ET.parse(path).getroot()
        verb_names = []
        usage_list = []
        self._extract_verbs(root, verb_names)
        self._extract_usages(root, usage_list)

        for sub_class in root.find('SUBCLASSES').findall("VNSUBCLASS"):
            self._extract_verbs(sub_class, verb_names)
            self._extract_usages(sub_class, usage_list)

        return verb_names, usage_list

    def is_match(self, c):
        return c == self.grammar_string

    def generate_free(self):
        return "%s%s" % (self.grammar_string,
                         next(self.free_count))

    def reset_free(self):
        self.free_count = itertools.count(1)
        self.predicate_arity.clear()
        self.constant_vocabulary.reset_free()
        for _ in range(self.num_predicates):
            arity = random.randint(self.min_arity, self.max_arity)
            self.predicate_arity[random.choice(self.verb_dict[arity])[0]] = arity

    def generate(self):

        predicate_name = random.choice(list(self.predicate_arity.keys()))

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

if __name__ == "__main__":
    from name_vocabulary import NameVocabulary
    name_vocab = NameVocabulary(num_variables=3)
    verb_vocab = VerbVocabulary('../../verbnet',5,name_vocab)
    random.seed(9869765)
    verb_vocab.reset_free()
    print(verb_vocab.generate())

    random.seed(9869765)
    verb_vocab.reset_free()
    print(verb_vocab.generate())