
import pathlib
import json
import os
print(os.path.abspath(os.curdir))
os.chdir('..')
print(os.path.abspath(os.curdir))

from nlfs.dataset.propositional_dataset import LogicExpressionStats
from nltk.sem.logic import read_logic
from nlfs.verifier import logic
class Prompt:

    PROMPTS_FILE = pathlib.Path(__file__).parent.joinpath("prompts.json")

    def __init__(self, filepath_or_key_list):

        assert not isinstance(filepath_or_key_list, str)


        prompts_dict = json.load(open(Prompt.PROMPTS_FILE, "r"))
        for key in filepath_or_key_list[:-1]:
            prompts_dict = prompts_dict[key]

        self.prompt = prompts_dict[filepath_or_key_list[-1]]

    def generate(self, *args):

        return self.prompt % (args)

class LogicTranslationPrompt:

    def __init__(self):

        pass

    def read_logic(self, string):

        if isinstance(string, str):
            return read_logic(string)[0]
        else:
            return string

    def get_task_string(self, les):

        if les.is_first_order:
            logic_type = "first-order logic"
        else:
            logic_type = "propositional logic"

        string = """
[TASK]
Your task is to convert a %s formula, appearing after [FORMULA], to a natural description that represents the formula. Only natural language terms are allowed to be used and do not copy the formula in your description. Your description should allow one to reconstruct the formula without having access to it, so make sure to use the correct names in your description. Explicitly describe the predicates. You may use terms verbatim as specified in the vocabulary below.
""" % (logic_type)

        string += "\n"
        return string

    def get_representation_string(self, les):

        general_operators = [
            "v represents disjunction",
            "∧ represents conjunction",
            "¬ represents negation",
            "→ represents implication",
            "( and ) represent parentheses",
            "propositions can be used verbatim",
        ]
        fol_operators = [
            "predicates can be used verbatim",
            "∀ x1 <x2> ... <xn>. represents universal quantification with x1... representing free variables",
            "∃ x1 <x2> ... <xn>. represents existential quantification with x1... representing free variables",
        ]

        if les.is_first_order:
            return "\n".join(general_operators + fol_operators)
        else:
            return "\n".join(general_operators)

    def get_examples_string(self, les):
        proposition_examples = [
            "[EXAMPLE 1]",
            "(¬p2 v p1 v ¬p2)",
            "Disjunctive predicate logic expression consisting of three components: the negation of a proposition labeled p2, the proposition p1, and again the negation of p2.",
            "[EXAMPLE 2]",
            "(¬¬p2 ∧ ¬(p3 v p1))",
            "The expression asserts that p2 is not false while both p3 and p1 are not true.\n"
        ]
        fol_examples = [
            "[EXAMPLE 1]",
            "∀ x1.∃ x2.((pred1(x1, p1) v ¬pred2(x1, x2)) ∧ pred3(p1))",
            "For every entire x1, there exists another entity x2 such that either the predicate pred1 holds between x1 and a constant p1, or the predicate pred2 does not hold between x1 and x2. Additionally, the predicate pred3 always holds for p1.",
            "[EXAMPLE 2]",
            "∃ x1.¬(pred1(p1, x1) ∧ (¬pred2(p2, x2) ∧ ¬pred3(p3,p3)))",
            "There exists an entity x1 such that it is not the case that all the following conditions are true simultaneously: the predicate pred1 holds between the constant p1 and x1, the predicate pred2 holds between the constant p2 and x1, and the predicate pred3 does not hold between p3 and itself.\n"
        ]

        if les.is_first_order:
            "\n".join(fol_examples)
        return "\n".join(proposition_examples)

    def get_provided_example_string(self, nl, fs):
        example = [
            "[EXAMPLE 1]",
            fs,
            nl,
            ""
        ]
        return "\n".join(example)

    def get_vocabulary_string(self, les):

        string = "[VOCABULARY]\n"
        string += self.get_representation_string(les)
        string += "\n"
        string += self.get_object_string(les)
        string += self.get_predicate_string(les)
        string += self.get_free_variable_string(les)
        string += "\n"
        return string

    def get_object_string(self, les):

        if les.is_first_order:
            object_str = "objects"
        else:
            object_str = "propositions"

        string = "The %s are: %s\n" % (
            object_str,
            ", ".join([obj for obj in les.objects if obj not in les.free_variables]))

        return string

    def get_predicate_string(self, les):

        string = ""
        if les.is_first_order and len(les.predicates) > 0:
            string = "The parameterized predicates are: %s\n" % (
                ", ".join(sorted(les.predicates)))

        return string

    def get_free_variable_string(self, les):

        string = ""
        if len(les.free_variables) > 0:

            string = "The free variables are: %s\n" % (
                ", ".join(sorted(les.free_variables)))

        return string

    def generate(self, fs_datum, include_examples=False):

        expression = logic.str_to_expr(fs_datum["fs"])

        les = LogicExpressionStats()
        les.parse(expression)

        string = self.get_task_string(les)
        if include_examples:
            if "example_nl" in fs_datum and "example_fs" in fs_datum:
                string += self.get_provided_example_string(fs_datum["example_nl"],fs_datum["example_fs"])
            else:
                string += self.get_examples_string(les)

        string += self.get_vocabulary_string(les)

        string += "[FORMULA]\n"
        string += logic.expr_to_str(expression)

        return string

class LogicInterpretationPrompt:

    def __init__(self, is_first_order=False):

        self.is_first_order = is_first_order

    def read_logic(self, string):

        if isinstance(string, str):
            return read_logic(string)[0]
        else:
            return string

    def get_provided_example_string(self, nl, fs):
        example = [
            "[EXAMPLE 1]",
            nl,
            fs,
            ""
        ]
        return "\n".join(example)

    def get_examples_string(self):
        proposition_examples = [
            "[EXAMPLE 1]",
            "Disjunctive predicate logic expression consisting of three components: the negation of a proposition labeled p2, the proposition p1, and again the negation of p2.",
            "(¬p2 v p1 v ¬p2)",
            "[EXAMPLE 2]",
            "The expression asserts that p2 is not false while both p3 and p1 are not true.",
            "(¬¬p2 ∧ ¬(p3 v p1))\n"
        ]
        fol_examples = [
            "[EXAMPLE 1]",
            "For every entire x1, there exists another entity x2 such that either the predicate pred1 holds between x1 and a constant p1, or the predicate pred2 does not hold between x1 and x2. Additionally, the predicate pred3 always holds for p1.",
            "∀ x1.∃ x2.((pred1(x1, p1) v ¬pred2(x1, x2)) ∧ pred3(p1))",
            "[EXAMPLE 2]",
            "There exists an entity x1 such that it is not the case that all the following conditions are true simultaneously: the predicate pred1 holds between the constant p1 and x1, the predicate pred2 holds between the constant p2 and x1, and the predicate pred3 does not hold between p3 and itself.",
            "∃ x1.¬(pred1(p1, x1) ∧ (¬pred2(p2, x2) ∧ ¬pred3(p3,p3)))\n"
        ]

        if self.is_first_order:
            "\n".join(fol_examples)
        return "\n".join(proposition_examples)

    def get_task_string(self):

        if self.is_first_order:
            logic_type = "first-order logic"
        else:
            logic_type = "propositional logic"

        string = """
[TASK]
Your task is to interpret the natural language (NL) description of a %s formula and represent it as formal syntax using the vocabulary specified in the [VOCABULARY] block above. Only output the formula and no other text. The NL description appears immediately following the [NL DESCRIPTION] tag.
""" % (logic_type)

        string += "\n"
        return string

    def get_representation_string(self):

        general_operators = [
            "Use v to represent disjunction",
            "Use ∧ to represent conjunction",
            "Use ¬ to represent negation",
            "Use ( and ) to represent parentheses",
        ]
        fol_operators = [
            "Use ∀ <free_variable_list> to represent universal quantification",
            "Use ∃ <free_variable_list> to represent existential quantification",
            "The <free_variable_list> consists of a sequence of space separate free variables with the last variable immediately followed by a period. Examples: (1) all x1 x2. (2) exists x4.",
            "Use <predicate>(<parameter_list>) to represent predicates (Names and parameters are provided in the description)"
        ]

        if self.is_first_order:
            return "\n".join(general_operators + fol_operators)
        else:
            return "\n".join(general_operators)


    def get_object_string(self, les):

        if les.is_first_order:
            object_str = "objects"
        else:
            object_str = "propositions"

        string = "The %s are: %s\n" % (
            object_str,
            ", ".join([obj for obj in les.objects if obj not in les.free_variables]))

        return string

    def get_predicate_string(self, les):

        string = ""
        if les.is_first_order and len(les.predicates) > 0:
            string = "The parameterized predicates are: %s\n" % (
                ", ".join(sorted(les.predicates)))

        return string

    def get_free_variable_string(self, les):

        string = ""
        if len(les.free_variables) > 0:

            string = "The free variables are: %s\n" % (
                ", ".join(sorted(les.free_variables)))

        return string

    def get_vocabulary_string(self, les=None):

        string = "[VOCABULARY]\n"
        string += self.get_representation_string()
        string += "\n"
        if les is not None:
            string += self.get_object_string(les)
            string += self.get_predicate_string(les)
        string += "\n"
        return string
    def generate(self, nl, include_examples=False, fs_datum=None):

        les = None
        if fs_datum is not None:
            les = LogicExpressionStats()
            les.parse(logic.str_to_expr(fs_datum["fs"]))

        string = self.get_vocabulary_string(les)
        string += self.get_task_string()
        if include_examples:
            if fs_datum is not None:
                if "example_nl" in fs_datum and "example_fs" in fs_datum:
                    string += self.get_provided_example_string(fs_datum["example_nl"],fs_datum["example_fs"])
                else:
                    string += self.get_examples_string()
            else:
                string += self.get_examples_string()

        string += "[NL DESCRIPTION]\n"
        string += nl

        return string


if __name__ == "__main__":

    prompt = LogicTranslationPrompt()
    print(prompt.generate({"fs": "∃ x1.(twang(x1,x1) ∧ ¬(¬rant(dink,dink) ∧ (¬twang(dondre,x1) v ¬twang(dink,dondre) v (twang(dondre,florencio) ∧ ¬(pant(x1,x1) ∧ (twang(captain,captain) v (twang(dondre,florencio) ∧ ¬(¬enquire(dondre,dink) v ¬(twang(x1,dink) ∧ rant(dondre,florencio) ∧ twang(captain,dondre) ∧ rant(florencio,x1) ∧ ¬(enquire(dondre,captain) v (¬rant(x1,dink) ∧ enquire(dink,florencio) ∧ ¬twang(x1,x1) ∧ ¬enquire(florencio,captain) ∧ (rant(x1,x1) v (rant(x1,captain) ∧ ¬¬((¬rant(dondre,x1) v rant(captain,x1) v rant(florencio,florencio) v (enquire(florencio,dondre) ∧ (¬twang(captain,x1) v ¬twang(captain,captain) v pant(dondre,captain)))) ∧ (twang(captain,captain) v ¬twang(dink,captain))))))))))))))))"},True))

    prompt = LogicInterpretationPrompt()
    print(prompt.generate("p1 conjunction with p2",False,{"fs": "∃ x1.(twang(x1,x1) ∧ ¬(¬rant(dink,dink) ∧ (¬twang(dondre,x1) v ¬twang(dink,dondre) v (twang(dondre,florencio) ∧ ¬(pant(x1,x1) ∧ (twang(captain,captain) v (twang(dondre,florencio) ∧ ¬(¬enquire(dondre,dink) v ¬(twang(x1,dink) ∧ rant(dondre,florencio) ∧ twang(captain,dondre) ∧ rant(florencio,x1) ∧ ¬(enquire(dondre,captain) v (¬rant(x1,dink) ∧ enquire(dink,florencio) ∧ ¬twang(x1,x1) ∧ ¬enquire(florencio,captain) ∧ (rant(x1,x1) v (rant(x1,captain) ∧ ¬¬((¬rant(dondre,x1) v rant(captain,x1) v rant(florencio,florencio) v (enquire(florencio,dondre) ∧ (¬twang(captain,x1) v ¬twang(captain,captain) v pant(dondre,captain)))) ∧ (twang(captain,captain) v ¬twang(dink,captain))))))))))))))))"}))
