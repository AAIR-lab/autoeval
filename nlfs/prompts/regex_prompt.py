
class RegexTranslationPrompt:

    def __init__(self):

        pass

    def get_examples_string(self):
        regex_examples = [
            "[EXAMPLE 1]",
            "(1*)0*",
            "The regex matches strings that starts with any number (including none) of the digit '1', followed by any number (including none) of the digit '0'." ,
            "[EXAMPLE 2]",
            "(01*)",
            "The regex matches strings that begin with a '0' followed directly by any number (including none) of '1's.\n"
        ]
        return "\n".join(regex_examples)

    def get_task_string(self):

        string = """
[TASK]
Your task is to convert the regular expression appear after [REGEX], to a natural description that represents the regular expression. Only natural language terms are allowed to be used and do not copy the regular expression in your description. Your description should allow one to reconstruct the regular expression without having access to it, so make sure to use the correctly account for scoping. You may use terms verbatim as specified in the vocabulary below.
"""

        string += "\n"
        return string

    def get_representation_string(self):

        general_operators = [
            "you may use symbols from the vocabulary",
            "you can use *",
        ]

        return "\n".join(general_operators)


    def get_vocabulary_string(self):

        string = "[VOCABULARY]\n"
        string += self.get_representation_string()
        string += "\n\n"
        return string

    def generate(self, fs_datum, include_examples=False):

        regex = fs_datum["fs"]


        string = self.get_task_string()
        string += self.get_vocabulary_string()
        if include_examples:
            string += self.get_examples_string()

        string += "[FORMULA]\n"
        string += regex

        return string

class RegexInterpretationPrompt:

    def __init__(self):

        pass

    def get_examples_string(self):
        regex_examples = [
            "[EXAMPLE 1]",
            "The regex matches strings that starts with any number (including none) of the digit '1', followed by any number (including none) of the digit '0'." ,
            "(1*)0*",
            "[EXAMPLE 2]",
            "The regex matches strings that begin with a '0' followed directly by any number (including none) of '1's.",
            "(01*)\n"
        ]
        return "\n".join(regex_examples)

    def get_task_string(self):

        string = """
[TASK]
Your task is to interpret the natural language (NL) description of a regular expression and represent it as formal syntax using the vocabulary specified in the [VOCABULARY] block above. Only output the regular expression and no other text. The NL description appears immediately following the [NL DESCRIPTION] tag.
"""

        string += "\n"
        return string

    def get_representation_string(self):

        general_operators = [
            "Use * to represent zero or more duplications of the same expression",
            "Use ( and ) to represent parentheses",
        ]

        return "\n".join(general_operators)


    def get_vocabulary_string(self):

        string = "[VOCABULARY]\n"
        string += self.get_representation_string()
        string += "\n"
        return string

    def generate(self, nl, include_examples=False, fs_datum=None):

        string = self.get_vocabulary_string()
        string += self.get_task_string()
        if include_examples:
            string += self.get_examples_string()

        string += "[NL DESCRIPTION]\n"
        string += nl

        return string


if __name__ == "__main__":

    prompt = RegexTranslationPrompt()
    print(prompt.generate({"fs": "01*"},True))

    prompt = RegexInterpretationPrompt()
    print(prompt.generate("0 and 1",True))