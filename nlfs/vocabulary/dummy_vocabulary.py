

class DummyVocabulary:

    def __init__(self):

        pass

    def is_match(self, *args):

        return False

    def get_info(self):

        return None

    def reset_free(self):

        pass

    def generate_bindings(self, variables):

        return {}