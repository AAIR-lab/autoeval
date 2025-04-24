from abc import ABC
from abc import abstractmethod

class Dataset(ABC):

    def __init__(self, **kwargs):

        pass

    @abstractmethod
    def generate(self, size, filepath=None,
                 seed=None,
                 **kwargs):

        raise NotImplementedError