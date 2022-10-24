import re
from abc import ABC, abstractmethod


class AbstractClassifier(ABC):
    def __init__(self):
        #  self._post = post;
        self.__name = self.__class__.__name__
        return

    def countMatches(self, pattern, text):

        p = re.compile(pattern.lower())
        m = p.findall(text)

        return len(m), m

    @abstractmethod
    def classify(self, post):
        pass
