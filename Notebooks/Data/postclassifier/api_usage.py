import re
from collections import namedtuple

from .classifier import AbstractClassifier


class API_usage(AbstractClassifier):
    def __init__(self):
        self.res = namedtuple("res", "id, reason, points")
        self.pattern = [
            "how( to)?( [^\s]+){0,3} (do (this|it)?|get|use|achieve|access|make|show|display|accomplish|have|focus|start|call|write|implement|disable|enable|find|change|select|remove|close|add|restrict)",
            "how( [^\s]+)? (can|could|should|do|does) (I|you|one|it)",
            "how (can|could|should|do|does) (I|you|one|it) (do (this|it)?|get|use|achieve|access|make|show|display|accomplish|have|focus|start|call|write|implement|disable|enable|find|change|select|remove|close|add|restrict)",
            "how( [^\s]+){0,3} done",
            "(wonder\w{0,3}|tell me|know)( [^\s]+){0,3} how",
            "how to( [^\s]+){0,10}\?",
            "how (can|could|should|do|does) (I|you|one|it)( [^\s]+){0,10}\?",
            "where do I (find|start|change|call|use|have|get|add)",
            "what( [^\s]+){0,1} need to (do|have)",
        ]

        self.antipattern = [
            "(get|work)( )?around",
            "understand( [^\s]+){1,5} how",
            "(try|tried|trying)( [^\s]+){1,5} how",
            "how( [^\s]+){0,5} solve",
        ]

        return

    @property
    def name(self):
        return self.__class__.__name__.upper()

    def classify(self, title, question):
        count_yes = 0
        count_no = 0

        support = []
        for pat in self.pattern:
            found, m = self.countMatches(pat, title + " " + question)
            count_yes += found
            if found > 0:
                support.append([found, pat])
                # print(m)

        for pat in self.antipattern:
            found, m = self.countMatches(pat, title + " " + question)
            count_no += found
            if found > 0:
                support.append([(-1) * found, pat])

        points = count_yes - count_no
        if points > 0:
            return 1
        else:
            return 0
