from .classifier import AbstractClassifier


class Errors(AbstractClassifier):
    def __init__(self):
        self.pattern = [
            "(get(\w{0,4})|got|throw(\w{0,3})|threw|show(\w{0,3})|giv(\w{0,3})|gave|hav(\w{0,3})|had|see(\w{0,3})|display(\w{0,3})|catch(\w{0,3})|(un)?caught|rece?ive)( [^\s]+){0,5} (error|(\w{0,35})exception)",
            "(error|exceptions?)( [^\s]+){0,5} (get(\w{0,3})|got|throw(\w{0,3})|show(\w{0,3})|giv(\w{0,3})|gave|hav(\w{0,3})|had|display(\w{0,3})|catch(\w{0,3})|caught)",
            "(log\s?cat|stack\s?trace|log|gradle)( [^\s]+){0,5} (error|exception)",
            "(errors?|exceptions?)( [^\s]+){0,5} (logcat|stacktrace|log|message)",
            "((re)?solve|fix)( [^\s]+){0,5} (error|(\w{0,35})exception)",
            "(java.lang.)?NullpointerException",
            "(android.view.)?InflateException",
            "java.lang.RuntimeException",
            "Null Object Reference",
            "crash",
        ]

        self.antipattern = [
            "(not get(\w{0,4})|no) (error|exception)",
            "(don't|dont|does not|doesnt|doesn't|no)( [^\s]+){0,3} (crash|throw)",
        ]
        return

    @property
    def name(self):
        return self.__class__.__name__.upper()

    def classify(self, title, question):
        # dlog(id, ":  ", rr)
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
