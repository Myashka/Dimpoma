from .classifier import AbstractClassifier


class Documentation(AbstractClassifier):
    def __init__(self):

        self.pattern = [
            "(find(\w{0,3})|search(\w{0,3})|know(\w{0,3})|suggest(\w{0,5})|provid(\w{0,3})|giv(\w{0,3})|us(e|ing)|post(\w{0,3})|provid(\w{0,3})|can i( [^\s]+)? have|look(\w{0,3}))( [^\s]+){0,5} "
            "(material|tutorial|docu(mentation)?|information|study|book(s)?)",
            "(what|any|(are|is) there|which)( [^\s]+){0,5} (material(\w{0,3})|tutorial(\w{0,3})|docu(mentation)?|study|book(s)?|instructions?)",
            "(material(\w{0,3})|tutorial(\w{0,3})|docu(mentation)?|study|book(s)?) (available|help(ful)?|explain)",
            "walkthrough",
            "(don't|dont|does not|doesnt|doesn't|can't|cant|cannot|could|couldn't|couldnt|not)( [^\s]+){0,6} (material(\w{0,3})|tutorial(\w{0,3})|docu?(mentation)?|guide(\w{0,5})|study|book(s)?)(\s|\.|\,)",
        ]

        self.antipattern = ["problem|issue"]

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

        for pat in self.antipattern:
            found, m = self.countMatches(pat, title + " " + question)
            count_no += found
            if found > 0:
                support.append([(-1) * found, pat])

        points = count_yes - count_no
        if points > 0:
            # dlog("classify post with id: ", post.id, "into ",name)
            return 1
        else:
            # dlog("classify post with id: ", post.id, " into " + "NOT-"+name)
            return 0
