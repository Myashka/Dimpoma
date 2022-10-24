from .classifier import AbstractClassifier


class Discrepancy(AbstractClassifier):
    def __init__(self):
        self.pattern = [
            "i('m)?( [^\s]+){1,5} (try(\w{0,3})|tried)( [^\s]+){0,4} (but|however)",
            "(what('?s)?|where|point)( [^\s]+){0,5} (wrong|problem|mistake|issue)",
            "(but|however)( [^\s]+){0,5} (intend|wrong) ",
            "((solv(\w{0,3}))|resolv(\w{0,3})|fix(\w{0,3}))( [^\s]+){0,5} (problem\w?|this|it)",
            "make( [^\s]+){0,5} mistake",
            "why( [^\s]+){1,5}( )?(n't|not|nt) (work|install|run)",
            "(I|my)( [^\s]+){0,5} (missing|cause)",
            "(what)( [^\s]+){0,5} (solve|cause)",
            "(is)( [^\s]+){0,5} (missing)",
            "why( [^\s]+){0,5} happen",
            "(weird|strange|not expected|unusual|expected) (behaviour|behavior|result)",
            "not working( [^\s]+){0,2} expected",
            "(,|\.|!) why\?",
            "problem with",
            "ideas?( [^\s]+){0,5} (prevent|problem|issue)",
            "(it|this|solution)?( [^\s]+){0,5} (didn't|did not|didnt|don't|dont|does not|doesnt|doesn't|can't|cant|cannot|could|couldn't|couldnt)( [^\s]+){0,5} work",
            "to avoid this",
            "(doing|going)( [^\s]+){0,3} wrong",
            "wrong( [^\s]+){0,3} code",
            "how( [^\s]+){0,3} fix",
            "nothing happen",
            "any idea\w? what",
            "but( [^\s]+){0,4} (not|doesn't|didn't|did not|didnt|don't|couldn't|can't)( [^\s]+){0,4} work(ing)?",
        ]

        self.antipattern = [
            "(find(\w{0,3})|search(\w{0,3})|know(\w{0,3})|suggest(\w{0,3})|provid(\w{0,3}))( [^\s]+){0,5} (material|link|tutorial|docu(mentation)?)",
            "work\s?around",
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
            found, m = self.countMatches(pat, title.lower() + " " + question.lower())
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
