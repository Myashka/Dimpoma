from .classifier import AbstractClassifier


class Review(AbstractClassifier):
    def __init__(self):
        self.pattern = [
            "is there( [^\s]+){0,3} a(n|ny)?( [^\s]+){0,3} (better|more efficient|other|simple(r)?|easy|easier) (way|method|solution) (to|of)?",
            "(what|what's|guidance|explain)( [^\s]+){0,5} (best|other|proper|suggested) (way|method|solution|approach|option|implementation)",
            "is( [^\s]+){0,5} the (best|usual) (approach|way|method|solution)",
            "(what|is there|there('s| is)|hav(\w{1,3})|suggest)( [^\s]+){0,5} (alternat(\w{0,5})|work\s?around)",
            "what is better( [^\s]+){1,5} or( [^\s]+){1,5}",
            "what (is|are)( [^\s]+){0,5} (better|best|good) practice",
            "( [^\s]+){0,5}, right\?",
            "(what|which)( [^\s]+){1,5} (should I use|to improve)",
            "(is there|what (is|are))( [^\s]+){0,5} (guideline|rule of thumb|best practi(c|s)e|improvement)",
            "(what|how|which)( [^\s]+){0,5} (improve|optimize|work\s?out|recommend)",
            "is( [^\s]+){0,5} used( [^\s]+){0,5} (correctly|right)",
            "should i( [^\s]+){0,5} (switch|us(\w{0,3})|do|launch|call|run)",
            "is it( [^\s]+){0,1} better (to|if)",
            "is( [^\s]+){1,4} (unusual|ok|usual|normal)",
            "what( [^\s]+){0,7} make\w? sense",
            "what( [^\s]+){0,4} (wrong|think|idea)( [^\s]+){0,4} concept",
            "(what is|need|ask for)( [^\s]+){0,7} opinion",
            "(is it|would( [^\s]+){0,2} be) better",
            "is( [^\s]+){0,3} good (idea|decision)",
            "(is this|am I|is my|is there)( [^\s]+){0,3} (right|wrong|correct|satisf)",
        ]

        self.antipattern = ["what am i doing wrong"]
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
            return 1
        else:
            return 0
