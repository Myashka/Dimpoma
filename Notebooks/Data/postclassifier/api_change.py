from .classifier import AbstractClassifier


class API_change(AbstractClassifier):
    def __init__(self):
        self.apinames = "Petit Four|Cupcake|Donut|Eclair|Froyo|Gingerbread|Honeycomb|Ice Cream Sandwich|ICS|Jelly Bean|KitKat|Lollipop|Marshmallow|Nougat|Oreo|Pie"

        self.pattern = [
            "(before|after|above|below|higher|lower|different|migrat(\w{0,3}))( [^\s]+){0,5} (API( )?(level)?|version)",
            "(before|after)( [^\s]+){0,5} (upgrad(\w{1,4})|updat(\w{1,4})|downgrad(\w{0,4})|switch(\w{0,4})|change(\w{1,4})|mov(\w{1,4})|migrat(\w{0,3}))( [^\s]+){0,5}",
            "(upgrad(\w{1,4})|updat(\w{1,4})|switch(\w{0,3})|chang(\w{1,4})|mov(\w{1,4})|migrat(\w{0,3}))( [^\s]+){0,5} (API( level)?|(API )?version)",
            "(not|don't|does[^\s]{0,3})? work\w{0,3} (with|on)( [^\s]+){0,3} (but|if)( [^\s]+) (not|don't|does[^\s]{0,3})? work\w{0,3}",
            "remov\w{0,4}( [^\s]+){0,5} (functionality|ability)",
            "(version|API|level)( [^\s]+){0,2} from( [^\s]+){0,2} ("
            + self.apinames
            + "|\d\.(\d|\w)(\.(\d|\w))?) to( [^\s]+){0,2} ("
            + self.apinames
            + "|\d\.(\d|\w)?(\.(\d|\w))?)",
            "downgrad",
            "work( [^\s]+){0,5}(API|level|version)",
            "(API( )?(level)?|version)( [^\s]+){0,5} (before|after|above|below|higher|lower|different|migrat(\w{0,3}))( [^\s]+){0,5} work",
            "(before|after) (API|level|version)",
            "crash(\w{0,3}) (on|with|at|by)( (Android|version|API|level))?( [^\s]+){0,1} ("
            + self.apinames
            + "|\d\.(\d|\w)?(\.(\d|\w))?)( [^\s]+){0,3} work(\w{0,3}) (on|with|at|by)( (Android|version|API|level))? ("
            + self.apinames
            + "|\d\.(\d|\w)?(\.(\d|\w))?)",
            "work(\w{0,3}) (on|with|at|by)( (Android|version|API|level))?( [^\s]+){0,1} ("
            + self.apinames
            + "|\d\.(\d|\w)?(\.(\d|\w))?)( [^\s]+){0,3} crash(\w{0,3}) (on|with|at|by)( (Android|version|API|level))? ("
            + self.apinames
            + "|\d\.(\d|\w)?(\.(\d|\w))?)",
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

        points = count_yes - count_no
        if points > 0:
            return 1
        else:

            return 0
