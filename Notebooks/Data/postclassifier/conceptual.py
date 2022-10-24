from .classifier import AbstractClassifier


class Conceptual(AbstractClassifier):
    def __init__(self):
        self.pattern_easy = [
            "Is it possible to",
            "Is there a way to",
            "What is the difference between",
            "Is there any way to",
            "Is there any",
            "Can I use",
            "why",
            "Can i",
            "is there any way I can",
            "Is this possible?",
            "is it possible",
            "what is",
            "can anyone tell me how the " "Buy" " button fires",
            "if there is a way to",
            "What is the use of",
            "Is there",
            "Is there a way to do",
            "Why would",
            "Is there any way",
            "shouldn't it get",
            "How much RAM is available for use by each app?",
            "what pattern do you use to",
            "Is this still true?",
            "Is startDocument() with Androids XMLSerializer required",
            "whether it is possible",
            "where I set",
            "Why?",
            "Anyone has this working properly?",
            "What is the startGC() function?",
            "if there is a way to restrict",
            "to understand what",
            "Is there any additional information I need to",
            "want to know if this was possible",
            "what the difference is",
            "What are the implications of",
            "Is there a comparable way",
            "Is there an alternative library",
            "if exist a shortcut for this operation.",
            "know if I understand correctly",
            "is there way",
            "what does this button",
            "if there is anyway to",
            "How is it different with",
            "can I increase",
            "Is there any good way",
            "Can anyone clarify that ?",
            "Are there other objects that need to be destroyed or removed when the Activity is destroyed?",
            "What does sign mean",
            "Is there any issue",
            "Where would the proper place be",
            "what is the purpose for",
            "why is there a",
            "what are the pitfalls",
            "Or am I completely wrong and all three are completely different entities?",
            "can we use",
            "What knowledge/expertize is required to",
            "or it's overridden by",
            "So what are the differences",
            "Is there any good way of doing this?",
            "Is it not possible to",
            "Is there a way to realize",
            "If the handler was instantiated in the main UI thread, does a post with a Runnable create a child thread that gets added to the message queue, or does",
            "Is their a reason",
            "Can anybody give such a use case.",
            "Is there a maximum number of copies I can sell?",
            "Do they even exist?",
            "is the connection already opend",
            "Does anyone have any experience using both of these systems?",
            "What does that # mean?",
            "What are the equivalent",
            "Is WebView affected",
            "what the full path of my xml file",
            "when is a context different from ",
            "I have tried in Normal class but I am unable",
            "What does " "@link" " mean?",
            "Is this some kind of bugs?",
            "Shouldnt this",
            "where is the database on the real device",
            "What is happening above?",
            "Is there any solution for this?",
            "I couldn't understand where should I",
            "what's the difference between",
            "Can it only be used",
            "Is there anything else I need to consider?",
            "how the " "Buy" " button works",
            "is their any chance to",
            "can I do",
            "Can it be used",
            "when is one method favored (or more appropriate) over the other and why?",
            "Is there a built in way to accomplish this or will I need to",
            "Is this possible",
            "What does that mean?",
            "do I need to",
            "Can I create",
            "What does android:enabled mean",
            "Could you tell me why?",
            "what happens if",
            "Can only MyBackgroundView listen for touches, or can",
            "is there any prebuilt files ",
            "And can I run",
            "Why could it be?",
            "Is android application not running on Virtual Box?",
            "where do I put",
            "What will happen if I don't use",
            "What is a valid",
            "WHat will be pros and cons of",
            "is this mode exclussive for android or does android also support HCI or Bnep",
            "What is the cons of using",
            "Can anyone explain me why",
            "if there an easy way to do this",
            "When using SQLite?",
            "Is a service considered",
            "What is exact use of",
            "Will it also be possible to",
            "when is a context different from",
            "Whether any form of pre-image processing would be useful?",
            "what would a multithreaded version of the below code look like",
            "I am wondering why",
            "I wonder if I can",
            "Is there anything special I should know",
            "If there is a way to do",
            "Is Qt available for it",
            "do not understand why it doesn't work",
            "but I just wonder if I can",
            "At what point is the button and its events created",
            "Does anyone know if I can",
            "why this is a problem",
            "why inconsistently?",
            "can I assume",
            "what are the most common naming conventions",
            "is it not necessary?",
            "there has to be a reason why",
            "Is there such a library",
            "How would you design",
            "Is using ActionBarSherlock still necessary",
            "Could I use",
            "If there is any",
            "Are there other",
            "What is this behavior",
            "Can I sell",
            "Will it be faster and less battery expensive?",
            "if the activity goes, will the countdowntimer go as well?",
            "I would like to know about, the Design Guidelines",
            "Is there any other way?",
            "Have anybody did the same things before",
            "is there some formula",
            "explain to me why",
            "Is there some way to",
            "difficulty in understanding the use of",
            "Anyone who has used",
            "can this be done on a more real time way",
            "whats the use of",
            "How should I structure my UI?",
            "what exactly is going on here?",
            "Can this ImageLoader place image to a Bitmap somehow?",
            "can I also get",
            "Is there a standard " "Loading, please wait" " dialog",
            "Does a custom Adapter that works for a ListView also work for a Spinner?",
            "Is it required that",
            "is there any other way to",
        ]
        self.pattern = [
            "(is|it)( [^\s]+){0,5} (possible)",
            "(is)( [^\s]+){0,5} (a|any|some)( [^\s]+){0,3} (way|chance|reason|library|bug|information|pattern)",
            "(does|know|do)( [^\s]+){0,5} (exist)",
            "(a|any|some) (way|chance|solution|library|information|pattern) to (use|apply|detect)",
            "can (I|we) (use|apply|detect)",  # design)"
            "(what('s|s)?|why|when|how|does|tell)( [^\s]+){0,5} (diff(e|i)?rence|alternative|equivalent|different|purpose|(use|usage) (of|to|in)|pitfalls|available|criteria|limit|pattern)",
            "(what|how)( [^\s]+){0,3} (structure|design|pattern)",
            "what( [^\s]+){0,5} (mean|implications|valid|right|correct|behaviour|behavior)",
            "is( [^\s]+){0,5} (necessary|affected|save to|required|maximum|minimum)",
            "(know|wonder\w{0,3}|reason)( [^\s]+){0,3} (why|guideline)",
            "(tell me|understand|explain)( [^\s]+){0,5} (where|why|correct)",
            "(when|what)( [^\s]+){0,3} (method|way) (favored|preferred)",
            "(what('s|s)?)( [^\s]+){0,5} (happens if|require|available|going on|scenarios|pitfalls|limit)",
            "(are|is) there( [^\s]+){0,2} any?"  # ( [^\s]+){0,3} (technique|tool|library|issue|API)"
            # , "(is|are)( [^\s]+){0,3}( (and|or)?( [^\s]+){0,3})? (required|need|available|preferred|allowed)"
            ,
            "(what is|which|is there)( [^\s]+){0,3} (class|method) (to|for|when|that)",
            "(tell|explain)( [^\s]+){0,3} why",
            "where( [^\s]+){0,3} ((to|I) (set|put|store)|correct place)"
            # , "does( [^\s]+){0,3} work"
            #
            #    "Is there any",, "why",
            #    ", "what is",
            #     "can anyone tell me how the ""Buy"" button fires",, "Is there",
            #     , "Can a Thread be started only once?", "Why would",
            #    , "doesnt look the same a", "shouldn't it get",
            #      "Does it make a difference if I'm creating my APK with Proguard?",
            #     "What updates are doing this?",
            #  , "Why?",
            #     "Anyone has this working properly?",
            #
            #    , "How to avoid",
            # "What is the startGC() function?",
            #     "How is that possible ",
            #    , "what the difference is",
            #  , "is there way",
            #     "what does this button", "what are the common scenarios in", "Is it possible like to have a",
            #     "if exist a shortcut for this operation.",
            #    "Any way to use holo theme colors in lower API", "Is there any good way",
            #     "Can anyone clarify that ?", "Can I modify TextView so tha",
            #     "What are the ways to read a PDF Document in Android?", "How is it different with",
            #     , "can I increase",
            #     "Is there any issue", "Where would the proper place be", "What is the default database location of",
            #     "can I set android:uiOptions="splitActionBarWhenNarrow
            #     " from the code, instead of hardcode it in the Manifest?",
            #   "What is the use",  "Where to find the soure code of", "what are the pitfalls",
            #     "Or am I completely wrong and all three are completely different entities?",
            #     "I was just wondering if I can"
            #     "what is the concrete class that has storing/retrieve", "why is there a", "Is it same as for rooted one?",
            #
            #     "is it just how certain browsers behave", "
            #     "where does the external (concrete) reading/storing of values from xml lie?", "or it's overridden by",
            #     "Is their a reason", "Can anybody give such a use case.",
            #     "what would be the problem in the future?",
            #
            #     "If the handler was instantiated in the main UI thread, does a post with a Runnable create a child thread that gets added to the message queue, or does",
            #     "Why is this?", "Does anyone have any experience using both of these systems?",
            #  ", "Can I get notified through an",
            #     "Do they even exist?", "How would I", "is the connection already opend",
            #     "Is there something like extraSettings.get(""myKey"")?"","what the full path of my xml file","when is a
            #     context different from this , "1",, "is there any specific class to do it in Java?",  "Is WebView affected", "how to test", "Does Xamarin.Android have any requirements against the device,",  "if it would be possible to", "What is happening above?", "I'm currently looking for a possibility to implement", "Is this some kind of bugs?", "Is there a chance for", "Shouldnt this", "Is it that MediaPlayer can't play files on cache?", "Is there an exclude analog", "what's the difference between", "when should I use getResources() as opposed to directly calling getString()?", "Can it only be used", "is there any way to solve the issue?", "Is there any solution for this?", "Is there any way to put", "I couldn't understand where should I", "Is there an equivalent to", "is there a way to read only that part of text", "can I do", "Can I hide the", "Can it be used", "Is it possible to test", "Is there anything else I need to consider?", "Is there a way to set", "how the ""Buy"" button works", "Could you explain what each of these jar files mean?", "Is it even possible?", "is their any chance to", "what should include", ,
            # , "Can anything be done to allow", "is there a way to modify", "What does android:enabled mean",
            #  , , "Is there a way to do this easily,", "Is SomeFragment removed too? Can it be collected? Or is FragmentManager still holding reference to it?", "Can I load", "do I need to", "is there any prebuilt files ", "What is the size, in pixels, of", "And can I run", "Can I make regular calls from tablets.", "Why could it be?", "help me figure out what is different", "what happens if", "Is there any way of achieving", "Is there anyway I can do this?", "Can only MyBackgroundView listen for touches, or can", "What will happen if I don't use", "I am wondering if the following is possible in android", "What is a valid", "Are @id/ and @android:id/ the same?", "WHat will be pros and cons of", "Is there any policy from google play that", "Is android application not running on Virtual Box?", "are there any other kinds of restrictions ?", "How does twitter/facebook know on", "where do I put", "What is the cons of using", "What kinds of devices can be Android compatible?", "Can anyone explain me why", "What's the maximum size for", "if there an easy way to do this", "is this a feature of Java too or not", "is this mode exclussive for android or does android also support HCI or Bnep", "would it be possible to launch", "How many interface we can declare within a class", "What is exact use of", "does the tablets support telephony", "Will it also be possible to", "How do I know which", "when is a context different from", "what is the name of this pattern or feature", "When using SQLite?", "Is there is any way to", "Is there any way to trigger", "Is a service considered", "how to determine", "Is there anyway to", "I am wondering why", "Can I use these services to", "Whether any form of pre-image processing would be useful?", "Does anybody know if an AES implementation is guaranteed to come", "Is there a way to prevent", "what would a multithreaded version of the below code look like", "what the practical limit is", "Is Qt available for it", "how to get around", "do not understand why it doesn't work", "is there a way to make sure that", "I wonder if I can", "Why does it wait at the reserved IPs", "is there any detection around", "Is there anything special I should know", "If there is a way to do", "what the actual limit is", "Does anyone know if I can", "why this is a problem", "Is it possible to do it?", "but I just wonder if I can", "Is there a tool to", "Can I get the UUID", "At what point is the button and its events created", "is it not necessary?", "I wondered whether it could be done for", "there has to be a reason why", "Can I get a scroll", "why inconsistently?", "Is there a way that can", "is there a easy and better way to accoumplish this?", "can I assume", "what are the most common naming conventions", "Reasons to use", "How would you design", "Is Scrolling textview allowed in", "Is there a solution or is it impossible", "it is possible to adjust the app window", "what is its purpose / what does it do?", "Is there such a library", "Dialog.show() vs. Activity.showDialog()", "Are there other", "Is it legal to", "What is this behavior", "Is there any idea", "Is using ActionBarSherlock still necessary", "Does one need to clear state between inferences?", "What's the use of classpath", "Could I use", "If there is any", "how do I", "Can anyone tell me the difference in how", "I would like to know about, the Design Guidelines", "does it matter how the REST web service is built ?", "Can I sell", "what are the common cases in", "is ther any other solution whic could", "how does it store/retrieve", "Will it be faster and less battery expensive?", "if the activity goes, will the countdowntimer go as well?", "whether the rotation of an Android device can be detected", "Have anybody did the same things before", "Where or how should I use adapter.notifyDataSetChanged()", "is there some formula", "which type of web services can a Android App talk to?", "I am wondering whether I can leave", "do I have to resize", "Is there any other way?", "what do I do if I want to run the thread again?", "Anyone who has used", "What are the common version in the Android devices this days?", "can this be done on a more real time way", "is the URL specifically for a network resource?", "explain to me why", "Where can I find a default layout", "Is there any thing like Internet Listener?", "is it possible to simply run", "Is there some way to", "difficulty in understanding the use of", "What can I do?", "what SDK(s) should i choose to install", "Can this ImageLoader place image to a Bitmap somehow?", "I want to use weight if it is possible", "whats the use of", "Is it possible for remote service to", "Is it possible to check if", "Does "
            # cacheColorHint attribute works here?",  "what exactly is going on here?", "How can I use", "Is it required that", "is there any other way to store", "is there any other way to", "Possible to", "can I also get", "Is there a way to keep the state of the variable?", "is there anyway I can solve this without", "why is it doing so", "Is there a standard ""Loading, please wait"" dialog", "Does a custom Adapter that works for a ListView also work for a Spinner?", "does M4V format file play"
        ]

        self.antipattern = [
            "better way",
            "how( [^\s]+){1,3} (possible to|differ)",
            "bypass",
            "workaround",
            "don't know",
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
