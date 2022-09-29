"""
Transform files from csv to pickle for faster reading
"""
import pickle

import pandas as pd

df = pd.read_csv("..\..\data\StackSample\Answers.csv", encoding_errors="replace")
with open("../data/answers.p", "wb") as f:
    pickle.dump(df, f)

df = pd.read_csv("..\..\data\StackSample\Questions.csv", encoding_errors="replace")
with open("../data/questions.p", "wb") as f:
    pickle.dump(df, f)

df = pd.read_csv("..\..\data\StackSample\Tags.csv", encoding_errors="replace")
with open("../data/tags.p", "wb") as f:
    pickle.dump(df, f)
