import sys

import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer



status = { "error" : "Not Yet Loaded" }

vectorizer = None
LRModel    = None


def load_model():
    global status
    global vectorizer
    global LRModel

    vectorizer_path = "../model/vectorizer.pickle"
    model_path = "../model/LRModel.pickle"


    try:
        f = open(vectorizer_path, "rb")

    except OSError:
        error_str = "Error Loading Vectorizer"
        print(error_str)
        status = { "error": error_str }

        return status

    vectorizer = pickle.load(f)
    f.close()


    try:
        f = open(model_path, "rb")

    except OSError:
        error_str = "Error Loading Model"
        print(error_str)
        status = { "error": error_str }

        return status

    LRModel = pickle.load(f)
    f.close()


    status = { "model" : model_path, "vectorizer" : vectorizer_path }

    return status


def predict(text):
    if "error" in status: return "!<ERROR>: Check status."


    text      = vectorizer.transform([text])
    sentiment = LRModel.predict(text)

    result = "Error"

    if(sentiment   == 1): result = "Positive"
    elif(sentiment == 0): result = "Negative"

    return result
