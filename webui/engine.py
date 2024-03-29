import re

import pickle

import nltk
from   nltk.stem   import WordNetLemmatizer
from   nltk.corpus import stopwords




class Engine:
    __labels_path     = "../model/emotions.csv"
    __vectorizer_path = "../model/vectorizer.pickle"
    __model_path      = "../model/DTCModel.pickle"


    status = {
            "status"     : "error",
            "labels"     : "",
            "vectorizer" : "",
            "model"      : ""
        }
    
    
    def __load_labels(self):
        try:
            f = open(self.__labels_path, "r")
        except OSError:
            self.status["status"] = "error"
            self.status["labels"] = "error"
            return


        labels = f.read().split(",")
        f.close()

        self.status["status"] = "ok"
        self.status["labels"] = self.__labels_path

        return labels


    def __load_vectorizer(self):
        try:
            f = open(self.__vectorizer_path, "rb")
        except OSError:
            self.status["status"]     = "error"
            self.status["vectorizer"] = "error"
            return


        vectorizer = pickle.load(f)
        f.close()

        self.status["status"] = "ok"
        self.status["vectorizer"] = self.__vectorizer_path

        return vectorizer


    def __load_model(self):
        try:
            f = open(self.__model_path, "rb")
        except OSError:
            self.status["status"] = "error"
            self.status["model"]  = "error"
            return


        model = pickle.load(f)
        f.close()

        model.verbose = 0

        self.status["status"] = "ok"
        self.status["model"] = self.__model_path

        return model
    

    def preprocess(self, text):
        res = str()
        
        text = text.lower()

        # regexes
        url = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
        usr_name = r"@[^\s]+"
        non_numalpha = r"[^a-zA-Z0-9]"

        # apply them
        text = re.sub(url, " URL", text)        # remove urls
        text = re.sub(usr_name, " USER", text)  # remove user tags
        text = re.sub(non_numalpha, " ", text)  # remove special chars

        
        # remove stopwords and lemmatize  the rest
        for token in text.split():
            if token not in self.stop_words:
                token = self.lemmatizer.lemmatize(token)
                res += (token + ' ')

        return res
    

    def predict(self, text):
        if self.status["status"] != "ok": return { "error": "check status" }
        
        result = {}

        test = self.preprocess(text)

        text = self.vectorizer.transform([text])
        pred = self.model.predict_proba(text)

        for i in range(len(pred)):
            result[self.labels[i]] = round(pred[i][0][1] * 100, 2)
        
        
        return result
    

    def __init__(self):
        self.labels     = self.__load_labels()
        self.vectorizer = self.__load_vectorizer()
        self.model      = self.__load_model()

        nltk.download('stopwords')
        nltk.download('wordnet')

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = stopwords.words("english")




