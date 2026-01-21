"""
Author: Viktoriia Nowotka

Celem ćwiczenia jest implementacja algorytmu naiwnego klasyfikatora Bayesa.
Następnie należy wykorzystać stworzony algorytm do stworzenia i zbadania jakości
klasyfikatorów dla zbioru danych SMS Spam Collection Dataset.
Klasą jest pole class.
"""
import re
from lab7_bayes.solver import Solver

class Bayes(Solver):
    def __init__(self, classes):
        self.classes = classes
        print(classes)

    def get_parameters(self):
        pass

    def __preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z ]', '', text)
        return text.split()

    def fit(self, X, y):
        self.word_counts = {c: defaultdict(int) for c in self.classes}
        self.class_counts = defaultdict(int)

        for text, label in zip(X, y):
            self.class_counts[label] += 1
            for word in self.__preprocess_text(text):
                self.word_counts[label][word] += 1

        self.vocab = set()
        for c in self.classes:
            self.vocab |= self.word_counts[c].keys()

    def predict(self, X):
        pass


