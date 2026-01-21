"""
Author: Viktoriia Nowotka

Celem ćwiczenia jest implementacja algorytmu naiwnego klasyfikatora Bayesa.
Następnie należy wykorzystać stworzony algorytm do stworzenia i zbadania jakości
klasyfikatorów dla zbioru danych SMS Spam Collection Dataset.
Klasą jest pole class.
"""
import re
import numpy as np
from collections import defaultdict
from lab7_bayes.solver import Solver

class Bayes(Solver):
    def __init__(self):
        self.classes = []
        self.word_counts = {}
        self.class_counts = {}
        self.vocab = set()

    def get_parameters(self):
        return {'classes': self.classes}

    def __preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z ]', '', text)
        return text.split()

    # TODO
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.word_counts = {c: defaultdict(int) for c in self.classes}
        self.class_counts = {c: 0 for c in self.classes}

        for text, label in zip(X, y):
            self.class_counts[label] += 1
            sentences = self.__preprocess_text(text)

            for word in sentences:
                self.word_counts[label][word] += 1

        for c in self.classes:
            self.vocab |= self.word_counts[c].keys()

        print(self.vocab)
        print(self.word_counts)
        print(self.class_counts)

    # TODO
    def predict(self, X):
        pass


