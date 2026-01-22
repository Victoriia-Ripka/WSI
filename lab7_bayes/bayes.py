"""
Author: Viktoriia Nowotka

Celem ćwiczenia jest implementacja algorytmu naiwnego klasyfikatora Bayesa.
"""
import math
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

    @staticmethod
    def __preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^a-z ]', '', text)
        return text.split()

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

    def predict(self, X):
        predictions = []
        for sentence in X:
            predictions.append(self.__predict(sentence))

        return predictions

    def __predict(self, text):
        scores = {}

        for c in self.classes:
            score = math.log(self.class_counts[c] / sum(self.class_counts.values()))
            total_words = sum(self.word_counts[c].values()) + len(self.vocab)

            for word in self.__preprocess_text(text):
                count = self.word_counts[c].get(word, 0)
                score += math.log((count + 1) / total_words)

            scores[c] = score

        return max(scores, key=scores.get)
