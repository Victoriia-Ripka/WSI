"""
Autor: Viktoriia Nowotka, Karol Łukasik

Dla chętnych:
Zaproponuj sposób poprawy jakości klasyfikacji dla najmniej licznych klas w zbiorze danych.
Porównaj ogólną jakość klasyfikacji oraz jakość klasyfikacji najmniej licznych klas z siecią
przygotowaną w ramach ćwiczenia.
"""
import numpy as np
from data_reader import DataReader
from nn import NeuralNetwork
from activation_functions import relu, sigmoid, tanh, softmax


def main():
    file = 'data.csv'
    target = 'quality'

    fr = DataReader(file, target)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = fr.read_data()

    # print(X_train.shape, Y_train.shape)

    # dla testów na początek
    X_train_small = X_train[0:100, :]
    Y_train_small = Y_train[0:100]
    # print(X_train_small, Y_train_small)

    n_epoch = 1001
    l_rate = 0.01

    nn_params = [
        {'neurons': X_train_small.shape[1],
         'activation': relu},
        {'neurons': 12,
         'activation': relu},
        {'neurons': 6,
         'activation': softmax},
    ]

    nn = NeuralNetwork(nn_params, n_epoch, l_rate)
    print(nn.visualization())
    nn.fit(X_train_small, Y_train_small)









if __name__ == "__main__":
    main()