"""
Autor: Viktoriia Nowotka

Dla chętnych:
Zaproponuj sposób poprawy jakości klasyfikacji dla najmniej licznych klas w zbiorze danych.
Porównaj ogólną jakość klasyfikacji oraz jakość klasyfikacji najmniej licznych klas z siecią
przygotowaną w ramach ćwiczenia.
"""
import numpy as np
from lab5_nn.data_reader import DataReader
from lab5_nn.nn import NeuralNetwork


def main():
    file = 'data.csv'
    target = 'quality'

    fr = DataReader(file, target)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = fr.read_data()

    # print(X_train.shape, Y_train.shape)

    # dla testów na początek
    X_train_small = X_train[0:5, 0:3]
    Y_train_small = Y_train[0:5]
    # print(X_train_small, Y_train_small)

    n_epoch = 100
    l_rate = 0.01

    nn_params = [
        {'neurons': X_train_small.shape[1],
         'activation': 'sigmoid'},
        {'neurons': 3,
         'activation': 'sigmoid'},
        {'neurons': len(np.unique(Y_train_small)),
         'activation': 'relu'},
    ]

    nn = NeuralNetwork(nn_params, n_epoch, l_rate)
    print(nn.visualization())
    # nn.fit(X_train_small, Y_train_small)









if __name__ == "__main__":
    main()