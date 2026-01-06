"""
Autor: Viktoriia Nowotka, Karol Łukasik

Dla chętnych:
Zaproponuj sposób poprawy jakości klasyfikacji dla najmniej licznych klas w zbiorze danych.
Porównaj ogólną jakość klasyfikacji oraz jakość klasyfikacji najmniej licznych klas z siecią
przygotowaną w ramach ćwiczenia.
"""
from data_reader import DataReader
from nn import NeuralNetwork
from activation_functions import relu, sigmoid, tanh, softmax


def main():
    file = 'data.csv'
    target = 'quality'

    fr = DataReader(file, target)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = fr.read_data()

    n_epoch = 100000
    l_rate = 0.01

    inputs = X_train.shape[1]
    outputs = Y_train.shape[1]

    nn_params = [
        {'neurons': inputs,
         'activation': relu},
        {'neurons': inputs,
         'activation': relu},
        {'neurons': outputs,
         'activation': softmax},
    ]

    nn = NeuralNetwork(nn_params, n_epoch, l_rate)
    # print(nn.get_parameters())
    print(nn.visualization(), '\n\n')

    nn.fit(X_train, X_val, Y_train, Y_val)

    loss_val = nn.calculate_loss(X_test, Y_test)
    print("\n\nLoss on test data: ", loss_val)


if __name__ == "__main__":
    main()