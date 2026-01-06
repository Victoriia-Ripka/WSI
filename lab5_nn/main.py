"""
Autor: Viktoriia Nowotka, Karol Łukasik
"""
from data_reader import DataReader
from nn import NeuralNetwork
from activation_functions import relu, softmax, tanh


def main():
    file = 'data.csv'
    target = 'quality'

    fr = DataReader(file, target)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = fr.read_data()

    n_epoch = 100000
    l_rate = 0.02

    inputs = X_train.shape[1]
    outputs = Y_train.shape[1]

    nn_params = [
        {'neurons': inputs,
         'activation': relu},
        {'neurons': 32,
         'activation': tanh},
        {'neurons': 16,
         'activation': relu},
        {'neurons': outputs,
         'activation': softmax},
    ]

    nn = NeuralNetwork(nn_params, n_epoch, l_rate)
    # print(nn.visualization(), '\n\n')

    nn.fit(X_train, X_val, Y_train, Y_val)
    y_pred = nn.predict(X_test)
    loss_val = nn.calculate_loss(X_test, Y_test)
    acc_val = nn.calculate_accuracy(X_test, Y_test)

    print("NN params: ", nn.get_parameters())
    print("Loss on test data: ", loss_val)
    print("Accuracy on test data: ", acc_val, '\n')

    recalls = nn.recall_per_class( Y_test, y_pred, n_classes=outputs)

    for cls, r in recalls.items():
        print(f"Label {cls}: recall = {r:.3f}")


if __name__ == "__main__":
    main()