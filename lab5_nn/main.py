"""
Autor: Viktoriia Nowotka, Karol Łukasik
"""

from data_reader import DataReader
from nn import NeuralNetwork
import numpy as np
from activation_functions import relu, softmax, tanh
import matplotlib.pyplot as plt


def plot_nn_results(values_list, best_value, name, nn_params, stds):
    x = range(len(values_list))

    plt.figure(figsize=(12, 7))
    #plt.plot(x, values_list, marker="o", linestyle="-", label=name)
    
    plt.errorbar(x, values_list, yerr=stds, fmt='o-', capsize=5, 
                 label=f"{name} (mean ± std)", color='tab:blue', ecolor='gray')

    best_index = values_list.index(best_value)
    plt.scatter(best_index, best_value, s=120)
    plt.axhline(best_value, linestyle="--", alpha=0.6)

    if name == "Structure":
        x_labels = [
            f"{len(p)} layers\n" + "-".join(str(layer["neurons"]) for layer in p)
            for p in nn_params
        ]
    elif name == "Learning rate":
        x_labels = nn_params
    else:
        x_labels = nn_params

    plt.xticks(x, x_labels, rotation=45, ha="right")

    plt.xlabel(name)
    plt.ylabel("Accuracy")
    plt.title(f"Porównanie NN – {name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    seed = 42
    seeds_count = 25
    file = "data.csv"
    target = "quality"

    fr = DataReader(file, target, augmentation=True)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = fr.read_data()

    n_epoch = 10000
    l_rate = 0.005

    inputs = X_train.shape[1]
    outputs = Y_train.shape[1]

    nn_accuracies = []
    nn_accuracies_stds = []
    nn_params_list = [
        [
            {"neurons": inputs, "activation": relu},
            {"neurons": 36, "activation": relu},
            {"neurons": 16, "activation": relu},
            {"neurons": outputs, "activation": softmax},
        ],
        [
            {"neurons": inputs, "activation": relu},
            {"neurons": 36, "activation": relu},
            {"neurons": outputs, "activation": softmax},
        ],
        [
            {"neurons": inputs, "activation": relu},
            {"neurons": 16, "activation": relu},
            {"neurons": outputs, "activation": softmax},
        ],
        [
            {"neurons": inputs, "activation": tanh},
            {"neurons": 16, "activation": tanh},
            {"neurons": outputs, "activation": softmax},
        ],
        [
            {"neurons": inputs, "activation": relu},
            {"neurons": 16, "activation": tanh},
            {"neurons": outputs, "activation": softmax},
        ],
        [
            {"neurons": inputs, "activation": relu},
            {"neurons": outputs, "activation": softmax},
        ],
    ]

    print("START searching best nn_params")
    for nn_params in nn_params_list:
        acc_sum = 0
        current_params_scores = []

        for i in range(seeds_count):
            np.random.seed(seed + i)

            nn = NeuralNetwork(nn_params, n_epoch, l_rate)
            nn.fit(X_train, X_val, Y_train, Y_val)
            acc = nn.calculate_accuracy(X_test, Y_test)
            acc_sum += acc
            current_params_scores.append(acc)
            print("NN params: ", nn.get_parameters())
            print(f"Accuracy on test data: {acc:.2f}")

        nn_accuracies.append(acc_sum / seeds_count)
        std_acc = np.std(current_params_scores)
        nn_accuracies_stds.append(std_acc)

    best_index = np.argmax(nn_accuracies)
    best_accuracy = nn_accuracies[best_index]
    best_params = nn_params_list[best_index]
    print(f"BEST NN params: {best_params}")
    plot_nn_results(nn_accuracies, best_accuracy, "Structure", nn_params_list, nn_accuracies_stds)


    nn_accuracies = []
    nn_accuracies_std = []
    l_rates = [0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1]

    print("\n\nSTART searching best l_rate")
    for l_rate in l_rates:
        acc_sum = 0
        current_l_rate_scores = []

        for i in range(seeds_count):
            np.random.seed(seed + i)
            nn = NeuralNetwork(best_params, n_epoch, l_rate)
            nn.fit(X_train, X_val, Y_train, Y_val)
            acc = nn.calculate_accuracy(X_test, Y_test)
            acc_sum += acc
            current_l_rate_scores.append(acc)
            print("NN params: ", nn.get_parameters())
            print(f"Accuracy on test data: {acc:.2f}")

        nn_accuracies.append(acc_sum / seeds_count)
        std_acc = np.std(current_l_rate_scores)
        nn_accuracies_std.append(std_acc)

    best_index = np.argmax(nn_accuracies)
    best_accuracy = nn_accuracies[best_index]
    best_l_rate = l_rates[best_index]
    print(f"BEST l_rate: {best_l_rate:.2f}")
    plot_nn_results(nn_accuracies, best_accuracy, "Learning rate", l_rates, nn_accuracies_std)


    nn = NeuralNetwork(best_params, n_epoch, l_rate)
    nn.fit(X_train, X_val, Y_train, Y_val)
    y_pred = nn.predict(X_test)
    acc = nn.calculate_accuracy(X_test, Y_test)

    print(f"Accuracy on best structure + best l_rate: {acc:.2f}\n")
    recalls = nn.recall_per_class(Y_test, y_pred, n_classes=outputs)

    for cls, r in recalls.items():
        print(f"Label {cls}: recall = {r:.2f}")


def test_main():
    file = "data.csv"
    target = "quality"

    fr = DataReader(file, target)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = fr.read_data()

    n_epoch = 10000
    l_rate = 0.01

    inputs = X_train.shape[1]
    outputs = Y_train.shape[1]

    nn_params = [
        {"neurons": inputs, "activation": relu},
        {"neurons": outputs, "activation": softmax},
    ]

    nn = NeuralNetwork(nn_params, n_epoch, l_rate)
    print(nn.visualization(), "\n\n")

    nn.fit(X_train, X_val, Y_train, Y_val)
    y_pred = nn.predict(X_test)
    # loss_val = nn.calculate_loss(X_test, Y_test)
    acc_val = nn.calculate_accuracy(X_test, Y_test)

    print("NN params: ", nn.get_parameters())
    # print("Loss on test data: ", loss_val)
    print("Accuracy on test data: ", acc_val, "\n")

    recalls = nn.recall_per_class(Y_test, y_pred, n_classes=outputs)

    for cls, r in recalls.items():
        print(f"Label {cls}: recall = {r:.3f}")


if __name__ == "__main__":
    main()
