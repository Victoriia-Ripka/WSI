"""
Autor: Viktoria Nowotka

parametryzowalna liczba warstw i neuronów w każdej warstwie, możliwość wyboru
różnych funkcji straty, aktywacji oraz algorytmu optymalizacyjnego.
"""

import numpy as np
from lab5_nn.activation_functions import sigmoid, relu, tanh, softmax
from lab5_nn.solver import Solver

# TODO normalizacja wag oraz wejścia
class NeuralNetwork(Solver):
    def __init__(self, params, n_epoch, l_rate):
        self.n_epoch = n_epoch
        self.l_rate = l_rate

        self.layers_amount = len(params)
        self.layers_conf = params

        # macierz 3D: [] -> warstwa[ndarray] -> neuron[]
        """
            [
                array([[ 0.07009308, -0.37176889, -1.59886717],
                       [-0.97365716, -0.42259641,  0.63713844],
                       [-0.44674054, -1.20586259, -0.54179732]]), 
                array([[-0.15468705,  0.3472812 ],
                       [ 1.0628925 ,  1.06916121],
                       [ 0.18613386,  2.01445385]])
            ]
        """
        self.weights = self.__initialize_weights()

        # macierz 3D: [] -> warstwa[ndarray] -> biases[]
        """
            [
                array([[-1.39059735, -1.34781964, -0.26771641]]), 
                array([[-0.12038373,  1.24983815]])
            ]
        """
        self.biases = self.__initialize_biases()

    def __initialize_weights(self):
        weights = []

        for i in range(len(self.layers_conf) - 1):
            n_in = self.layers_conf[i]['neurons']
            n_out = self.layers_conf[i + 1]['neurons']

            w = np.random.normal(0, 1, (n_in, n_out))
            weights.append(w)

        return weights

    # TODO jaki format? (n, 1) czy (1, n)
    def __initialize_biases(self):
        biases = []
        for i in range(len(self.layers_conf) - 1):
            # n_in = self.layers_conf[i]['neurons']
            n_out = self.layers_conf[i + 1]['neurons']

            b = np.random.normal(0, 1, (1, n_out))
            biases.append(b)

        return biases

    def __update_weights(self, data, weights, delta):
        return weights + delta * self.l_rate * data

    def get_parameters(self):
        return [self.n_epoch, self.l_rate, self.layers_amount]

    def visualization(self):
        lines = []

        for i, layer in enumerate(self.layers_conf):
            neurons = layer['neurons']
            activation = layer['activation']

            if i == 0:
                box = f"[ Warstwa wejściowa ({neurons}) ]"
            elif i == self.layers_amount - 1:
                box = f"[ Warstwa wyjściowa ({neurons}) | {activation} ]"
            else:
                box = f"[ Warstwa ukryta ({neurons}) | {activation} ]"

            lines.append(box)

            if i < len(self.layers_conf) - 1:
                lines.append("      |")
                lines.append("      ▼")

        return "\n".join(lines)

    def fit(self, X, y, error_threshold=0.0001):
        for epoch in range(self.n_epoch):
            errors = []
            output = self.forward_propagate(self.dataset_inputs)
            self.layer_outputs = self.layer_inputs.copy()
            self.layer_outputs.append(output)
            self.backward_propagate_error(self.l_rate)

            # if epoch % 2000 == 0:
            #     error = -np.mean(np.log(output[np.arange(len(self.dataset_outputs)), np.argmax(self.dataset_outputs)]))
            #     print('[INFO] epoch=%d, error=%.4f' % (epoch, error))
            #
            #     if error < error_threshold:
            #         print('[INFO] Training stopped. Error is below the threshold.')
            #         break

    def forward_propagate(self, data):
        layer_input = data
        self.layer_inputs = [layer_input]

        n_hidden_layers = len(self.weights) - 1
        for i in range(n_hidden_layers):
            # вираховується зважена сума до прошарків
            layer_output = np.dot(layer_input, self.weights[i]) + self.biases[i]
            layer_input = self.activation_function(layer_output)
            self.layer_inputs.append(layer_input)

        # вихід з останнього прихованого шару з функцією активації softmax
        output = np.dot(layer_input, self.weights[-1]) + self.biases[-1]
        # output = np.clip(output, -700, 700)
        output = self.softmax(output)
        return output

    def predict(self, X):
        return np.argmax(self.forward_propagate(X), axis=1)
