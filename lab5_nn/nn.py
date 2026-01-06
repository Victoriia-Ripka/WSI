"""
Autor: Viktoria Nowotka, Karol Łukasik
"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from solver import Solver

class NeuralNetwork(Solver):
    def __init__(self, params, n_epoch, l_rate):
        self.encoder = OneHotEncoder(sparse_output=False)

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

            w = np.random.normal(0, np.sqrt(2 / n_in), (n_in, n_out))
            weights.append(w)

        return weights

    def __initialize_biases(self):
        biases = []
        for i in range(len(self.layers_conf) - 1):
            n_out = self.layers_conf[i + 1]['neurons']
            biases.append(np.zeros((1, n_out)))
        return biases
    
    def __update_weights(self, grads_w, grads_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.l_rate * grads_w[i]
            self.biases[i] -= self.l_rate * grads_b[i]

    def get_parameters(self):
        struct = self.get_structure()
        return {"n_epoch": self.n_epoch, "l_rate": self.l_rate, "struct": struct}

    def get_structure(self):
        struct = {}

        for i, layer in enumerate(self.layers_conf):
            struct[i] = layer['neurons']

        return struct

    def visualization(self):
        lines = []

        for i, layer in enumerate(self.layers_conf):
            neurons = layer['neurons']
            activation = layer['activation']

            if i == 0:
                box = f"[ Warstwa wejściowa ({neurons}) | {activation.__name__} ]"
            elif i == self.layers_amount - 1:
                box = f"[ Warstwa wyjściowa ({neurons}) | {activation.__name__} ]"
            else:
                box = f"[ Warstwa ukryta ({neurons}) | {activation.__name__} ]"

            lines.append(box)

            if i < len(self.layers_conf) - 1:
                lines.append("      |")
                lines.append("      ▼")

        return "\n".join(lines)

    def fit(self, X, X_val, y, y_val, best_loss=np.inf, patience=5):
        for epoch in range(self.n_epoch):
            # TRAIN
            y_pred = self.forward_propagate(X)
            grads_w, grads_b = self.backward_propagate(y)
            self.__update_weights(grads_w, grads_b)

            # VALIDATION
            loss_val = self.calculate_loss(X_val, y_val)

            # EARLY STOPPING
            if loss_val < best_loss:
                best_loss = loss_val
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.n_epoch = epoch + 1
                    print(f"Wczesne zatrzymanie uczenia (epoch {epoch}). \nloss={loss_val:.3f}, delta loss={loss_val-best_loss:.6f}")
                    break

            # LOG
            if epoch % 1000 == 0:
                loss = -np.mean(np.sum(y * np.log(y_pred + 1e-9), axis=1))
                print(f'Epoch {epoch}, Loss: {loss:.4f}')

    def calculate_loss(self, X_val, y_val):
        y_pred = self.forward_propagate(X_val)
        loss = -np.mean(np.sum(y_val * np.log(y_pred + 1e-9), axis=1))
        return loss

    def calculate_accuracy(self, X, y):
        y_pred = self.predict(X)
        y_labels = np.argmax(y, axis=1)
        return np.mean(y_pred == y_labels)

    def forward_propagate(self, X):
        self.layer_inputs = [] #"Z values"
        self.layer_activations = [X] #"A values"
        current_input = X

        for i in range(len(self.weights)):
            Z = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.layer_inputs.append(Z)
            act_func = self.layers_conf[i+1]['activation']
            if act_func.__name__ == 'softmax': #albo dodać parametr derive do softmax
                A = act_func(Z)
            else:
                A = act_func(Z, derive=False)
            self.layer_activations.append(A)
            current_input = A
        return current_input
    
    def backward_propagate(self, y_true):
        m = y_true.shape[0]
        gradients_w = []
        gradients_b = []
        A_last = self.layer_activations[-1]
        delta = A_last - y_true #dC/dZ = dC/dA * dA/dZ (softmax + cross-entropy), jak zmienimy funkcje straty to trzeba bedzie to zmienic
        for i in range(len(self.weights) - 1, -1, -1):
            A_prev = self.layer_activations[i]
            dW = np.dot(A_prev.T, delta) / m #dz/dW
            db = np.sum(delta, axis=0, keepdims=True) / m
            gradients_w.insert(0, dW)
            gradients_b.insert(0, db)
            if i > 0:
                W_curr = self.weights[i]
                Z_prev = self.layer_inputs[i-1]
                prev_act_func = self.layers_conf[i]['activation']
                derivative = prev_act_func(Z_prev, derive=True)
                delta = np.dot(delta, W_curr.T) * derivative
        return gradients_w, gradients_b

    def recall_per_class(self, y, y_pred_labels, n_classes, eps=1e-9):
        y_true_labels = np.argmax(y, axis=1)

        recalls = {}

        for c in range(n_classes):
            TP = np.sum((y_true_labels == c) & (y_pred_labels == c))
            FN = np.sum((y_true_labels == c) & (y_pred_labels != c))

            recalls[c] = TP / (TP + FN + eps)

        return recalls

    def predict(self, X):
        return np.argmax(self.forward_propagate(X), axis=1)
