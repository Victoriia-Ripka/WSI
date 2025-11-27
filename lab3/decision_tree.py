# Viktoriia Nowotka
import numpy as np
from lab3.solver import Solver
from lab3.node import Node
from typing import Any, Dict


class DecisionTree(Solver):
    def __init__(self, depth, feature_names, X_train, X_test, y_train, y_test):
        """ Klas drzewa decyzyjnego """
        self.max_depth = depth

        self.feature_names = feature_names # ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'age_numeric', 'ap_hi_numeric', 'ap_lo_numeric', 'height_numeric', 'weight_numeric']

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.tree = None # przechowuje korzeń zwrócony przez id3
        self.global_modal_class = None

        print(self.feature_names)

    def id3(self, c, s, r):
        """ Rekurencyjna metoda algorytmu ID3 indukujący drzewo decyzyjne.
        Algorytm stara się znaleźć podział danych, który minimalizuje entropię, co oznacza, że grupy danych są bardziej jednorodne w kontekście przynależności do danej klasy

        Args:
            c: zbiór klas [0 1]
            r: # zbiór atrybutów poza klasą [0-10]
            s: # zbiór objektów [{dane}]
            default_class: klasa do zwrócenia w przypadku pustego s

        Returns:
            Drzewo decyzyjne z korzeniem oznaczonym przed D i krawędziami d_j
        """
        if len(s) > 0:
            current_modal_class = self._get_most_common_class(s)
        else:
            # print("ID3 stop: pusty podzbiór")
            # w przypadku pustego s -> zwróć klasę z poprzedniego kroku
            return Node(label=self.global_modal_class)

        if len(r) == 0:
            # print("ID3 stop: brak atrybutów do podziału")
            return Node(current_modal_class)

        if self._is_homogeneous(s):
            # print("ID3 stop: etykiety liści to ta sama klasa")
            label = self.y_train[s][0]
            return Node(label)

        # wybór najlepszego atrybutu i tworzenie węzła wewnętrznego
        best_feature_idx = self._find_best_split(r, s)
        best_feature_name = self.feature_names[best_feature_idx]
        current_node = Node(feature=best_feature_name, modal_class=current_modal_class)

        r_new = r[r != best_feature_idx]

        feature_values_subset = self.X_train[s, best_feature_idx]

        # iteracja po unikalnych wartościach atrybutu (krawędzie węzła)
        for value in np.unique(feature_values_subset):

            # Znajdź maskę boolowską obiektów w S, które mają wartość 'value' dla tego atrybutu
            mask = (self.X_train[s, best_feature_idx] == value)

            # Stwórz nowy podzbiór indeksów S_child (s_child_indices)
            s_child_indices = s[mask]

            child_node = self.id3(c, s_child_indices, r_new)

            # krawędź = wartość atrybutu, wartość = węzeł potomny
            current_node.children[value] = child_node

        return current_node

    def _calculate_entropy(self, s_indices):
        """ I(S) = - suma [ P(c|S) * log2(P(c|S)) ]  """
        # Nieczystość Giniego skupia się na minimalizacji błędów klasyfikacji, podczas gdy entropia mierzy stopień nieporządku w danych.
        # Entropia mierzy jak bardzo dane są pomieszane lub nieuporządkowane.
        if len(s_indices) == 0:
            return 0.0

        # 1. Wyizolowanie etykiet klas dla podzbioru S
        y_subset = self.y_train[s_indices]

        # 2. Obliczenie liczności (counts) dla każdej klasy
        # classes i counts są równoległymi tablicami
        classes, counts = np.unique(y_subset, return_counts=True)

        # 3. Obliczenie prawdopodobieństwa P(c|S) (częstotliwość występowania fc(S))
        # Dzielimy liczność każdej klasy przez całkowitą liczbę obiektów w podzbiorze S
        total_samples = len(s_indices)
        probabilities = counts / total_samples

        # 4. Obliczenie Entropii
        entropy = 0.0
        for p in probabilities:
            # Pamiętaj, że log2(0) jest nieokreślone, ale lim p*log2(p) przy p->0 jest 0.
            # Dlatego pomijamy przypadki, gdy p jest bardzo bliskie 0.
            if p > 0:
                # Używamy logarytmu o podstawie 2, typowego dla entropii Shannona.
                entropy -= p * np.log2(p)

        return entropy

    def _calculate_inf_ds(self, s_indices, feature_index):
        total_samples = len(s_indices)
        weighted_entropy_sum = 0.0

        for value in np.unique(self.X_train[s_indices, feature_index]):
            # podzbiór S_j
            s_j_indices = s_indices[self.X_train[s_indices, feature_index] == value]

            # |S_j| / |S|
            weight = len(s_j_indices) / total_samples

            # I(S_j)
            entropy_sj = self._calculate_entropy(s_j_indices)

            weighted_entropy_sum += weight * entropy_sj

        return weighted_entropy_sum

    def _calculate_inf_gain(self, s_indices, feature_index):
        # A jak porównujemy i szukamy najlepszego podziału naszej zmiennej (lub zmiennych, jeśli mamy ich więcej)? Robimy to poprzez obliczenie nieczystości Giniego dla każdej z tych części i sumujemy je, uwzględniając proporcję elementów w każdej z nich. Następnie wybieramy podział, który minimalizuje tę sumę, co oznacza, że najlepiej segreguje dane według klasy.
        # Uwaga. Warto pamiętać, że jeśli nieczystość Giniego dla dwóch węzłów podrzędnych nie jest niższa niż nieczystość Giniego dla węzła nadrzędnego, to algorytm przestanie szukać podziałów.
        # 0.0-0.5
        entropy_parent = self._calculate_entropy(s_indices)
        inf_ds = self._calculate_inf_ds(s_indices, feature_index)
        information_gain = entropy_parent - inf_ds
        return information_gain

    def _find_best_split(self, r_indices, s_indices):
        best_gain = -1.0
        best_feature_index = -1

        for feature_idx in r_indices:
            gain = self._calculate_inf_gain(s_indices, feature_idx)

            if gain > best_gain:
                best_gain = gain
                best_feature_index = feature_idx

        return best_feature_index

    def _get_most_common_class(self, s_indices: np.ndarray) -> int:
        y_subset = self.y_train[s_indices]
        classes, counts = np.unique(y_subset, return_counts=True)
        max_count_index = np.argmax(counts)
        most_common_class = classes[max_count_index]

        return most_common_class

    def _is_homogeneous(self, s_indices: np.ndarray)  -> bool:
        y_subset = self.y_train[s_indices]
        unique_classes = np.unique(y_subset)
        return len(unique_classes) == 1

    def print_tree(self):
        if self.tree is None:
            print("empty tree")
            return
        self._print_node(self.tree, depth=0, edge_value="Korzeń")

    def _print_node(self, node, depth, edge_value):
        indentation = "  |  " * (depth - 1)
        if depth > 0:
            print(f"{indentation}  +--- [{edge_value}] ---> ", end="")
        else:
            print(f"Korzeń: ", end="")

        print(node)
        if not node.is_leaf():
            for value, child_node in node.children.items():
                self._print_node(child_node, depth + 1, value)

    def get_parameters(self) -> Dict[str, Any]:
        return {'max_depth': self.max_depth, 'feature_names': self.feature_names}

    def fit(self):
        c = np.unique(self.y_train)
        r_initial = np.arange(self.X_train.shape[1])
        s_initial = np.arange(self.X_train.shape[0])
        self.global_modal_class = self._get_most_common_class(s_initial)
        self.tree = self.id3(c, s_initial, r_initial)

    def test(self):
        correct_predict = 0
        wrong_predict = 0

        for index, row in enumerate(self.X_test):
            result = self.predict(row)
            if result == self.y_test[index]:
                correct_predict += 1
            else:
                wrong_predict += 1
        accuracy = correct_predict / (correct_predict + wrong_predict)
        return accuracy

    def predict(self, x):
        return self._traverse_tree(self.tree, x)

    def _traverse_tree(self, current_node, x):
        if current_node.is_leaf():
            return current_node.label

        feature_name = current_node.feature
        try:
            feature_index = self.feature_names.index(feature_name)
        except:
            raise ValueError(f"brak atrybutu '{feature_name}' w liście nazw cech")

        attribute_value_in_x = x[feature_index]

        if attribute_value_in_x in current_node.children:
            next_node = current_node.children[attribute_value_in_x]
            return self._traverse_tree(next_node, x)

        else:
            # Implementacja powinna obsługiwać sytuację, w której w zbiorze trenującym nie ma wszystkich wartości jakiegoś atrybutu
            print(f"error: brak gałęzi")
            return current_node.modal_class

