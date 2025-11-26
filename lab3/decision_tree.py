# Viktoriia Nowotka
import numpy as np
from lab3.solver import Solver
from lab3.node import Node
from typing import Any, Dict


# TODO Implementacja powinna obsługiwać sytuację, w której w zbiorze trenującym nie ma wszystkich wartości jakiegoś atrybutu
# -> Analogicznie jak przy braku atrybutów do podziału, tworzony jest liść zawierający najczęstszą klasę w pozostałym zbiorze.
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

    def id3(self, c, s, r):
        """ Rekurencyjna metoda algorytmu ID3 indukujący drzewo decyzyjne.
        Algorytm stara się znaleźć podział danych, który minimalizuje entropię, co oznacza, że grupy danych są bardziej jednorodne w kontekście przynależności do danej klasy

        Args:
            c: zbiór klas [0 1]
            r: # zbiór atrybutów poza klasą [0-10]
            s: # zbiór objektów [{dane}]

        Returns:
            Drzewo decyzyjne z korzeniem oznaczonym przed D i krawędziami d_j
        """

        # warunki stopu
        if len(s) == 0:
            # pusty podzbiór
            # TODO w przypadku pustego S, najlepiej zwrócić klasę z poprzedniego kroku
            raise ValueError("brak objektów")

        if len(r) == 0:
            # brak atrybutów do podziału
            label = self._get_most_common_class(s)
            return Node(label)

        if self._is_homogeneous(s):
            # etykiety liści to ta sama klasa
            label = self.y_train[s][0]
            return Node(label)

        # wybór najlepszego atrybutu i tworzenie węzła wewnętrznego
        best_feature_idx = self._find_best_split(r, s)
        best_feature_name = self.feature_names[best_feature_idx]
        current_node = Node(feature=best_feature_name)

        # aktualizacja dostępnych atrybotów
        r_new = r[r != best_feature_idx]

        feature_values_subset = self.X_train[s, best_feature_idx]

        # Iteracja po unikalnych wartościach atrybutu (to będą krawędzie)
        for value in np.unique(feature_values_subset):

            # Znajdź maskę boolowską obiektów w S, które mają wartość 'value' dla tego atrybutu
            mask = (self.X_train[s, best_feature_idx] == value)

            # Stwórz nowy podzbiór indeksów S_child (s_child_indices)
            s_child_indices = s[mask]

            # ⚠️ Warunek bezpieczeństwa: Jeśli podzbiór jest pusty, pomiń gałąź lub zaimplementuj handling.
            # W ID3 zazwyczaj się to nie dzieje, jeśli atrybut został wybrany na podstawie danych S.
            if len(s_child_indices) == 0:
                # Można np. stworzyć liść z najczęściej występującą klasą w zbiorze macierzystym S
                continue

            # Wywołanie rekurencyjne: zbuduj poddrzewo dla gałęzi
            # Wynik jest nowym węzłem potomnym
            child_node = self.id3(c, s_child_indices, r_new)

            # Dodaj potomka do bieżącego węzła (jako krawędź w słowniku children)
            # Klucz = wartość atrybutu (krawędź), Wartość = węzeł potomny
            current_node.children[value] = child_node

        return current_node

    # Nieczystość Giniego skupia się na minimalizacji błędów klasyfikacji, podczas gdy entropia mierzy stopień nieporządku w danych.
    # Entropia mierzy jak bardzo dane są pomieszane lub nieuporządkowane.
    def _calculate_entropy(self, s_indices):
        """ I(S) = - suma [ P(c|S) * log2(P(c|S)) ]  """
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

    # A jak porównujemy i szukamy najlepszego podziału naszej zmiennej (lub zmiennych, jeśli mamy ich więcej)? Robimy to poprzez obliczenie nieczystości Giniego dla każdej z tych części i sumujemy je, uwzględniając proporcję elementów w każdej z nich. Następnie wybieramy podział, który minimalizuje tę sumę, co oznacza, że najlepiej segreguje dane według klasy.
    # Uwaga. Warto pamiętać, że jeśli nieczystość Giniego dla dwóch węzłów podrzędnych nie jest niższa niż nieczystość Giniego dla węzła nadrzędnego, to algorytm przestanie szukać podziałów.
    # 0.0-0.5
    def _calculate_inf_gain(self, s_indices, feature_index):
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

        if best_feature_index >= self.X_train.shape[1]:
            print("error: feature index ", best_feature_index, " is out of range")

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

    def print_tree(self, node: Node, depth=0):
        for i in range(self.max_depth):
            print("\t", end="")
        print(node.feature, end="")
        if node.is_leaf():
            print(" -> ", node.label)
        print()
        for child in node.children:
            self.print_tree(child, depth + 1)

    def get_parameters(self) -> Dict[str, Any]:
        return {'max_depth': self.max_depth, 'feature_names': self.feature_names}

    def fit(self):
        c = np.unique(self.y_train)
        r_initial = np.arange(self.X_train.shape[1])
        s_initial = np.arange(self.X_train.shape[0])
        self.tree = self.id3(c, s_initial, r_initial)

    def test(self):
        correct_preditct = 0
        wrong_preditct = 0

        for index, row in self.X_test.iterrows():
            result = self.solve(self.X_test.iloc[index])  # predict the row
            if result == self.y_test.iloc[index]:  # predicted value and expected value is same or not
                correct_preditct += 1
            else:
                wrong_preditct += 1
        accuracy = correct_preditct / (correct_preditct + wrong_preditct)
        return accuracy

    def solve(self, x, *args, **kwargs):
        for node in self.tree.children:
            if node.value == x[self.tree.value]:
                if node.is_leaf():
                    return node.label
                else:
                    self.solve(node.children[0], x)
        return None

