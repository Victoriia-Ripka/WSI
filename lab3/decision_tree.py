# Viktoriia Nowotka
from lab3.solver import Solver
from typing import Any, Callable, Dict, Tuple

class DecisionTree(Solver):
    def __init__(self, depth, feature_names):
        """ Klas drzewa decyzyjnego """
        self.max_depth = depth
        self.feature_names = feature_names
        self.tree = None

    def get_parameters(self) -> Dict[str, Any]:
        return {'max_depth': self.max_depth, 'feature_names': self.feature_names, 'tree': self.tree}

    def fit(self, X, Y):
        pass

    def id3(self, X, Y, c, r, s):
        """ Rekurencyjna metoda algorytmu ID3 indukujący drzewo decyzyjne.

        Args:
            X: macierz danych (cechy)
            Y: wektor klas (etykiety)
            c: lista unikalnych klas [0,1]
            r: lista dostępnych do podziału indeksów atrybutów {age, height, weight, gender, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active}
            s: zbiór obiektów [{dane}]

        Returns:
            Drzewo decyzyjne z korzeniem oznaczonym przed D i krawędziami d_j
        """
        if not s:
            return TypeError

        # 1. Sprawdź warunki stopu (czysty zbiór, brak atrybutów r, brak obiektów s)
        # ...

        # 2. Wybierz najlepszy atrybut do podziału 'A' z listy 'r'
        # Obliczenie Entropy, Information Gain, itd.
        # Dla każdego atrybutu 'i' in 'r', używasz X[s, i]

        # 3. Usuń wybrany atrybut 'A' z 'r'
        # ...

        # 4. Rekurencyjne wywołania dla podziałów
        # ...

    def test(self):
        pass

    def solve(self, X, *args, **kwargs):
        pass

    def _calculate_entropy(self, Y_subset):
        """Oblicza Entropię Shannona dla danego podzbioru etykiet klas"""
        pass

    def _calculate_information_gain(self, X_subset, Y_subset, feature_index):
        """Oblicza Zysk Informacyjny (Gain) dla danego atrybutu"""
        pass

    def _find_best_split(self, X_subset, Y_subset, available_features):
        """Iteruje po wszystkich dostępnych atrybutach i zwraca indeks atrybutu, który maksymalizuje Zysk Informacyjny."""
        pass

    def _traverse_tree(self, sample, node):
        """ Rekurencyjnie przechodzi przez strukturę drzewa (self.tree) dla pojedynczego obiektu wejściowego (sample), aż dotrze do węzła liścia

            Args:
                sample: lista danych
                node: lista danych

            Return:
                etykietę klasy z węzła liścia
        """
        pass

    def __str__(self):
        pass

