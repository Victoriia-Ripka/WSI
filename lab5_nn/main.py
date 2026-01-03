"""
W szczególności: możliwość zastosowania rozwiązania do dowolnego zbioru danych o wartościach
rzeczywistych, parametryzowalna liczba warstw i neuronów w każdej warstwie, możliwość wyboru
różnych funkcji straty, aktywacji oraz algorytmu optymalizacyjnego.

Dla chętnych:
Zaproponuj sposób poprawy jakości klasyfikacji dla najmniej licznych klas w zbiorze danych.
Porównaj ogólną jakość klasyfikacji oraz jakość klasyfikacji najmniej licznych klas z siecią
przygotowaną w ramach ćwiczenia.
"""
from lab5_nn.data_reader import DataReader


def main():
    file = 'data.csv'
    target = 'quality'

    fr = DataReader(file, target)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = fr.read_data()





if __name__ == "__main__":
    main()