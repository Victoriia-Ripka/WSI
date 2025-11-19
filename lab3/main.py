# Viktoriia Nowotka
import csv
import numpy as np
from lab3.decision_tree import DecisionTree

# 69 301 par uczących

def main():
    np.set_printoptions(suppress=True)

    tree_depth = 4
    feature_names = ['age', 'height', 'weight', 'gender', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    dt = DecisionTree(tree_depth, feature_names)

    data = np.genfromtxt('data/cardio_train.csv', delimiter=';', skip_header=1)
    x = data[:, :-1]
    y = data[:, -1]
    c = np.unique(y)
    r = np.arange(x.shape[1])
    s = np.arange(x.shape[0])

    dt.get_parameters()

    tree = dt.id3(x, y, c, r, s)
    print(tree)



if __name__ == "__main__":
    main()