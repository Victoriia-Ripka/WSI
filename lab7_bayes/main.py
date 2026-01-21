"""
Author: Viktoriia Nowotka
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from lab7_bayes.bayes import Bayes

def do_charts():
    pass


def work_with_data(filename, target):
    df = pd.read_csv(filename)

    y = df[target].values
    X = df.drop(target, axis=1)

    return X, y


def k_cross_validation(X, y, n_folds=4):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = Bayes()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        scores.append(acc)

    return np.mean(scores), scores


def main():
    filename = 'data/spam.csv'
    target = 'C1' # v1

    X, y = work_with_data(filename, target)

    mean_acc, all_acc = k_cross_validation(X, y, n_folds=4)

    print("Accuracy per fold:", all_acc)
    print("Mean accuracy:", mean_acc)


if __name__ == "__main__":
    main()
