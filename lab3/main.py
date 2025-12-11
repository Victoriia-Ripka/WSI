# Viktoriia Nowotka
import numpy as np
import pandas as pd
from lab3.decision_tree import DecisionTree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def format_data(df):
    df['age_years'] = df['age'] / 365
    labels_age = ['baby', 'teenager', 'young', 'middle_aged', 'senior']
    age_mapping = {'baby': 0, 'teenager': 1, 'young': 2, 'middle_aged': 3, 'senior': 4}
    bins_age = [0, 12, 25, 45, 60, df['age_years'].max()]
    df['age_group'] = pd.cut(df['age_years'], bins=bins_age, labels=labels_age, right=True, include_lowest=True)
    df['age_numeric'] = df['age_group'].map(age_mapping)

    labels_ap = ['normal', 'elevated', 'hypertensive_crisis']
    ap_mapping = {'normal': 0, 'elevated': 1, 'hypertensive_crisis': 2}
    bins_ap_hi = [0, 120, 140, df['ap_hi'].max()]
    df['ap_hi_cat'] = pd.cut(df['ap_hi'], bins=bins_ap_hi, labels=labels_ap, right=True, include_lowest=True)
    df['ap_hi_numeric'] = df['ap_hi_cat'].map(ap_mapping)
    bins_ap_lo = [0, 70, 90, df['ap_lo'].max()]
    df['ap_lo_cat'] = pd.cut(df['ap_lo'], bins=bins_ap_lo, labels=labels_ap, right=True, include_lowest=True)
    df['ap_lo_numeric'] = df['ap_lo_cat'].map(ap_mapping)

    labels_height = ['short', 'normal', 'tall']
    height_mapping = {'short': 0, 'normal': 1, 'tall': 2}
    bins_height = [0, 150, 180, df['height'].max()]
    df['height_group'] = pd.cut(df['height'], bins=bins_height, labels=labels_height, right=True, include_lowest=True)
    df['height_numeric'] = df['height_group'].map(height_mapping)

    labels_weight = ['thin', 'normal', 'fat']
    weight_mapping = {'thin': 0, 'normal': 1, 'fat': 2}
    bins_weight = [0, 50, 80, df['weight'].max()]
    df['weight_group'] = pd.cut(df['weight'], bins=bins_weight, labels=labels_weight, right=True, include_lowest=True)
    df['weight_numeric'] = df['weight_group'].map(weight_mapping)

    df.drop(['age', 'age_years', 'age_group', 'ap_hi', 'ap_lo', 'ap_hi_cat', 'ap_lo_cat', 'weight', 'weight_group',
             'height', 'height_group'], axis=1, inplace=True)

    return df


def do_charts(depths, validation_accuracies):
    plt.figure(figsize=(12, 7))
    plt.plot(depths, validation_accuracies, label='Dokładność Validacyjna', marker='o', linestyle='-')

    plt.title('Zależność Dokładności od Maksymalnej Głębokości Drzewa (ID3)')
    plt.xlabel('Maksymalna Głębokość Drzewa')
    plt.ylabel('Dokładność')

    optimal_depth = depths[np.argmax(validation_accuracies)]
    max_test_acc = max(validation_accuracies)

    plt.axvline(x=optimal_depth, color='r', linestyle='-.', linewidth=1)

    plt.legend()
    plt.grid(True, linestyle='--')
    plt.xticks(depths)
    plt.savefig(f"Zależność Dokładności od Maksymalnej Głębokości Drzewa.png")
    plt.show()

    print(f"\n\nOptymalna głębokość na zbiorze testowym: {optimal_depth} (Dokładność: {max_test_acc:.4f})")


def main():
    df = pd.read_csv('data/cardio_train.csv', sep=';')
    df.drop('id', axis=1, inplace=True)

    # 69 301 par uczących
    df = format_data(df)
    y = df['cardio'].values
    feature_names = df.drop('cardio', axis=1).columns.tolist()
    x = df.drop('cardio', axis=1).values

    X_train, X_tmp, Y_train, Y_tmp = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, Y_val, Y_test = train_test_split(X_tmp, Y_tmp, test_size=0.5, random_state=42, stratify=Y_tmp)

    tree_depths = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    validation_accuracies = []

    for tree_depth in tree_depths:
        dt = DecisionTree(tree_depth, feature_names, X_train, X_val, Y_train, Y_val)
        dt.get_parameters()
        dt.fit()
        # dt.print_tree()
        validation_accuracy = dt.test()
        validation_accuracies.append(validation_accuracy)
        print(f"Głębokość {tree_depth}: Test={validation_accuracy:.4f}")

    do_charts(tree_depths, validation_accuracies)

    best_depth = tree_depths[np.argmax(validation_accuracies)]
    dt = DecisionTree(best_depth, feature_names, X_train, X_test, Y_train, Y_test)
    dt.get_parameters()
    dt.fit()
    predicts = []
    for x in X_test:
        predicts.append(dt.predict(x))

    correct_predictions = (Y_test == predicts)
    num_correct = np.sum(correct_predictions)
    predict_accuracy = num_correct / len(Y_test)

    print(f"Przewiduj dokładność dla ID3 z głębokością {best_depth}: {predict_accuracy:.4f}")


if __name__ == "__main__":
    main()
