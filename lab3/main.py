# Viktoriia Nowotka
import csv
import numpy as np
import pandas as pd
from lab3.decision_tree import DecisionTree
from sklearn.model_selection import train_test_split

# 69 301 par uczących

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

def main():
    np.set_printoptions(suppress=True)

    df = pd.read_csv('data/cardio_train.csv', sep=';')
    df.drop('id', axis=1, inplace=True)

    df = format_data(df)
    # print(df.iloc[0])
    small_df = df.iloc[0:20]

    y = small_df['cardio'].values
    feature_names = small_df.drop('cardio', axis=1).columns.tolist()
    x = small_df.drop('cardio', axis=1).values

    # TODO Należy znaleźć taką wartość parametru maksymalnej głębokosci, która da najlepszy wynik.
    tree_depth = 4

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # , stratify=y

    dt = DecisionTree(tree_depth, feature_names, X_train, X_test, y_train, y_test)
    # dt.get_parameters()
    dt.fit()
    print(y_train)
    print(dt.tree)
    # dt.print_tree()
    # accuracy = dt.test()
    # print(f"test accuracy: {accuracy}")
    # dt.solve()



#     Potrzebny jest trzeci zbiór do oceny jakości wybranego modelu
# Brakujące wartości?
# Co zrobić, jeżeli w danych pojawi się wartość, której nie było w zbiorze
# trenującym?


if __name__ == "__main__":
    main()
