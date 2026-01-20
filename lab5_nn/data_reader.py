"""
Autor: Viktoriia Nowotka, Karol Łukasik
"""

import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


class DataReader:
    def __init__(self, file_name, target, augmentation=False, random_state=42):
        self.file_path = "data/" + file_name
        self.target = target
        self.augmentation = augmentation
        self.random_state = random_state
        self.encoder = OneHotEncoder(sparse_output=False)

    def read_data(self):
        df = pd.read_csv(self.file_path)

        x = df.drop(columns=[self.target]).values
        y_raw = df[self.target].values.ravel()

        if self.augmentation:
            smote = SMOTE(random_state=self.random_state)
            x, y_raw = smote.fit_resample(x, y_raw)

            print("Class distribution AFTER augmentation:",
                  Counter(y_raw))

        y_encoded = self.encoder.fit_transform(y_raw.reshape(-1, 1))

        X_train, X_tmp, Y_train, Y_tmp = train_test_split(
            x, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )
        X_val, X_test, Y_val, Y_test = train_test_split(
            X_tmp, Y_tmp, test_size=0.5, random_state=42, stratify=Y_tmp
        )

        return X_train, X_val, X_test, Y_train, Y_val, Y_test
