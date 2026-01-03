import pandas as pd
from sklearn.model_selection import train_test_split

class DataReader:
    def __init__(self, file_name, target):
        self.file_path = 'data/' + file_name
        self.target = target

    def read_data(self):
        df = pd.read_csv(self.file_path)

        y = df[self.target].values
        x = df.drop(columns=[self.target]).values

        X_train, X_tmp, Y_train, Y_tmp = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
        X_val, X_test, Y_val, Y_test = train_test_split(X_tmp, Y_tmp, test_size=0.5, random_state=42, stratify=Y_tmp)

        return X_train, X_val, X_test, Y_train, Y_val, Y_test

