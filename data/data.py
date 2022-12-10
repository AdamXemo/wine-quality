# Libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np


class data:
    @staticmethod
    def load_format_split(path):
        # Data loading
        wine = pd.read_csv(path, sep=',')
        # A couple of our elements
        wine.head()
        # Dtype, memory, isnull
        # wine.info()
        # How much null values we have
        wine.isnull().sum()

        # Preprocessing data

        # 2 - 'bad' and 'good', 6 - bellow this wine is bad, 8 - cuz we are doing 0-8 quality
        bins = (2, 6, 8)
        # 2 different quality labels
        group_names = ['bad', 'good']
        # Applying what we wrote above
        wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)
        wine['quality'].unique()
        # print(wine['quality'])


        # Bad = 0, Good = 1
        label_quality = LabelEncoder()
        # Transforms wine quality to out variable above
        wine['quality'] = label_quality.fit_transform(wine['quality'])
        # print(wine['quality'])

        # Amount of each value (bad(0) - 1382, good(1) - 217)
        wine['quality'].value_counts()

        # Seperate dataset as response variable and feature variables

        # All the features except quality
        X = wine.drop('quality', axis=1)
        # Only quality
        y = wine['quality']

        # Train and test splitting                          X, y,  20% of data   random seed num   
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Applying standard scaling to get optimized result
        sc = StandardScaler()
        # Transforming training data
        X_train = sc.fit_transform(X_train)
        # And testing data aswell
        X_test = sc.fit_transform(X_test)

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        return X_train, y_train, X_test, y_test

class datasets:
    @staticmethod
    def wine():
        return 'data/winequality-red.csv'

class accuracy:
    # Calculating how much wrong & right predictions
    @staticmethod
    def calculate(predictions, y_test):
        right, wrong = 0, 0
        for pred, y in zip(predictions, y_test):
            if pred == y:
                right += 1
            else:
                wrong += 1

        # Calculating our accuracy (76.6)
        accuracy = "{:.1f}".format(right/(right+wrong)*100)

        return accuracy