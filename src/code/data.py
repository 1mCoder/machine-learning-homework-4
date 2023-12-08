import pandas as pd

def get_test_data():
    return pd.read_csv('../../kmaml223/test.csv')

def get_train_data():
    return pd.read_csv('../../kmaml223/train.csv')