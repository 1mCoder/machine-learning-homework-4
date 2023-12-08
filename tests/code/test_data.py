import pandas as pd
from src.code.data import get_train_data
from src.code.data import get_test_data

# Test case for loading test data
def test_get_test_data():
    test_data = get_test_data()
    assert isinstance(test_data, pd.DataFrame)

# Test case for loading train data
def test_get_train_data():
    train_data = get_train_data()
    assert isinstance(train_data, pd.DataFrame)