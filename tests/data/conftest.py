import great_expectations as ge
import pandas as pd
import pytest

def pytest_addoption(parser):
    """Add options to specify dataset locations when executing tests from CLI."""
    parser.addoption(
        "--train-dataset-loc", action="store", default=None, help="Train dataset location."
    )
    parser.addoption(
        "--test-dataset-loc", action="store", default=None, help="Test dataset location."
    )

@pytest.fixture(scope="module")
def train_df(request):
    """Fixture for train dataset."""
    train_dataset_loc = request.config.getoption("--train-dataset-loc")
    train_df = ge.dataset.PandasDataset(pd.read_csv(train_dataset_loc))
    return train_df

@pytest.fixture(scope="module")
def test_df(request):
    """Fixture for test dataset."""
    test_dataset_loc = request.config.getoption("--test-dataset-loc")
    test_df = ge.dataset.PandasDataset(pd.read_csv(test_dataset_loc))
    return test_df
