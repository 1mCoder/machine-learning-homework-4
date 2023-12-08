# Import necessary modules
import pytest

@pytest.fixture(scope="session", autouse=True)
def tmp_path():
    return "tmp"

# Fixture to initialize nltk downloads
@pytest.fixture(scope="session", autouse=True)
def initialize_nltk_downloads():
    import nltk

    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
