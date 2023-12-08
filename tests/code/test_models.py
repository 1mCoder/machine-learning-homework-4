import joblib
from sklearn.linear_model import LogisticRegression
from src.code.baseline import train_baseline 
from src.code.logreg_acc import train_logistic_regression_with_accuracy 
from src.code.logreg_f1 import train_logistic_regression_with_f1

def test_train_baseline_models_and_tfidf(tmp_path):
    train_baseline()  # Execute the function

    # Define the expected directory path
    folder = tmp_path / "baseline"
    expected_tfidf_filename = folder / "tfidf_vectorizer.pkl"

    # Check if the directory and files exist
    assert folder.exists()
    assert expected_tfidf_filename.exists()

    # Check if models for each feature exist
    feature_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    for feature in feature_columns:
        expected_model_filename = folder / f"{feature}_model.pkl"
        assert expected_model_filename.exists()

        # Check if the model file can be loaded
        loaded_model = joblib.load(expected_model_filename)
        assert isinstance(loaded_model, type(LogisticRegression()))

def test_train_logreg_acc_models_and_tfidf(tmp_path):
    train_logistic_regression_with_accuracy()  # Execute the function

    # Define the expected directory path
    folder = tmp_path / "logistic_regression_accuracy"
    expected_tfidf_filename = folder / "tfidf_vectorizer.pkl"

    # Check if the directory and files exist
    assert folder.exists()
    assert expected_tfidf_filename.exists()

    # Check if models for each feature exist
    feature_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    for feature in feature_columns:
        expected_model_filename = folder / f"{feature}_model.pkl"
        assert expected_model_filename.exists()

        # Check if the model file can be loaded
        loaded_model = joblib.load(expected_model_filename)
        assert isinstance(loaded_model, type(LogisticRegression()))

def test_train_logreg_f1_models_and_tfidf(tmp_path):
    train_logistic_regression_with_f1()  # Execute the function

    # Define the expected directory path
    folder = tmp_path / "logistic_regression_f1"
    expected_tfidf_filename = folder / "tfidf_vectorizer.pkl"

    # Check if the directory and files exist
    assert folder.exists()
    assert expected_tfidf_filename.exists()

    # Check if models for each feature exist
    feature_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    for feature in feature_columns:
        expected_model_filename = folder / f"{feature}_model.pkl"
        assert expected_model_filename.exists()

        # Check if the model file can be loaded
        loaded_model = joblib.load(expected_model_filename)
        assert isinstance(loaded_model, type(LogisticRegression()))
