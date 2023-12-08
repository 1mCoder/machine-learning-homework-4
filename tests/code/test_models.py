import os
import joblib
from sklearn.linear_model import LogisticRegression
from src.code.baseline import train_baseline 
from src.code.logreg_acc import train_logistic_regression_with_accuracy 
from src.code.logreg_f1 import train_logistic_regression_with_f1

def test_train_baseline_models_and_tfidf(tmp_path):

    # Folder
    folder = f"{tmp_path}/baseline"

    # Execute the function
    train_baseline(folder, 1000)

    # Define the expected directory path
    expected_tfidf_filename = f"{folder}/tfidf_vectorizer.pkl"

    # Check if the directory and files exist
    assert os.path.exists(folder)
    assert os.path.exists(expected_tfidf_filename)

    # Check if models for each feature exist
    feature_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    for feature in feature_columns:
        expected_model_filename = f"{folder}/{feature}_model.pkl"
        assert os.path.exists(expected_model_filename)

        # Check if the model file can be loaded
        loaded_model = joblib.load(expected_model_filename)
        assert isinstance(loaded_model, type(LogisticRegression()))

def test_train_logreg_acc_models_and_tfidf(tmp_path):

    # Folder
    folder = f"{tmp_path}/logistic_regression_accuracy"

    # Execute the function
    train_logistic_regression_with_accuracy(folder, 1000)

    # Define the expected directory path
    expected_tfidf_filename = f"{folder}/tfidf_vectorizer.pkl"

    # Check if the directory and files exist
    assert os.path.exists(folder)
    assert os.path.exists(expected_tfidf_filename)

    # Check if models for each feature exist
    feature_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    for feature in feature_columns:
        expected_model_filename = f"{folder}/{feature}_model.pkl"
        assert os.path.exists(expected_model_filename)

        # Check if the model file can be loaded
        loaded_model = joblib.load(expected_model_filename)
        assert isinstance(loaded_model, type(LogisticRegression()))

def test_train_logreg_f1_models_and_tfidf(tmp_path):

    # Folder
    folder = f"{tmp_path}/logistic_regression_f1"

    # Execute the function
    train_logistic_regression_with_f1(folder, 1000)  

    # Define the expected directory path
    expected_tfidf_filename = f"{folder}/tfidf_vectorizer.pkl"

    # Check if the directory and files exist
    assert os.path.exists(folder)
    assert os.path.exists(expected_tfidf_filename)

    # Check if models for each feature exist
    feature_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    for feature in feature_columns:
        expected_model_filename = f"{folder}/{feature}_model.pkl"
        assert os.path.exists(expected_model_filename)

        # Check if the model file can be loaded
        loaded_model = joblib.load(expected_model_filename)
        assert isinstance(loaded_model, type(LogisticRegression()))
