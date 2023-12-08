import pandas as pd
from src.code.make_submission import make_submission 

# Helper function to check columns in the submission file
def check_submission_columns(submission_file):
    expected_columns = ['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    submission = pd.read_csv(submission_file)
    assert list(submission.columns) == expected_columns

# Updated test case for making submission using baseline model with column check
def test_make_submission_baseline(tmp_path):
    output_folder =  f"{tmp_path}/test_submission_baseline"
    make_submission('baseline', output_folder)
    submission_file =  f"{output_folder}/submission.csv"
    assert submission_file.exists()
    check_submission_columns(submission_file)

# Updated test case for making submission using logreg_acc model with column check
def test_make_submission_logreg_acc(tmp_path):
    output_folder =  f"{tmp_path}/test_submission_logreg_acc"
    make_submission('logreg_acc', output_folder)
    submission_file = f"{output_folder}/submission.csv"
    assert submission_file.exists()
    check_submission_columns(submission_file)

# Updated test case for making submission using logreg_f1 model with column check
def test_make_submission_logreg_f1(tmp_path):
    output_folder = f"{tmp_path}/test_submission_logreg_f1"
    make_submission('logreg_f1', output_folder)
    submission_file = f"{output_folder}/submission.csv"
    assert submission_file.exists()
    check_submission_columns(submission_file)
