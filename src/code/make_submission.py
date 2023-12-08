import pandas as pd
import joblib
from pathlib import Path
from src.code.data import get_test_data
from src.code.clean_text import clean_text

# model can be 'baseline' | 'logreg_acc' | 'logreg_f1'
def make_submission(model, output_folder):

    models_folder = "../../models/"
    if model == "baseline":
        model_directory = f"{models_folder}/baseline"
    elif model == "logreg_acc":
        model_directory = f"{models_folder}/logistic_regression_accuracy"
    elif model == "logreg_f1":
        model_directory = f"{models_folder}/logistic_regression_f1"
    else:
        raise NotImplementedError(f"{model} not implemented")
    
    # Features to consider for classification
    feature_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'] 

    # Load the test data
    test_data = get_test_data()
    X_test = test_data['comment_text'].apply(clean_text)

    # Load the saved TfidfVectorizer
    loaded_tfidf_vectorizer = joblib.load(f"{model_directory}/tfidf_vectorizer.pkl")

    # Create a dictionary to store predictions
    submission = pd.DataFrame()
    submission['id'] = test_data['id']

    # Iterate through each feature and make predictions
    for feature in feature_columns:
        print(f"===== Predicting for '{feature}' =====")
        
        # Load the stored model for the feature
        model_filename = f"{model_directory}/{feature}_model.pkl"
        loaded_model = joblib.load(model_filename)
        
        # Transform test data using the pre-fitted TfidfVectorizer
        X_test_tfidf = loaded_tfidf_vectorizer.transform(X_test)
        
        # Make predictions using the loaded model
        predictions = loaded_model.predict(X_test_tfidf)
        
        # Add predictions to the submission DataFrame
        submission[feature] = predictions

    # saving model, data, and preprocessing
    log_dir = Path(output_folder)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions to a submission CSV file
    submission.to_csv(log_dir / "submission.csv", index=False)