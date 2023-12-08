from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from clean_text import clean_text
from data import get_train_data

def train_baseline():

    # Getting first 1000 samples
    train_subset = get_train_data().sample(n=1000)

    # Cleaning 
    X_train = train_subset['comment_text'].apply(clean_text)

    # Preprocess text data, initialize and fit TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

    # Making a directory
    folder = "../../models/baseline"
    log_dir = Path(folder)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save the TfidfVectorizer
    tfidf_filename = f"{folder}/tfidf_vectorizer.pkl"
    joblib.dump(tfidf_vectorizer, tfidf_filename)

    # Models dictionary to store models' parameters
    models = {}

    # Features to consider for classification
    feature_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    for feature in feature_columns:
        print(f"===== Predicting for '{feature}' =====")
        
        # Initialize and train Logistic Regression model
        logreg = LogisticRegression(max_iter=1000)
        logreg.fit(X_train_tfidf, train_subset[feature])

        # Store the trained model in the dictionary
        models[feature] = logreg

        # Save the trained model
        model_filename = f"{folder}/{feature}_model.pkl"
        joblib.dump(logreg, model_filename)