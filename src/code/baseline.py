from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from src.code.clean_text import clean_text
from src.code.data import get_train_data

def get_subset_data(data, num_samples):
    if num_samples == -1:
        return data  # Return all data if num_samples is -1

    # Calculate the subset size based on the provided number of samples or the data size (whichever is smaller)
    subset_size = min(num_samples, len(data))

    # Extract the subset of the data
    subset = data[:subset_size]  # Use data[:subset_size] for pandas DataFrame or data[:subset_size, :] for numpy array

    return subset

def train_baseline(folder, num_samples):

    # Getting training samples
    train_subset = get_subset_data(get_train_data(), num_samples)

    # Cleaning 
    X_train = train_subset['comment_text'].apply(clean_text)

    # Preprocess text data, initialize and fit TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

    # Making a directory
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