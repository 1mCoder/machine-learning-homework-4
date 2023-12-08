from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
import joblib
from src.code.clean_text import clean_text
from src.code.data import get_train_data

def get_best_model(train_data, train_labels):
    # Define hyperparameters and their distributions to sample from
    param_dist_lr = {
        'C': [0.1, 0.5, 1, 10, 100, 300, 1000, 100000],
        'penalty': ['l1', 'l2'],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'max_iter': [100, 300, 500, 700, 1000]
    }

    # Initialize the Logistic Regression model
    model_lr = LogisticRegression()

    # Perform randomized search to find the best combination of hyperparameters
    random_search_lr = RandomizedSearchCV(
        model_lr,
        param_dist_lr,
        scoring=make_scorer(f1_score, average='weighted'),
        cv=5,
        random_state=42
    )

    # Fit the randomized search to your data
    random_search_lr.fit(train_data, train_labels)

    # Return the best model
    return random_search_lr.best_estimator_


def train_logistic_regression_with_f1():
    # Getting first 1000 samples
    train_subset = get_train_data().sample(n=1000)

    # Cleaning 
    X_train = train_subset['comment_text'].apply(clean_text)

    # Preprocess text data, initialize and fit TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

    # Making a directory
    folder = "../../models/logistic_regression_f1"
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
        
        y_train = train_subset[feature]  # Select the target feature    
        
        # Initialize and train Logistic Regression model
        logreg = get_best_model(X_train_tfidf, y_train)

        # Store the trained model in the dictionary
        models[feature] = logreg

        # Save the trained model
        model_filename = f"{folder}/{feature}_model.pkl"
        joblib.dump(logreg, model_filename)