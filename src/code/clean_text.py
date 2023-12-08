import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet

# Function to perform text cleaning
def clean_text(text):
    nltk.download('punkt', quiet=True)

    # Check if 'stopwords' are downloaded
    if not stopwords.fileids():
        nltk.download('stopwords', quiet=True)

    # Check if 'wordnet' is downloaded
    if not wordnet.fileids():
        nltk.download('wordnet', quiet=True)

    # Remove special characters and punctuation except @, #, and '
    cleaned_text = re.sub(r"[^\w\s#@']", "", text)

    # Normalize text (convert to lowercase)
    cleaned_text = cleaned_text.lower()

    # Tokenize the text
    tokens = word_tokenize(cleaned_text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Initialize lemmatization
    lemmatizer = WordNetLemmatizer()

    # Apply stemming and lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    # Join tokens back into text
    cleaned_text = ' '.join(lemmatized_tokens)

    # Handle whitespaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text