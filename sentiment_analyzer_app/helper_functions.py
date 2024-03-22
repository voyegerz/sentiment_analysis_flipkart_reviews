import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def clean_text(text):
    # Remove special characters and punctuation
    text = re.sub(r"[^a-zA-Z]", " ", text)

    # Remove special characters and digits
    text = re.sub(r"\W+", " ", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Convert text to lowercase
    text = text.lower()

    # Remove end read more characters
    text = re.sub(r"read more", " ", text)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]
    return " ".join(cleaned_words)


def lemmatize_text(text):
    # Initialize WordNet lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Tokenize text
    tokens = nltk.word_tokenize(text)

    lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(lemmatized_words)
