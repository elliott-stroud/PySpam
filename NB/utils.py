
import string

import nltk
from nltk.corpus import stopwords

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

SCREEN_SIZE = 40

# Download the stopworks package.
nltk.download('stopwords')

# Define a text processor that will:
# 1. Convert the text to lowercase
# 2. Remove all punctuations
# 3. Remove all stopwords (a, an, the, etc.)


def denoise_text(text):
    clean_text = [symbol for symbol in text.lower() if symbol not in string.punctuation]
    clean_text = ''.join(clean_text)

    relevant_text = [token for token in clean_text.split() if token.lower() not in stopwords.words('english')]

    return relevant_text


def show_results(msg, test_data, predictions):
    print("[RESULTS]", msg)
    print("Accuracy Score:", accuracy_score(test_data, predictions))
    print("-"*SCREEN_SIZE)

    print("Confusion Matrix")
    print(confusion_matrix(test_data, predictions))
    print("-"*SCREEN_SIZE)

    print("Classification report")
    print(classification_report(test_data, predictions, digits=4))
    print("="*SCREEN_SIZE, "\n")
