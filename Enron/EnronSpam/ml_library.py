import os
import numpy as np

import nltk
from nltk.corpus import stopwords
from collections import Counter


def extract_features_from(path, dictionary):
    emails = [os.path.join(path, f) for f in os.listdir(path)]

    features_matrix = np.zeros((len(emails), len(dictionary)))
    labels = np.zeros(len(emails))

    index = 0
    for mail in emails:
        with open(mail, encoding="latin1") as m:
            all_words = []
            for line in m:
                words = line.split()
                all_words += words

            for word in all_words:
                wordID = 0
                for i, d in enumerate(dictionary):
                    if d[0] == word:
                        wordID = i
                        features_matrix[index, wordID] = all_words.count(word)

        labels[index] = int(mail.split(".")[-2] == 'spam')
        index = index + 1

    return features_matrix, labels


def make_dictionary_from(path, max_size):
    emails = [os.path.join(path, f) for f in os.listdir(path)]
    all_words = []
    for email in emails:
        with open(email, encoding='latin1') as m:
            content = m.read()
            all_words += nltk.word_tokenize(content)

    dictionary = [word.lower() for word in all_words if word.isalnum()]
    dictionary = [word for word in dictionary if word not in stopwords.words('english')]
    dictionary = Counter(dictionary).most_common(max_size)

    return dictionary
