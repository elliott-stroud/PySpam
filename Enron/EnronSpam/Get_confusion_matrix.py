# This is code to get the confusion matrix
import os
import numpy as np
import nltk
import datetime
import random

import config
import utils
import ml_library as ml

from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from nproc import npc

nltk.download('punkt')
nltk.download('stopwords')

utils.log('NLTK Updated')

# set seed to zero so that results are deterministic & repeatable
random.seed(0)

num_of_train_emails = len(os.listdir(config.DIR_DATASET_TRAIN))
num_of_test_emails = len(os.listdir(config.DIR_DATASET_TEST))
utils.log("Size of {dir}: {size}".format(dir=config.DIR_DATASET_TRAIN, size=num_of_train_emails))
utils.log("Size of {dir}: {size}".format(dir=config.DIR_DATASET_TEST, size=num_of_test_emails))


utils.log('Creating a dictionary from the training data')
dictionary = ml.make_dictionary_from(config.DIR_DATASET_TRAIN, 3000)
utils.log('Dictionary creation complete')

utils.log('Extracting features for training')
train_matrix, train_labels = ml.extract_features_from(config.DIR_DATASET_TRAIN, dictionary)
utils.log('Feature extraction complete')

NPNBTrainer = npc()

model1 = MultinomialNB()
model2 = LinearSVC()

utils.log('Begin Training: Model 1 (NB)')
model1.fit(train_matrix, train_labels)
utils.log('Training Complete: Model 1 (NB)')

utils.log('Begin Training: Model 2 (SVM)')
model2.fit(train_matrix, train_labels)
utils.log('Training Complete: Model 2 (SVM)')

utils.log('Begin Training: Model 3 default (NP NB)')
model_npnb_default = NPNBTrainer.npc(train_matrix, train_labels, 'nb')
utils.log('Training Complete: Model 3 default (NP NB)')

utils.log('Begin Training: Model 3 one percent (NP NB)')
model_npnb_1p = NPNBTrainer.npc(train_matrix, train_labels, 'nb', alpha=0.01)
utils.log('Training Complete: Model 3 one percent (NP NB)')

utils.log('Begin Training: Model 3 a tenth of a percent (NP NB)')
model_npnb_tenth = NPNBTrainer.npc(train_matrix, train_labels, 'nb', alpha=0.001)
utils.log('Training Complete: Model 3 a tenth of a percent (NP NB)')

utils.log('Begin Training: Model 4 (NP SVM)')
model4 = NPNBTrainer.npc(train_matrix, train_labels, 'svm')
utils.log('Training Complete: Model 4 (NP SVM)')

utils.log('Extracting features for testing')
test_matrix, test_labels = ml.extract_features_from(config.DIR_DATASET_TEST, dictionary)
utils.log('Feature extraction complete')

utils.log('Predicting NB and SVC')

utils.log('result1 - Predicting Results')
result1 = model1.predict(test_matrix)
utils.log('result1 - Prediction Complete')

utils.log('result2 - Predicting Results')
result2 = model2.predict(test_matrix)
utils.log('result2 - Prediction Complete')

utils.log('Predicting results Neyman Pearson')

utils.log('npnb_default - Predicting Results')
result_npnb_default = NPNBTrainer.predict(model_npnb_default, test_matrix)[0]
utils.log('npnb_default - Prediction Complete')

utils.log('npnb_1p - Predicting Results')
result_npnb_1p = NPNBTrainer.predict(model_npnb_1p, test_matrix)[0]
utils.log('npnb_1p - Prediction Complete')

utils.log('npnb_tenth - Predicting Results')
result_npnb_tenth = NPNBTrainer.predict(model_npnb_tenth, test_matrix)[0]
utils.log('npnb_tenth - Prediction Complete')

utils.log('result4 - Predicting Results')
result4 = NPNBTrainer.predict(model4, test_matrix)[0]
utils.log('result4 - Prediction Complete')

utils.log('All Predictions Complete')

print("MultinomialNB results:")
print(confusion_matrix(test_labels, result1))
print(classification_report(test_labels, result1, digits=4))
print("")
print("LinearSVC results:")
print(confusion_matrix(test_labels, result2))
print(classification_report(test_labels, result2, digits=4))
print("")
print("MultinomialNB with default NP results:")
print(confusion_matrix(test_labels, result_npnb_default))
print(classification_report(test_labels, result_npnb_default, digits=4))
print("")
print("MultinomialNB 1% alpha NP results:")
print(confusion_matrix(test_labels, result_npnb_1p))
print(classification_report(test_labels, result_npnb_1p, digits=4))
print("")
print("MultinomialNB 0.1% alpha NP results:")
print(confusion_matrix(test_labels, result_npnb_tenth))
print(classification_report(test_labels, result_npnb_tenth, digits=4))
print("")
print("LinearSVC with default NP results:")
print(confusion_matrix(test_labels, result4))
print(classification_report(test_labels, result4, digits=4))

utils.log('Execution completed successfully')
