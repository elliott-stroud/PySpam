import pandas as pd
import utils

import sklearn

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

from nproc import npc

# Load the dataset
dataset = pd.read_csv('spam_sms.csv', header=0)

# Prepare the columns; Rename the relevant columsn and remove the excess
dataset.rename({'v1': 'SPAM', 'v2': 'MESSAGE'}, axis='columns', inplace=True)
dataset = dataset.drop(dataset.columns[[2, 3, 4]], axis=1)

# Change labels to boolean values
dataset['SPAM'].replace({'ham': False, 'spam': True}, inplace=True)

# INFO: Print the number of unique values
print(dataset['SPAM'].value_counts(normalize=True))

# # Obtain a downsampled dataset since analysing the entire dataset can take a lot of time
# # We'll filter the dataset for ham and spam, and then combine the results
# dataset_downsampled = pd.concat([
#   # The spam data, number of samples can be changed by modifying the value for n_samples
#   resample(dataset[dataset['SPAM'] == True], replace=False, n_samples=500),
#   # The ham data, number of samples can be changed by modifying the value for n_samples
#   resample(dataset[dataset['SPAM'] == False], replace=False, n_samples=1000)
#   ])

dataset_downsampled = dataset

# Segregate the message text from the spam flag
txt_msg_spam_flag = dataset_downsampled['SPAM'].copy()
txt_msg_messages = dataset_downsampled['MESSAGE'].copy()

# We now convert the data from a string to a frequency distribution
vectorizer = CountVectorizer(analyzer=utils.denoise_text)
text_message_freq_vector = vectorizer.fit_transform(txt_msg_messages)

# INFO: The number of entries in the vector
print(text_message_freq_vector.shape)

# We now split the data into training and testing datasets
msg_train, msg_test, is_spam_train, is_spam_test = train_test_split(text_message_freq_vector, txt_msg_spam_flag, train_size=0.666666666)

######################

# We now begin creating the model using the Multinomial Naive Bayes method
MxNB = MultinomialNB()

# Train the model
MxNB.fit(msg_train, is_spam_train)

# Predict outcomes based on testing data
is_spam_pred = MxNB.predict(msg_test)

# Show the results
utils.show_results("Multinomial Naive Bayes", is_spam_test, is_spam_pred)

######################

# Neyman Pearson now

NPNBTrainer = npc()

for alpha in [0.05, 0.01, 0.001]:

  # Train the model
  npnb_model = NPNBTrainer.npc(msg_train.toarray(), is_spam_train.to_numpy(), 'nb', alpha=alpha)

  # Predict outcomes based on the testing data
  is_spam_pred_np = NPNBTrainer.predict(npnb_model, msg_test.toarray())[0]

  # Show the results
  utils.show_results("Multinomial Naive Bayes with Neyman Pearson: alpha={}%".format(alpha*100), is_spam_test, is_spam_pred_np)
