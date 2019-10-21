# # Load packages

# Ignore warnings
import warnings

def warn(*args, **kwargs):
    pass

warnings.warn = warn

import os
import time
import numpy as np
import pandas as pd
import re

import keras
from keras import *
from keras import layers
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from keras.models import Model
from keras.preprocessing import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pickle

import seaborn as sns
import matplotlib.pyplot as plt

#for reproducible results
from numpy.random import seed
seed(302)
from tensorflow import set_random_seed
set_random_seed(302)


# # Load and explore data

sms_df=pd.read_csv('SMSSpamCollection',sep='\t',header=None)

sms_df.head()

sms_df.tail()

# rename variables
sms_df=sms_df.rename(columns={0: 'Labels', 1: 'Text'})

# create binary where 1 = spam and 0 = ham
sms_df['Labels_Binary'] = 0
sms_df.loc[sms_df['Labels']=='spam', 'Labels_Binary'] = 1

texts = sms_df['Text']
labels = sms_df['Labels_Binary']

# Partition data into train, valid, test

#Create train/test sample
other_texts, test_texts, other_labels, test_labels  = train_test_split(texts, labels, test_size=0.1, random_state=302)
#Create validation sample
train_texts, valid_texts, train_labels, valid_labels  = train_test_split(other_texts, other_labels, test_size=0.2, random_state=302)

# Evaluate ham/spam breakdown

cases_count = labels.value_counts(dropna=False)
        
# Plot  results 
plt.figure(figsize=(6,6))
sns.barplot(x=cases_count.index, y=cases_count.values)
plt.ylabel('Texts', fontsize=12)
plt.xticks(range(len(cases_count.index)), ['Ham', 'Spam'])

plt.show()

# # Train model

# Define vocabulary size (you can tune this parameter and evaluate model performance)
VOCABULARY_SIZE = 5000

# Create input feature arrays
tokenizer = Tokenizer(num_words=VOCABULARY_SIZE)
tokenizer.fit_on_texts(train_texts)

# Convert words into word ids
meanLength = np.mean([len(item.split(" ")) for item in train_texts])
MAX_SENTENCE_LENGTH = int(meanLength + 5) # we let a text go 10 words longer than the mean text length.

# Convert train, validation, and test text into lists with word ids
trainFeatures = tokenizer.texts_to_sequences(train_texts)
trainFeatures = pad_sequences(trainFeatures, MAX_SENTENCE_LENGTH, padding='post')
trainLabels = train_labels.values

validFeatures = tokenizer.texts_to_sequences(valid_texts)
validFeatures = pad_sequences(validFeatures, MAX_SENTENCE_LENGTH, padding='post')
validLabels = valid_labels.values

testFeatures = tokenizer.texts_to_sequences(test_texts)
testFeatures = pad_sequences(testFeatures, MAX_SENTENCE_LENGTH, padding='post')
testLabels = test_labels.values

# we will use this hardcoded sentence length in the prep for the flask app
MAX_SENTENCE_LENGTH

# Define filter and kernel size for CNN (can adjust in tuning model)
FILTERS_SIZE = 16
KERNEL_SIZE = 5

# Define embeddings dimensions (columns in matrix fed into CNN and nodes in hidden layer of built-in keras function)
EMBEDDINGS_DIM = 10

# Hyperparameters for model tuning
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 7

# Word CNN
model = Sequential()

# We use built-in keras funtion to generate embeddings. Another option is pre-trained embeddings with Word2vec or GloVe.
model.add(Embedding(input_dim=VOCABULARY_SIZE + 1, output_dim=EMBEDDINGS_DIM, input_length=len(trainFeatures[0])))
model.add(Conv1D(FILTERS_SIZE, KERNEL_SIZE, activation='relu'))
model.add(Dropout(0.5))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.5))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
            
optimizer = optimizers.Adam(lr=LEARNING_RATE)
model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])

print(model.summary())

history = model.fit(trainFeatures, trainLabels, validation_data = (validFeatures, validLabels), batch_size=BATCH_SIZE, epochs=EPOCHS)

# summarize accuracy by epoch
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['training', 'validation'], loc='upper left')
plt.figure(figsize=(20,10))
plt.show()


# Get examples from embeddings

from keras import backend as K

get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[0].output,])
activations = get_activations([trainFeatures,0])[0]

pd.set_option('display.max_colwidth', -1)
pd.DataFrame(train_texts).iloc[[113]]

trainFeatures[113]

df1 = pd.DataFrame(activations[113])

df1

# # Evaluate in test set

# Predict binary and probabilities
predictions_df = pd.DataFrame(model.predict(testFeatures))
predictions_binary_df = round(predictions_df)
accuracy_score(testLabels, predictions_binary_df)


predictions_binary_df[0].value_counts(dropna=False)

# # Save model architecture and pre-trained weights for Flask

# save to the directory with flask app
current_dir = os.getcwd()
output_dir = re.sub('Model and data', 'Flask application', current_dir)
os.chdir(output_dir)

# save tokenizer for preprocessing
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# serialize model to JSON for Flask App
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")

