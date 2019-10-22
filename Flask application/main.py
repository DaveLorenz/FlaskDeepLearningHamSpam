# load packages
import os

import flask
app = flask.Flask(__name__)
from flask import Flask, render_template, request

#load model preprocessing
import numpy as np
import pandas as pd
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.models
from keras.models import model_from_json

# Load tokenizer for preprocessing
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load pre-trained model into memory
json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load weights into new model
loaded_model.load_weights("model.h5")

def prepData(text):
    # Convert to array
    textDataArray = [text]
    
    # Convert into list with word ids
    Features = tokenizer.texts_to_sequences(textDataArray)
    Features = pad_sequences(Features, 20, padding='post')
    
    return Features
    
loaded_model.compile(optimizer="Adam",loss='binary_crossentropy',metrics=['accuracy'])

# define a predict function as an endpoint 

@app.route('/', methods=['GET','POST'])
def predict():
    
    #whenever the predict method is called, we're going
    #to input the user entered text into the model
    #and return a prediction
    
    if request.method=='POST':
        textData = request.form.get('text_entered')
        Features = prepData(textData)
        prediction = int((np.asscalar(loaded_model.predict(Features)))*100)
        return render_template('prediction.html', prediction=prediction)
    
    else:
        return render_template("search_page.html")   

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080)
    
