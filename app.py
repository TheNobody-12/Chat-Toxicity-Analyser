import pandas as pd
import numpy as np
from flask import Flask,jsonify, request, render_template
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization
import json
import nltk
from Support import TextPreprocessing
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


class_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# load model
TextClassifier = load_model('trial1.h5')
# Load the vectorizer configuration
with open('vectorizer_config.json', 'r') as f:
    vectorizer_config = json.load(f)

# Create a new vectorizer using the loaded configuration
vectorizer = TextVectorization.from_config(vectorizer_config)

# Load the vocabulary
vocabulary = []
with open('vocabulary.txt', 'r',encoding="utf8") as f:
    for line in f:
        word = line.strip()
        vocabulary.append(word)

# Adapt the vectorizer to the loaded vocabulary
vectorizer.set_vocabulary(vocabulary)
# Create a preprocessor object
preprocessor = TextPreprocessing()

def Get_prediction(text):
    user_input = preprocessor.preprocess_text(text)
    user_input=' '.join(user_input)
    # print(user_input)
    vectorized_text = vectorizer(user_input)
    # print(vectorized_text)
    prediction = TextClassifier.predict(np.expand_dims(vectorized_text,0))
    # Convert the prediction probabilities to binary form
    binary_predictions = np.where(prediction > 0.5, 1, 0)
    predicted_classes = [class_labels[i] for i, pred in enumerate(binary_predictions[0]) if pred == 1]
    return predicted_classes

# create flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    For rendering results on HTML GUI

    request.json is a dictionary object with 
    key as 'text' and value as the text entered 
    by the user in the text box


    """
    text = request.form['text'] 
    prediction = Get_prediction(text)
    if len(prediction) == 0:
        prediction = 'Not Toxic'
    else:
        prediction =' ,'.join(prediction)
    return render_template('index.html', prediction_text='The comment is {}.'.format(prediction))


if __name__ == '__main__':
    app.run(debug=True)