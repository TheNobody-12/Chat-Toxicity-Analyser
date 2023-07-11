import numpy as np
from flask import Flask,jsonify, request, render_template
import asyncio
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization
import json
import nltk
from Support import TextPreprocessing
from flask_cors import CORS
from waitress import serve
import pandas as pd



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
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_api_json', methods=['POST'])
async def predict_api_json():
    """
    Endpoint for rendering results in JSON format

    [5:04 PM] Moeen Mahmud

// request body

{

  "text": "Text will go here"

}

// respose

{

  "status": 200,

  "predictions": ["preds"],

  "text": "asdlfasdfkl"

}

    """
    data = request.get_json()  # Get the JSON data from the request

    predictions = []



    text = data['text']  # Get the 'text' field from each item in the JSON data



@app.route('/predict_api', methods=['POST'])
def predict_api():
    """
    Endpoint for rendering results in JSON format
    """
    text = request.form['text']  # Get the 'text' field from the form data

    # Perform prediction on the text
    prediction = Get_prediction(text)

    if len(prediction) == 0:
        prediction = 'Not Toxic'
    else:
        prediction = ', '.join(prediction)

    response = {'prediction': prediction}  # Create a response dictionary\
    try:
        return jsonify(response)  # Return the response as JSON
    except:
        return jsonify({'trace': 'An error occurred during prediction'})
    




    
if __name__ == '__main__':
    # serve(app, host="0.0.0.0", port=80)

    app.run(debug=True,port=5000)