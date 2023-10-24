from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
import re
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove emojis
    text = re.sub(r'\\x[a-f0-9]{2}', '', text)

    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    return text


app = Flask(__name__, template_folder='D:/youtube_comments/templates',static_url_path='/static')

# Load the saved model
model = tf.keras.models.load_model("D:/youtube_comments/streamlit.h5")

# Define the tokenizer globally
tokenizer = Tokenizer(num_words=5000) 
max_sequence_length = 100 
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    new_comment = data["comment"]

    # Preprocess the new comment
    cleaned_new_comment = clean_text(new_comment)  # Assuming you have the 'clean_text' function

    # Tokenize and pad the new comment
    new_comment_sequence = tokenizer.texts_to_sequences([cleaned_new_comment])
    new_comment_padded = pad_sequences(new_comment_sequence, maxlen=max_sequence_length)
    prediction = (model.predict(new_comment_padded) > 0.5).astype(int)

    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run()
