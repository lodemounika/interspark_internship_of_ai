# ============================================================
# FLASK API FOR SPAM DETECTION
# ============================================================

from flask import Flask, request, jsonify

import pickle
import string

# ============================================================
# LOAD SAVED MODEL
# ============================================================

model = pickle.load(
    open("spam_model.pkl", "rb")
)

vectorizer = pickle.load(
    open("vectorizer.pkl", "rb")
)

# ============================================================
# CREATE FLASK APP
# ============================================================

app = Flask(__name__)

# ============================================================
# PREPROCESS FUNCTION
# ============================================================

def preprocess(text):

    text = text.lower()

    text = ''.join(
        [char for char in text if char not in string.punctuation]
    )

    return text

# ============================================================
# HOME ROUTE
# ============================================================

@app.route('/')

def home():

    return "Spam Detection API Running Successfully!"

# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.route('/predict', methods=['POST'])

def predict():

    # Get JSON data
    data = request.get_json()

    # Extract message
    message = data['message']

    # Preprocess
    processed = preprocess(message)

    # Vectorize
    vector = vectorizer.transform([processed])

    # Predict
    prediction = model.predict(vector)[0]

    # Probability
    probability = model.predict_proba(vector)[0][1]

    # Final result
    result = "Spam" if prediction == 1 else "Ham"

    # Return JSON response
    return jsonify({

        "message": message,

        "prediction": result,

        "spam_probability": round(float(probability), 4)
    })

# ============================================================
# RUN FLASK APP
# ============================================================

if __name__ == '__main__':

    app.run(

        host='0.0.0.0',

        port=5000,

        debug=True
    )