# app.py
# This is the code for the backend server using the Flask framework.
# In a real application, you would run this file on a server.

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Import the CORS library
import joblib
import string
import warnings

# --- Suppress the InconsistentVersionWarning for cleaner output ---
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


# --- Model Loading ---
try:
    model = joblib.load('spam_model.pkl')
except FileNotFoundError:
    print("Model file 'spam_model.pkl' not found. Backend functionality will be limited.")
    model = None

try:
    vectorizer = joblib.load('vectorizer.pkl')
except FileNotFoundError:
    print("Vectorizer file 'vectorizer.pkl' not found. Backend functionality will be limited.")
    vectorizer = None


# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)  # <<< NEW: Enable CORS for your Flask app. This is the main fix.


# --- API Routes ---

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/analysis')
def analysis():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives a message in JSON format, processes it,
    predicts using the loaded model, and returns the result.
    """
    if not model or not vectorizer:
        return jsonify({'error': 'Model or vectorizer not loaded on the server.'}), 500

    data = request.get_json(force=True)
    message = data.get('message', '')

    if not message:
        return jsonify({'error': 'No message provided'}), 400

    message_vector = vectorizer.transform([message])
    prediction_code = model.predict(message_vector)[0]
    probabilities = model.predict_proba(message_vector)[0]
    prediction_label = 'Spam' if prediction_code == 1 else 'Not Spam (Ham)'
    confidence = round(max(probabilities) * 100, 2)

    return jsonify({
        'message': message,
        'prediction': prediction_label,
        'confidence': confidence
    })


# --- Main Execution ---
if __name__ == '__main__':
    # The host '0.0.0.0' makes the server accessible on your local network.
    app.run(host='0.0.0.0', port=5000, debug=True)
