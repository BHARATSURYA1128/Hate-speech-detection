from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load model and vectorizer
rf_model = pickle.load(open("RF.sav", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Home page
@app.route('/')
def home():
    return render_template("index.html")

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data['text']
        model_type = data.get('model', 'rf')

        vec = vectorizer.transform([text])

        # Use Random Forest
        prediction = rf_model.predict(vec)[0]

        labels = {0: "Hate Speech", 1: "Offensive", 2: "Neutral"}
        result = labels.get(prediction, str(prediction))

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)})