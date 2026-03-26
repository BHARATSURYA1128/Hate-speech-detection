from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load model + vectorizer
model = pickle.load(open("RF.sav", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return "✅ ContextGuard API is running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['text']
        vector = vectorizer.transform([data])
        prediction = model.predict(vector)[0]

        # Optional: map output to label
        labels = {0: "Hate Speech", 1: "Offensive", 2: "Neutral"}
        result = labels.get(prediction, str(prediction))

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run()