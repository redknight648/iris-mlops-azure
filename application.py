from flask import Flask, render_template, request, jsonify
import joblib
import os

app = Flask(__name__)

# Load the model
def load_model():
    path = os.path.join(os.path.dirname(__file__), "model.pkl")
    model = joblib.load(path)
    return model

model = load_model()
TARGET_NAMES = ['setosa', 'versicolor', 'virginica']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_class', methods=['POST'])
def predict_class():
    try:
        data = request.json['data']
        y_pred = model.predict([data])[0]
        pred_class = TARGET_NAMES[y_pred]
        response = {"class": pred_class}
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
