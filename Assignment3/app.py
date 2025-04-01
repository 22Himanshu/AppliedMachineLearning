from flask import Flask, request, jsonify
import joblib
from score import score
app = Flask(__name__)
model = joblib.load("/home/himanshu/Downloads/best_model.pkl")  # Load the trained model

@app.route("/score", methods=["POST"])
def score_endpoint():
    # data = request.get_json()
    text = request.json["text"]
    prediction, propensity = score(text, model, threshold=0.5)
    return jsonify({"prediction": int(prediction), "propensity": propensity})

# Generate coverage report
if __name__ == "__main__":
    app.run(debug=True)
