import requests
import subprocess
import time
import joblib
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from score import score

def test_score():
    """Unit test for score function."""
    print("Running test_score...")  # This should print when the test is run
    # model = joblib.load("/home/himanshu/Downloads/best_model.pkl")
    model = joblib.load("best_model.pkl")

    # Smoke test
    assert score("test", model, 0.5) is not None

    # Format test
    prediction, propensity = score("test", model, 0.5)
    assert isinstance(prediction, bool)
    assert isinstance(propensity, float)

    # Propensity constraints
    assert 0.0 <= propensity <= 1.0

    # Threshold behavior
    assert score("test", model, 0.0)[0] == True
    assert score("test", model, 1.0)[0] == False

    # Edge cases (example texts, assuming spam detection)
    assert score("Urgent! You have won a 1-week FREE vacation. Call 09012345678 NOW!", model, 0.5)[0] == True  # Obvious spam
    assert score("Hello, how are you?", model, 0.5)[0] == False  # Not spam
    assert score("This is to confirm our meeting scheduled for tomorrow, March 13th, at 10:00 AM in the conference room. We'll be discussing the new marketing strategy.", model, 0.5)[0] == False  # Likely non-spam
    assert score("Your email has been randomly selected to receive a $500 gift card. Click here to claim your reward", model, 0.5)[0] == True #Likely spam
def test_flask():
    """Integration test for Flask endpoint."""
    print("Running test_flask...")  # This should print when the test is run
    flask_process = subprocess.Popen(["python", "app.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(2)  # Give some time for server to start

    response = requests.post("http://127.0.0.1:5000/score", json={"text": "test"})
    # response = requests.post("http://127.0.0.1:5001/score", json={"text": "test"})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data and "propensity" in data

    flask_process.terminate()  # Stop Flask app
    flask_process.wait()
if __name__ == "__main__":
    pytest.main(["test.py"])
    subprocess.run(["pytest", "--cov=.", "--cov-report=term-missing"], text=True)
