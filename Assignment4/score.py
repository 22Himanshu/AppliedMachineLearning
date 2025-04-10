from sklearn.base import BaseEstimator

def score(text: str, model: BaseEstimator, threshold: float):
    """
    Scores a trained model on the given text.
    Returns prediction (bool) and propensity (float).
    """
    propensity = model.predict_proba([text])[0][1]  # Assuming binary classification
    prediction = bool(propensity >= threshold)  # Explicitly convert to Python bool
    return prediction, propensity
