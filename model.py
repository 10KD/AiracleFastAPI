from joblib import load
import numpy as np
import json

# Load pre-trained model from a file
model = load('pretrained_models/trained_logistic_model.joblib')

def predict_delay(input_json: str):
    input_data = json.loads(input_json)  # Decode the JSON string to get the array
    input_array = np.array([input_data])
    prediction_result = model.predict(input_array)[0]
    return prediction_result
