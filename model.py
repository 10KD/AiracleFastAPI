from joblib import load
import numpy as np

# Load pre-trained model from a file
model = load('pretrained_models/model.joblib')

def predict_delay(input_data: float):
    input_array = np.array([[input_data]])
    prediction_result = model.predict(input_array)[0]
    return prediction_result
