from joblib import load
import json
import pandas as pd

# Load pre-trained model and scaler from files
model = load('pretrained_models/trained_logistic_model.joblib')
scaler, feature_names = load('pretrained_models/scaler.joblib')

def predict_delay(input_json: str):
    input_data = json.loads(input_json)  # Decode the JSON string to get the array
    input_df = pd.DataFrame([input_data], columns=feature_names)  # Create DataFrame with correct columns
    scaled_input = scaler.transform(input_df)  # Apply the same scaling transformation
    prediction_result = model.predict_proba(scaled_input)[0]
    return prediction_result
