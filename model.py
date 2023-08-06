from sklearn.linear_model import LinearRegression
import numpy as np

# Train a model
model = LinearRegression()
X = np.array([[1], [2], [3]])
y = np.array([2, 4, 6])
model.fit(X, y)

def predict_delay(input_data: float):
    input_array = np.array([[input_data]])
    prediction_result = model.predict(input_array)[0]
    return prediction_result
