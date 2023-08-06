from sklearn.linear_model import LinearRegression
from joblib import dump
import numpy as np

# Train a model
model = LinearRegression()
X = np.array([[1], [2], [3]])
y = np.array([2, 4, 6])
model.fit(X, y)

# Save the trained model to a file
dump(model, 'model.joblib')
