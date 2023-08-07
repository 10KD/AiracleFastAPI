# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from joblib import dump

# Load Data
data = pd.read_csv('/Users/the_one/Desktop/Repos/flights.csv')

# Define Delay as a Binary Target Variable
data['IsDelayed'] = (data['DepDelay'] > 15).astype(int)

# Splitting Data into Features and Target Variable
X = data[['DayofMonth', 'OriginAirportID', 'DestAirportID']]
y = data['IsDelayed']

# Splitting Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Selection and Training
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
dump(model, 'trained_logistic_model.joblib')
