# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from joblib import dump

# Load Data
data = pd.read_csv('/Users/user/Desktop/Repos/testproj/flights.csv')

# Drop rows with NaN in 'DepDelay'
data.dropna(subset=['DepDelay'], inplace=True)

# Define Delay as a Categorical Target Variable (0-15, 16-30, 31-45, 46+ minutes)
bins = [-1, 15, 30, 45, float('inf')]
labels = [0, 1, 2, 3]
data['DelayCategory'] = pd.cut(data['DepDelay'], bins=bins, labels=labels)

# Check rows where 'DelayCategory' is NaN
nan_rows = data[data['DelayCategory'].isnull()]
print("Rows with NaN in 'DelayCategory':")
print(nan_rows)

# Remove rows with NaN in 'DelayCategory'
data.dropna(subset=['DelayCategory'], inplace=True)

# Convert 'DelayCategory' to integer
data['DelayCategory'] = data['DelayCategory'].astype(int)

# Splitting Data into Features and Target Variable
X = data[['DayofMonth', 'OriginAirportID', 'DestAirportID']]
feature_names = X.columns.tolist()  # Storing the names of the features
y = data['DelayCategory']

# Splitting Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Selection and Training (multinomial logistic regression)
model = LogisticRegression(multi_class='multinomial', random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Probability Estimates
y_prob = model.predict_proba(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Display Probability Estimates for the First 5 Predictions
print("Probability Estimates for the First 5 Predictions:")
for i in range(5):
    print("Prediction:", y_pred[i])
    print("Probabilities:", y_prob[i])

# Save Model, Scaler, and Feature Names
dump(model, 'trained_logistic_model.joblib')
dump((scaler, feature_names), 'scaler_and_features.joblib')  # Storing scaler and feature names together
