# Airacle: Flight Delay Prediction System

## Introduction

Airacle is a flight delay prediction application that leverages machine learning to provide you with accurate and timely information on flight delays. Here I'll explain the code and structure of the project, detailing each file's purpose and functionality.

## `airacle.py`: Python FastAPI Application

This file defines the main FastAPI application. It includes:

- **FastAPI Instance**: The main application object that defines the API endpoints.
- **Startup and Shutdown Events**: Functions to connect and disconnect from the database at startup and shutdown.
- **Prediction Endpoint**: A POST request to `/predict/` that takes flight data, predicts delays using the `predict_delay` function, saves the result in the PostgreSQL database, and returns the prediction.

## `database.py`: Database Management

This file manages the database connection and schema using SQLAlchemy:

- **Database Configuration**: Establishes the database connection using environment variables.
- **Prediction Class**: Defines the `Prediction` table schema to store input data and prediction results.
- **Save Prediction Function**: An async function to insert predictions into the database.

## `model.py`: AI Model (trained using SKLearn)

This file deals with predicting flight delays:

- **Model and Scaler Loading**: Loads a pre-trained logistic regression model and a scaler from disk.
- **Predict Delay Function**: Decodes input JSON, scales the features, and returns the prediction result.

## `train_model.py`: AI Model Training

This file covers the training of the logistic regression model:

- **Data Loading and Preprocessing**: Loads flight data, handles missing values, and categorizes delays.
- **Feature Selection**: Selects relevant features and stores feature names.
- **Data Splitting and Scaling**: Splits data into training and test sets, scales features.
- **Model Training**: Trains a multinomial logistic regression model.
- **Prediction and Evaluation**: Predicts on test data, calculates accuracy, and prints probability estimates for the first five predictions.
- **Saving Model and Scaler**: Saves the trained model, scaler, and feature names to disk.

## Tools and Libraries

- **Languages**: Python 3
- **Libraries**: FastAPI, SQLAlchemy, pandas, scikit-learn, joblib
- **Datastores**: Postgres (for storing predictions)
- **Version control**: Github

## Conclusion

Airacle provides a system for predicting flight delays, built using modern Python libraries and tools. Each file plays a specific role in managing the application, database connection, model prediction, or model training.