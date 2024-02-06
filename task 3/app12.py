# app/app.py
from flask import Flask, render_template, request
import pandas as pd
import mlflow.sklearn
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load RandomForest model using MLFlow
rf_model_path = ('file:///c:/Users/VARUN/OneDrive/Desktop/mlflow/mlruns/0/396f70585e764abdab165141175e32ad/artifacts/RandomForest Model')
rf_loaded_model = mlflow.sklearn.load_model(rf_model_path)

# Load LSTM model
lstm_model_path = r"C:\Users\VARUN\AI COURSE DIGICROME\1 PYTHON\models\LSTM_sales-2024-02-05-13-57-34.pkl"
lstm_loaded_model = load_model(lstm_model_path)

# Define a route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Define a route for handling RandomForest predictions
@app.route('/predict_rf', methods=['POST'])
def predict_rf():
    try:
        # Get input parameters from the frontend
        store_id = request.form['store_id']
        uploaded_file = request.files['csv_file']

        # Read the uploaded CSV file
        df = pd.read_csv(uploaded_file)

        # Additional parameters from the frontend
        is_holiday = request.form['is_holiday']
        is_weekend = request.form['is_weekend']
        is_promo = request.form['is_promo']

    

        # Make predictions using the loaded RandomForest model
        rf_predictions = rf_loaded_model.predict(df)

        # Add the RandomForest predictions to the dataframe
        df['PredictedSales_RF'] = rf_predictions

        return render_template('result_rf.html', predictions_rf=df.to_html())

    except Exception as e:
        return render_template('error.html', error=str(e))

# Define a route for handling LSTM predictions
@app.route('/predict_lstm', methods=['POST'])
def predict_lstm():
    try:
        # Get input parameters from the frontend
        store_id = request.form['store_id']
        uploaded_file = request.files['csv_file']

        # Read the uploaded CSV file
        df = pd.read_csv(uploaded_file)

    

        # Make predictions using the loaded LSTM model
        lstm_predictions = lstm_loaded_model.predict(df)  # Adjust this based on your LSTM model's input requirements

        # Add the LSTM predictions to the dataframe
        df['PredictedSales_LSTM'] = lstm_predictions

        return render_template('result_lstm.html', predictions_lstm=df.to_html())

    except Exception as e:
        return render_template('error.html', error=str(e))

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
