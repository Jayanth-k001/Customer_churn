from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
import os
import traceback
from modules import (
    get_default_metrics,
    store_prediction,
    calculate_customer_segments,
    calculate_feature_importance
)

app = Flask(__name__)

model = joblib.load('../Notebooks/final_logistic_regression_model.pkl')
sample_data = pd.read_csv('../data/processed/cleaned_data.csv')


# Define feature lists
categorical_features = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

numeric_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

# Create preprocessing pipeline
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Add MonthlyTenureInteraction to sample_data before fitting
sample_data['MonthlyTenureInteraction'] = sample_data['MonthlyCharges'] * sample_data['tenure']

# Update numeric_features to include the interaction term
numeric_features.append('MonthlyTenureInteraction')

# Fit preprocessor on sample data
preprocessor.fit(sample_data)

@app.route('/')
def home():
    return render_template('index.html', metrics=get_default_metrics())

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form and print for debugging
        input_data = {
            'gender': int(request.form['gender']),
            'SeniorCitizen': int(request.form['SeniorCitizen']),
            'Partner': int(request.form['Partner']),
            'Dependents': int(request.form['Dependents']),
            'tenure': float(request.form['tenure']),
            'PhoneService': int(request.form['PhoneService']),
            'MultipleLines': int(request.form['MultipleLines']),
            'InternetService': int(request.form['InternetService']),
            'OnlineSecurity': int(request.form['OnlineSecurity']),
            'OnlineBackup': int(request.form['OnlineBackup']),
            'DeviceProtection': int(request.form['DeviceProtection']),
            'TechSupport': int(request.form['TechSupport']),
            'StreamingTV': int(request.form['StreamingTV']),
            'StreamingMovies': int(request.form['StreamingMovies']),
            'Contract': int(request.form['Contract']),
            'PaperlessBilling': int(request.form['PaperlessBilling']),
            'PaymentMethod': int(request.form['PaymentMethod']),
            'MonthlyCharges': float(request.form['MonthlyCharges']),
            'TotalCharges': float(request.form['TotalCharges'])
        }

        # Create DataFrame from input
        input_df = pd.DataFrame([input_data])
        
        # Add MonthlyTenureInteraction feature
        input_df['MonthlyTenureInteraction'] = input_df['MonthlyCharges'] * input_df['tenure']

        # Transform the input data
        X_transformed = preprocessor.transform(input_df)

        # Make prediction
        prediction = model.predict(X_transformed)
        probabilities = model.predict_proba(X_transformed)[0]
        
        # Create result message
        result = "Churn" if prediction[0] == 1 else "No Churn"
        probability_text = f"{probabilities[1]:.2%}"
        
        # Store the prediction
        prediction_data = {
            'gender': input_data['gender'],
            'SeniorCitizen': input_data['SeniorCitizen'],
            'Partner': input_data['Partner'],
            'Dependents': input_data['Dependents'],
            'tenure': input_data['tenure'],
            'PhoneService': input_data['PhoneService'],
            'MultipleLines': input_data['MultipleLines'],
            'InternetService': input_data['InternetService'],
            'OnlineSecurity': input_data['OnlineSecurity'],
            'OnlineBackup': input_data['OnlineBackup'],
            'DeviceProtection': input_data['DeviceProtection'],
            'TechSupport': input_data['TechSupport'],
            'StreamingTV': input_data['StreamingTV'],
            'StreamingMovies': input_data['StreamingMovies'],
            'Contract': input_data['Contract'],
            'PaperlessBilling': input_data['PaperlessBilling'],
            'PaymentMethod': input_data['PaymentMethod'],
            'MonthlyCharges': input_data['MonthlyCharges'],
            'TotalCharges': input_data['TotalCharges'],
            'prediction': result,
            'probability': probabilities[1],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        store_prediction(prediction_data)

        return render_template('index.html',
                             metrics=get_default_metrics(),  # Add metrics here
                             model='Model: LogisticRegression',
                             prediction_text=f'Prediction: {result}')

    except Exception as e:
        print(f"Error: {str(e)}")
        return render_template('index.html', 
                             metrics=get_default_metrics(),  # Add metrics here
                             prediction_text='Error making prediction',
                             error_text=str(e))



@app.route('/dashboard-data')
def dashboard_data():
    try:
        # Get metrics from predictions file
        csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'predictions.csv'))
        metrics = get_default_metrics()
        if os.path.exists(csv_path):
            predictions_df = pd.read_csv(csv_path)
            total_predictions = len(predictions_df)
            churn_count = sum(predictions_df['prediction'] == 'Churn')
            retain_count = total_predictions - churn_count
            avg_confidence = predictions_df['probability'].mean() if 'probability' in predictions_df.columns else 0.857
        else:
            total_predictions = 100
            churn_count = 24
            retain_count = 76
            avg_confidence = 0.857
        
        response_data = {
            'churn_rate': round((churn_count / total_predictions * 100), 1) if total_predictions > 0 else 0,
            'avg_confidence': round(avg_confidence * 100, 1),
            'churn_count': int(churn_count),
            'retain_count': int(retain_count),
             'high_risk_count': metrics['high_risk_count'],
            'medium_risk_count': metrics['medium_risk_count'],
            'low_risk_count': metrics['low_risk_count'],
            'high_risk_percentage': metrics['high_risk_percentage'],
            'medium_risk_percentage': metrics['medium_risk_percentage'],
            'low_risk_percentage': metrics['low_risk_percentage'],
            'feature_importance': calculate_feature_importance(),
            'customer_segments': calculate_customer_segments()
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Dashboard data error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
