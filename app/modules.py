from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import os
import traceback

model = joblib.load('../notebooks/final_logistic_regression_model.pkl')
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

def get_default_metrics():
    """Calculate metrics from trained model on test data"""
    try:
        # Load test data
        test_data = pd.read_csv('../data/processed/cleaned_data.csv')
        X_test = test_data.drop(columns=['customerID', 'Churn'])
        
        # Add interaction term
        X_test['MonthlyTenureInteraction'] = X_test['MonthlyCharges'] * X_test['tenure']
        
        # Transform features
        X_transformed = preprocessor.transform(X_test)
        
        # Get predictions and probabilities
        predictions = model.predict(X_transformed)
        probabilities = model.predict_proba(X_transformed)
        
        # Calculate metrics
        churn_rate = (predictions.sum() / len(predictions) * 100)
        avg_confidence = np.mean(np.max(probabilities, axis=1) * 100)
        
        csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'predictions.csv'))
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            total = len(df)
            churn_count = sum(df['prediction'] == 'Churn')
            
            # Calculate probabilities for risk levels
            probabilities = df['probability'].values
            high_risk = np.sum(probabilities >= 0.7)
            medium_risk = np.sum((probabilities >= 0.3) & (probabilities < 0.7))
            low_risk = np.sum(probabilities < 0.3)
        else:
            # Default values if file doesn't exist
            total = 100
            churn_count = 24
            high_risk = 30
            medium_risk = 45
            low_risk = 25

        return {
            'churn_rate': round(churn_rate, 1),
            'avg_confidence': round(avg_confidence, 1),
            'high_risk_count': int(high_risk),
            'medium_risk_count': int(medium_risk),
            'low_risk_count': int(low_risk),
            'high_risk_percentage': round((high_risk/total) * 100, 1),
            'medium_risk_percentage': round((medium_risk/total) * 100, 1),
            'low_risk_percentage': round((low_risk/total) * 100, 1)
        }
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return {
            'churn_rate': 23.6,
            'avg_confidence': 85.7,
            'high_risk_count': 30,
            'medium_risk_count': 45,
            'low_risk_count': 25,
            'high_risk_percentage': 30.0,
            'medium_risk_percentage': 45.0,
            'low_risk_percentage': 25.0
        }

def store_prediction(prediction_data):
    """Store prediction results in CSV file"""
    try:
        df = pd.DataFrame([prediction_data])
        
        # Use absolute path
        csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'predictions.csv'))
        print(f"Storing prediction to: {csv_path}")
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        if os.path.exists(csv_path):
            print("Appending to existing predictions file")
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            print("Creating new predictions file")
            df.to_csv(csv_path, index=False)
            
        print(f"Successfully stored prediction: {prediction_data['prediction']} at {prediction_data['timestamp']}")
            
    except Exception as e:
        print(f"Error storing prediction: {str(e)}")
        raise e

def calculate_customer_segments():
    """Calculate customer risk segments based on model predictions"""
    try:
        # Load test data
        test_data = pd.read_csv('../data/processed/cleaned_data.csv')
        X_test = test_data.drop(columns=['customerID', 'Churn'])
        
        # Add interaction term
        X_test['MonthlyTenureInteraction'] = X_test['MonthlyCharges'] * X_test['tenure']
        
        # Transform features
        X_transformed = preprocessor.transform(X_test)
        
        # Get probabilities of churn
        probabilities = model.predict_proba(X_transformed)[:, 1]
        print(probabilities)
        
        # Define risk segments based on churn probability
        high_risk = np.sum(probabilities >= 0.7)
        medium_risk = np.sum((probabilities >= 0.3) & (probabilities < 0.7))
        low_risk = np.sum(probabilities < 0.3)
        print(f'{high_risk}, {medium_risk}, {low_risk}')
        
        # Calculate distribution percentages
        total = len(probabilities)
        distribution = [
            round((high_risk / total) * 100),
            round((medium_risk / total) * 100),
            round((low_risk / total) * 100)
        ]
        print(distribution)
        
        return {
            'segments': ['High Risk', 'Medium Risk', 'Low Risk'],
            'distribution': distribution
        }
    except Exception as e:
        print(f"Error calculating customer segments: {str(e)}")
        return {
            'segments': ['High Risk', 'Medium Risk', 'Low Risk'],
            'distribution': [30, 45, 25]
        }

def calculate_feature_importance():
    """Calculate feature importance from the model"""
    try:
        # Get feature importance values
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            importances = np.abs(model.coef_[0])
        
        # Get the fitted categorical transformer from the preprocessor
        fitted_cat_transformer = preprocessor.named_transformers_['cat']
        
        # Get all feature names
        feature_names = []
        
        # Add numeric feature names
        feature_names.extend(numeric_features)
        
        # Add categorical feature names with their transformed values
        for i, feature in enumerate(categorical_features):
            categories = fitted_cat_transformer.categories_[i]
            for category in categories:
                feature_names.append(f"{feature}_{category}")
        
        # Create feature importance pairs and sort
        feature_importance_pairs = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        print(feature_importance_pairs)
        # Separate features and importance values
        features = [pair[0] for pair in feature_importance_pairs]
        importance_values = [float(pair[1]) for pair in feature_importance_pairs]
        
        # Normalize importance values to sum to 1
        importance_values = np.array(importance_values)
        importance_values = importance_values / importance_values.sum()
        
        return {
            'features': features,
            'importance': importance_values.tolist()
        }
    except Exception as e:
        print(f"Error calculating feature importance: {str(e)}")
        traceback.print_exc()  # Add this to get more detailed error information
        return {
            'features': ['Contract', 'Tenure', 'MonthlyCharges', 'TotalCharges', 'InternetService'],
            'importance': [0.25, 0.20, 0.18, 0.15, 0.12]
        }
