"""
utils/prediction.py
Model loading, batch and single transaction inference, risk scoring,
and Explainable AI (XAI) feature analysis.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from utils.preprocessing import preprocess_features

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_FILE = os.path.join(MODELS_DIR, 'fraud_model.joblib')
METADATA_FILE = os.path.join(MODELS_DIR, 'model_metadata.json')

_CACHED_ARTIFACTS = None
_CACHED_METADATA = None

def get_model_status():
    """Returns whether a trained model is currently available and its details."""
    has_model = os.path.exists(MODEL_FILE)
    metadata = load_metadata() if has_model else None
    return {
        'is_loaded': has_model,
        'model_name': metadata.get('primary_model', {}).get('model_name', 'Random Forest') if metadata else 'None',
        'trained_at': metadata.get('trained_at', 'Not trained') if metadata else 'Not trained',
        'accuracy': metadata.get('primary_model', {}).get('accuracy', 0) if metadata else 0,
        'recall': metadata.get('primary_model', {}).get('recall', 0) if metadata else 0,
        'precision': metadata.get('primary_model', {}).get('precision', 0) if metadata else 0,
        'f1_score': metadata.get('primary_model', {}).get('f1_score', 0) if metadata else 0,
        'roc_auc': metadata.get('primary_model', {}).get('roc_auc', 0) if metadata else 0,
        'pr_auc': metadata.get('primary_model', {}).get('pr_auc', 0) if metadata else 0,
    }

def load_metadata():
    global _CACHED_METADATA
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, 'r') as f:
                _CACHED_METADATA = json.load(f)
                return _CACHED_METADATA
        except Exception:
            return None
    return None

def load_model_artifacts():
    global _CACHED_ARTIFACTS
    if _CACHED_ARTIFACTS is not None:
        return _CACHED_ARTIFACTS
    if not os.path.exists(MODEL_FILE):
        return None
    _CACHED_ARTIFACTS = joblib.load(MODEL_FILE)
    return _CACHED_ARTIFACTS

def reload_model_cache():
    global _CACHED_ARTIFACTS, _CACHED_METADATA
    _CACHED_ARTIFACTS = None
    _CACHED_METADATA = None
    return load_model_artifacts()

def get_risk_category(prob: float) -> str:
    """Classifies risk level based on fraud probability."""
    if prob >= 0.70:
        return 'High Risk'
    elif prob >= 0.35:
        return 'Medium Risk'
    else:
        return 'Low Risk'

def predict_dataframe(df: pd.DataFrame):
    """
    Runs model inference on a DataFrame.
    Returns:
        results_df: DataFrame with predictions and risk scores
        summary_stats: High-level metrics for cards
    """
    artifacts = load_model_artifacts()
    if artifacts is None:
        raise FileNotFoundError("Trained model not found. Please train or initialize the model first.")
        
    model = artifacts['model']
    scaler = artifacts['scaler']
    feature_names = artifacts['feature_names']
    
    # Preprocess
    X, y, ordered_cols, _ = preprocess_features(df, scaler=scaler, fit_scaler=False)
    
    # Ensure all required features exist and align
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_names]
    
    # Inference
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    
    # Build enriched result
    results = pd.DataFrame()
    results['Transaction_ID'] = [f"TX-{i+1:05d}" for i in range(len(df))]
    results['Row_Index'] = range(len(df))
    
    if 'Time' in df.columns:
        results['Time'] = df['Time'].values
    else:
        results['Time'] = 0.0
        
    if 'Amount' in df.columns:
        results['Amount'] = df['Amount'].values
    else:
        results['Amount'] = 0.0
        
    if 'Class' in df.columns:
        results['Actual_Class'] = df['Class'].astype(int).values
        results['Actual_Label'] = results['Actual_Class'].map({0: 'Genuine', 1: 'Fraudulent'})
    else:
        results['Actual_Class'] = None
        results['Actual_Label'] = 'Unknown'
        
    results['Predicted_Class'] = preds.astype(int)
    results['Prediction'] = results['Predicted_Class'].map({0: 'Genuine', 1: 'Fraudulent'})
    results['Fraud_Probability'] = np.round(probs, 4)
    results['Fraud_Probability_Pct'] = np.round(probs * 100, 2)
    results['Risk_Level'] = [get_risk_category(p) for p in probs]
    
    # Copy V1-V28 columns into results for transaction inspection
    for v in [f'V{i}' for i in range(1, 29)]:
        if v in df.columns:
            results[v] = df[v].values
        else:
            results[v] = 0.0
            
    # Calculate high-level summary
    total = len(results)
    fraud_count = int((results['Predicted_Class'] == 1).sum())
    genuine_count = total - fraud_count
    fraud_rate = round((fraud_count / total) * 100, 2) if total > 0 else 0.0
    
    high_risk_count = int((results['Risk_Level'] == 'High Risk').sum())
    med_risk_count = int((results['Risk_Level'] == 'Medium Risk').sum())
    low_risk_count = int((results['Risk_Level'] == 'Low Risk').sum())
    
    # Amounts
    fraud_rows = results[results['Predicted_Class'] == 1]
    genuine_rows = results[results['Predicted_Class'] == 0]
    
    avg_fraud_amount = round(float(fraud_rows['Amount'].mean()), 2) if len(fraud_rows) > 0 else 0.0
    avg_genuine_amount = round(float(genuine_rows['Amount'].mean()), 2) if len(genuine_rows) > 0 else 0.0
    total_suspicious_amount = round(float(fraud_rows['Amount'].sum()), 2) if len(fraud_rows) > 0 else 0.0
    
    summary = {
        'total_transactions': total,
        'genuine_count': genuine_count,
        'fraud_count': fraud_count,
        'fraud_rate': fraud_rate,
        'high_risk_count': high_risk_count,
        'med_risk_count': med_risk_count,
        'low_risk_count': low_risk_count,
        'avg_fraud_amount': avg_fraud_amount,
        'avg_genuine_amount': avg_genuine_amount,
        'total_suspicious_amount': total_suspicious_amount
    }
    
    return results, summary

def explain_transaction(row_data: dict, artifacts=None):
    """
    Provides Explainable AI (XAI) breakdown for a single transaction.
    Computes feature deviations from normal genuine baseline weighted by
    Random Forest global feature importance.
    """
    if artifacts is None:
        artifacts = load_model_artifacts()
    if artifacts is None:
        return {'explanation': 'Model artifacts not found.', 'top_factors': []}
        
    feature_names = artifacts.get('feature_names', [])
    rf_model = artifacts.get('model')
    baseline_means = artifacts.get('baseline_means', {})
    baseline_stds = artifacts.get('baseline_stds', {})
    
    # Global feature importances
    global_importances = dict(zip(feature_names, rf_model.feature_importances_))
    
    # Calculate anomaly contribution for each feature
    contributions = []
    for feat in feature_names:
        if feat in row_data:
            val = float(row_data[feat])
            mean = baseline_means.get(feat, 0.0)
            std = baseline_stds.get(feat, 1.0)
            if std == 0:
                std = 1.0
            
            # z-score deviation from genuine baseline
            z_score = (val - mean) / std
            imp = global_importances.get(feat, 0.01)
            
            # Impact score considers both deviation magnitude and feature importance
            impact_score = abs(z_score) * imp
            
            # Direction description
            if z_score > 1.5:
                direction = 'Abnormally High'
                desc = f'Exceeds normal genuine baseline by +{abs(z_score):.2f} standard deviations'
            elif z_score < -1.5:
                direction = 'Abnormally Low'
                desc = f'Drops significantly below genuine baseline by -{abs(z_score):.2f} standard deviations'
            else:
                direction = 'Normal Range'
                desc = f'Within standard transaction limits ({z_score:+.2f} SD)'
                
            contributions.append({
                'feature': feat,
                'value': round(val, 4),
                'baseline_mean': round(mean, 3),
                'z_score': round(z_score, 2),
                'importance': round(imp * 100, 2),
                'impact_score': round(impact_score, 4),
                'direction': direction,
                'description': desc
            })
            
    # Sort by impact score descending
    contributions.sort(key=lambda x: x['impact_score'], reverse=True)
    top_5 = contributions[:5]
    
    # Synthesize plain-language explanation
    pred_prob = float(row_data.get('Fraud_Probability', 0.0))
    risk_level = row_data.get('Risk_Level', get_risk_category(pred_prob))
    amount = float(row_data.get('Amount', 0.0))
    
    if risk_level == 'High Risk':
        summary_text = (
            f"This transaction exhibits acute behavioral anomalies. Multiple PCA features "
            f"({', '.join([f['feature'] for f in top_5[:3]])}) deviated severely from typical legitimate transaction patterns, "
            f"driving the fraud probability to {pred_prob*100:.1f}%."
        )
        recommendation = (
            "IMMEDIATE ACTION REQUIRED: Place a temporary hold on the transaction. "
            "Send an automated high-priority push verification to the cardholder and initiate manual fraud desk review."
        )
    elif risk_level == 'Medium Risk':
        summary_text = (
            f"This transaction shows moderate risk indicators ({top_5[0]['feature']} and {top_5[1]['feature']} show abnormal patterns). "
            f"The fraud probability is {pred_prob*100:.1f}%, warranting secondary authentication."
        )
        recommendation = (
            "RECOMMENDED ACTION: Enforce Step-Up Multi-Factor Authentication (SMS OTP or Mobile Banking Biometric) "
            "before clearing the payment."
        )
    else:
        summary_text = (
            f"Transaction parameters are aligned with typical cardholder activity. "
            f"Deviation metrics across all PCA components and amount (${amount:.2f}) remain within safe operational bounds."
        )
        recommendation = (
            "APPROVED: Standard payment processing. Continue routine heuristic monitoring."
        )
        
    return {
        'summary': summary_text,
        'recommendation': recommendation,
        'top_factors': top_5,
        'all_factors': contributions[:10],
        'methodology_note': 'Explanations derived from feature deviation against genuine baseline, weighted by Random Forest feature importances.'
    }
