"""
train_model.py
Trains the Machine Learning fraud detection model on a provided dataset.
Uses Random Forest with balanced class weights and evaluates with comprehensive
imbalanced classification metrics (Precision, Recall, F1, PR-AUC, ROC-AUC, Confusion Matrix).
Also trains a Logistic Regression baseline model for comparison.
"""

import os
import json
import datetime
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from utils.preprocessing import preprocess_features, validate_dataset

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_FILE = os.path.join(MODELS_DIR, 'fraud_model.joblib')
METADATA_FILE = os.path.join(MODELS_DIR, 'model_metadata.json')

def train_and_evaluate(dataset_path: str = None, df: pd.DataFrame = None):
    """
    Trains Random Forest and Logistic Regression baseline on the dataset.
    Saves model artifacts and performance metrics.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    if df is None:
        if dataset_path is None:
            dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'sample_creditcard.csv')
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        df = pd.read_csv(dataset_path)
        
    validation = validate_dataset(df)
    if not validation['is_valid']:
        raise ValueError(f"Dataset validation failed: {'; '.join(validation['errors'])}")
        
    if not validation['has_target']:
        raise ValueError("Cannot train model without target column 'Class'.")
        
    if validation['fraud_count'] < 2:
        raise ValueError("Dataset must contain at least 2 fraudulent transactions (Class=1) to perform stratified splitting.")

    # 1. Feature Preprocessing
    X, y, feature_names, scaler = preprocess_features(df, fit_scaler=True)
    
    # 2. Stratified Train-Test Split (avoid data leakage)
    test_size = 0.25
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=42
    )
    
    # Calculate baseline genuine and fraud distributions in splits
    train_genuine = int((y_train == 0).sum())
    train_fraud = int((y_train == 1).sum())
    test_genuine = int((y_test == 0).sum())
    test_fraud = int((y_test == 1).sum())
    
    # 3. Train Random Forest (Primary Model)
    # n_estimators=100, max_depth=12, balanced class weights for imbalanced fraud data
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Evaluate Random Forest on holdout test set
    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    
    rf_cm = confusion_matrix(y_test, rf_preds)
    tn, fp, fn, tp = [int(v) for v in rf_cm.ravel()]
    
    rf_metrics = {
        'model_name': 'Random Forest Classifier',
        'accuracy': round(float(accuracy_score(y_test, rf_preds)), 4),
        'precision': round(float(precision_score(y_test, rf_preds, zero_division=0)), 4),
        'recall': round(float(recall_score(y_test, rf_preds, zero_division=0)), 4),
        'f1_score': round(float(f1_score(y_test, rf_preds, zero_division=0)), 4),
        'roc_auc': round(float(roc_auc_score(y_test, rf_probs)), 4),
        'pr_auc': round(float(average_precision_score(y_test, rf_probs)), 4),
        'confusion_matrix': {
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
        }
    }
    
    # 4. Train Logistic Regression (Baseline Comparison Model)
    lr_model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    lr_model.fit(X_train, y_train)
    
    lr_preds = lr_model.predict(X_test)
    lr_probs = lr_model.predict_proba(X_test)[:, 1]
    lr_cm = confusion_matrix(y_test, lr_preds)
    lr_tn, lr_fp, lr_fn, lr_tp = [int(v) for v in lr_cm.ravel()]
    
    lr_metrics = {
        'model_name': 'Logistic Regression (Baseline)',
        'accuracy': round(float(accuracy_score(y_test, lr_preds)), 4),
        'precision': round(float(precision_score(y_test, lr_preds, zero_division=0)), 4),
        'recall': round(float(recall_score(y_test, lr_preds, zero_division=0)), 4),
        'f1_score': round(float(f1_score(y_test, lr_preds, zero_division=0)), 4),
        'roc_auc': round(float(roc_auc_score(y_test, lr_probs)), 4),
        'pr_auc': round(float(average_precision_score(y_test, lr_probs)), 4),
        'confusion_matrix': {
            'tn': lr_tn, 'fp': lr_fp, 'fn': lr_fn, 'tp': lr_tp
        }
    }
    
    # 5. Extract Feature Importances from Random Forest
    importances = rf_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    top_features = [
        {
            'feature': feature_names[i],
            'importance': round(float(importances[i]), 4),
            'importance_percent': round(float(importances[i] * 100), 2)
        }
        for i in sorted_idx[:10]
    ]
    
    # 6. Compute Feature Means for Genuine Baseline (for individual explainability)
    genuine_mask = (y_train == 0)
    genuine_means = X_train[genuine_mask].mean().to_dict()
    genuine_stds = X_train[genuine_mask].std().replace(0, 1.0).to_dict()
    
    # 7. Save Model Artifacts
    artifacts = {
        'model': rf_model,
        'scaler': scaler,
        'feature_names': feature_names,
        'baseline_means': genuine_means,
        'baseline_stds': genuine_stds
    }
    joblib.dump(artifacts, MODEL_FILE)
    
    # 8. Save Metadata
    metadata = {
        'trained_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_samples': len(df),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'train_distribution': {'genuine': train_genuine, 'fraud': train_fraud},
        'test_distribution': {'genuine': test_genuine, 'fraud': test_fraud},
        'features_used': feature_names,
        'total_features': len(feature_names),
        'primary_model': rf_metrics,
        'baseline_model': lr_metrics,
        'top_feature_importances': top_features
    }
    
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    return metadata

if __name__ == '__main__':
    print("Training FraudGuard AI Machine Learning Models...")
    meta = train_and_evaluate()
    print("\n--- Model Training Summary ---")
    print(f"Algorithm: {meta['primary_model']['model_name']}")
    print(f"Recall: {meta['primary_model']['recall'] * 100:.2f}%")
    print(f"Precision: {meta['primary_model']['precision'] * 100:.2f}%")
    print(f"F1-Score: {meta['primary_model']['f1_score'] * 100:.2f}%")
    print(f"PR-AUC: {meta['primary_model']['pr_auc'] * 100:.2f}%")
    print(f"ROC-AUC: {meta['primary_model']['roc_auc'] * 100:.2f}%")
    print(f"Confusion Matrix: TN={meta['primary_model']['confusion_matrix']['tn']}, "
          f"FP={meta['primary_model']['confusion_matrix']['fp']}, "
          f"FN={meta['primary_model']['confusion_matrix']['fn']}, "
          f"TP={meta['primary_model']['confusion_matrix']['tp']}")
    print(f"\nTop 5 Influential Features: {[f['feature'] for f in meta['top_feature_importances'][:5]]}")
