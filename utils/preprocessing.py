"""
utils/preprocessing.py
Data validation, cleaning, inspection, and scaling utilities.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler

STANDARD_V_COLS = [f'V{i}' for i in range(1, 29)]
REQUIRED_MIN_COLS = STANDARD_V_COLS + ['Amount']

def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Validates the uploaded dataset structure, types, missing values, and target.
    Supports standard Kaggle Credit Card Fraud dataset format.
    """
    errors = []
    warnings = []
    
    if df is None or not isinstance(df, pd.DataFrame):
        return {
            'is_valid': False,
            'errors': ['Uploaded file is not a valid tabular dataset.'],
            'warnings': [],
            'row_count': 0,
            'col_count': 0,
            'has_target': False
        }
        
    row_count, col_count = df.shape
    columns = list(df.columns)
    
    if row_count < 10:
        errors.append(f'Dataset has only {row_count} rows. At least 10 rows are required for analysis.')
        
    # Check for core columns
    missing_required = [c for c in REQUIRED_MIN_COLS if c not in df.columns]
    if missing_required:
        errors.append(f'Missing required feature columns: {", ".join(missing_required[:5])}' + 
                      (f' and {len(missing_required)-5} more' if len(missing_required) > 5 else '') + 
                      '. Expected Kaggle schema (V1 to V28, Amount, and optionally Time, Class).')
        
    # Check for target column
    has_target = 'Class' in df.columns
    if not has_target:
        warnings.append('Target column "Class" not found. The model can make predictions, but cannot evaluate ground-truth accuracy.')
    
    # Missing values check
    missing_series = df.isnull().sum()
    total_missing = int(missing_series.sum())
    missing_by_col = missing_series[missing_series > 0].to_dict()
    
    if total_missing > 0:
        warnings.append(f'Dataset contains {total_missing} missing value(s) across {len(missing_by_col)} column(s). These will be imputed with column medians.')

    # Non-numeric columns check
    non_numeric_cols = []
    for col in df.columns:
        if col in REQUIRED_MIN_COLS or col in ['Time', 'Class']:
            if not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric_cols.append(col)
                
    if non_numeric_cols:
        errors.append(f'Non-numeric values found in numerical column(s): {", ".join(non_numeric_cols[:4])}.')

    # Class distribution if available
    genuine_count = 0
    fraud_count = 0
    fraud_rate = 0.0
    if has_target and errors == []:
        class_counts = df['Class'].value_counts()
        genuine_count = int(class_counts.get(0, 0))
        fraud_count = int(class_counts.get(1, 0))
        total_valid = genuine_count + fraud_count
        if total_valid > 0:
            fraud_rate = (fraud_count / total_valid) * 100
        if fraud_count == 0:
            warnings.append('Warning: Target column "Class" contains 0 fraud instances (Class=1). Model training requires at least some fraud cases.')

    is_valid = len(errors) == 0

    return {
        'is_valid': is_valid,
        'has_target': has_target,
        'errors': errors,
        'warnings': warnings,
        'row_count': row_count,
        'col_count': col_count,
        'columns': columns,
        'total_missing': total_missing,
        'missing_by_col': missing_by_col,
        'genuine_count': genuine_count,
        'fraud_count': fraud_count,
        'fraud_rate': round(fraud_rate, 3)
    }

def clean_and_prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans missing values by imputing with median for numeric columns.
    Ensures V1-V28 and Amount are properly typed float64.
    """
    df_clean = df.copy()
    for col in df_clean.columns:
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            if df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    return df_clean

def preprocess_features(df: pd.DataFrame, scaler=None, fit_scaler=False):
    """
    Extracts feature matrix X and target y.
    Scales the 'Amount' feature with RobustScaler (or StandardScaler) to prevent outlier distortion.
    Leaves V1..V28 (already PCA-transformed) unscaled or standard.
    Returns (X, y, feature_names, scaler).
    """
    df_clean = clean_and_prepare_dataset(df)
    
    # Feature columns: standard V1-V28 plus Amount (and Time if present)
    feature_cols = [c for c in df_clean.columns if c not in ['Class', 'id', 'Unnamed: 0']]
    
    # Sort feature columns consistently: Time (if present), V1..V28, Amount
    ordered_cols = []
    if 'Time' in feature_cols:
        ordered_cols.append('Time')
    for c in STANDARD_V_COLS:
        if c in feature_cols:
            ordered_cols.append(c)
    if 'Amount' in feature_cols:
        ordered_cols.append('Amount')
    # Any other remaining columns
    for c in feature_cols:
        if c not in ordered_cols:
            ordered_cols.append(c)
            
    X = df_clean[ordered_cols].copy()
    
    # Scale Amount and Time (if present)
    cols_to_scale = [c for c in ['Amount', 'Time'] if c in X.columns]
    
    if cols_to_scale:
        if fit_scaler or scaler is None:
            scaler = RobustScaler()
            X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
        else:
            X[cols_to_scale] = scaler.transform(X[cols_to_scale])
            
    y = df_clean['Class'].values if 'Class' in df_clean.columns else None
    
    return X, y, ordered_cols, scaler

def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Computes a comprehensive statistical summary of the dataset for the UI.
    """
    val = validate_dataset(df)
    if not val['is_valid']:
        return {'is_valid': False, 'errors': val['errors']}
        
    summary = {
        'is_valid': True,
        'row_count': val['row_count'],
        'col_count': val['col_count'],
        'has_target': val['has_target'],
        'columns': val['columns'],
        'total_missing': val['total_missing'],
        'missing_by_col': val['missing_by_col'],
        'genuine_count': val['genuine_count'],
        'fraud_count': val['fraud_count'],
        'fraud_rate': val['fraud_rate'],
        'amount_stats': {}
    }
    
    if 'Amount' in df.columns:
        summary['amount_stats'] = {
            'overall_mean': round(float(df['Amount'].mean()), 2),
            'overall_median': round(float(df['Amount'].median()), 2),
            'overall_max': round(float(df['Amount'].max()), 2),
            'overall_min': round(float(df['Amount'].min()), 2),
            'overall_std': round(float(df['Amount'].std()), 2)
        }
        if val['has_target']:
            gen_amt = df[df['Class'] == 0]['Amount']
            frd_amt = df[df['Class'] == 1]['Amount']
            summary['amount_stats']['genuine_mean'] = round(float(gen_amt.mean()), 2) if len(gen_amt) > 0 else 0.0
            summary['amount_stats']['fraud_mean'] = round(float(frd_amt.mean()), 2) if len(frd_amt) > 0 else 0.0
            summary['amount_stats']['genuine_max'] = round(float(gen_amt.max()), 2) if len(gen_amt) > 0 else 0.0
            summary['amount_stats']['fraud_max'] = round(float(frd_amt.max()), 2) if len(frd_amt) > 0 else 0.0

    return summary
