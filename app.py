"""
app.py
FraudGuard AI – Intelligent Credit Card Fraud Detection & Prevention System
Main Flask Application
"""

import os
import io
import json
import pandas as pd
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, jsonify, send_file, session
)
from werkzeug.utils import secure_filename

from utils.preprocessing import (
    validate_dataset,
    get_dataset_summary
)
from utils.prediction import (
    get_model_status,
    load_metadata,
    reload_model_cache,
    predict_dataframe,
    explain_transaction,
    get_risk_category
)
from utils.analysis import (
    generate_dashboard_analytics,
    get_prevention_recommendations
)
from train_model import train_and_evaluate

app = Flask(__name__)
app.secret_key = 'fraudguard-ai-fintech-cybersecurity-2026'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
DATASET_FOLDER = os.path.join(BASE_DIR, 'dataset')
MODELS_FOLDER = os.path.join(BASE_DIR, 'models')
SAMPLE_DATASET_PATH = os.path.join(DATASET_FOLDER, 'sample_creditcard.csv')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max upload

# In-memory session/cache for current active dataset and predictions
CURRENT_DATA_CACHE = {
    'filepath': SAMPLE_DATASET_PATH if os.path.exists(SAMPLE_DATASET_PATH) else None,
    'filename': 'sample_creditcard.csv' if os.path.exists(SAMPLE_DATASET_PATH) else None,
    'is_sample': True,
    'df': None,
    'results_df': None,
    'summary': None,
    'analytics': None
}

def get_active_dataset():
    """Returns the active DataFrame, loading it if not already in memory."""
    filepath = CURRENT_DATA_CACHE.get('filepath')
    if filepath and os.path.exists(filepath):
        if CURRENT_DATA_CACHE['df'] is None:
            try:
                df = pd.read_csv(filepath)
                CURRENT_DATA_CACHE['df'] = df
            except Exception as e:
                print(f"Error reading dataset: {e}")
                return None
        return CURRENT_DATA_CACHE['df']
    return None

def refresh_predictions():
    """Runs predictions on current active dataset and stores analytics."""
    df = get_active_dataset()
    if df is not None:
        try:
            results_df, summary = predict_dataframe(df)
            analytics = generate_dashboard_analytics(results_df, summary)
            CURRENT_DATA_CACHE['results_df'] = results_df
            CURRENT_DATA_CACHE['summary'] = summary
            CURRENT_DATA_CACHE['analytics'] = analytics
            return results_df, summary, analytics
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, None, None
    return None, None, None

@app.context_processor
def inject_global_status():
    """Injects model status and active dataset info into all Jinja templates."""
    status = get_model_status()
    active_name = CURRENT_DATA_CACHE.get('filename', 'None')
    row_count = len(CURRENT_DATA_CACHE['df']) if CURRENT_DATA_CACHE['df'] is not None else 0
    return {
        'model_status': status,
        'active_dataset_name': active_name,
        'active_dataset_rows': row_count,
        'is_sample_dataset': CURRENT_DATA_CACHE.get('is_sample', False)
    }

@app.template_filter('number_format')
def format_number(val):
    """Formats an integer or float with thousands commas."""
    try:
        return f"{int(val):,}"
    except (ValueError, TypeError):
        return str(val)


# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------

@app.route('/')
@app.route('/dashboard')
def dashboard():
    """Dashboard: Overview statistics, doughnut distribution, activity, recent suspicious."""
    df = get_active_dataset()
    if df is None:
        return render_template('dashboard.html', has_data=False)
        
    if CURRENT_DATA_CACHE['results_df'] is None:
        refresh_predictions()
        
    results_df = CURRENT_DATA_CACHE['results_df']
    summary = CURRENT_DATA_CACHE['summary']
    analytics = CURRENT_DATA_CACHE['analytics']
    
    if results_df is None or summary is None or analytics is None:
        flash("Could not evaluate transactions with the current model.", "danger")
        return render_template('dashboard.html', has_data=False)

    return render_template(
        'dashboard.html',
        has_data=True,
        summary=summary,
        analytics=analytics,
        recent_suspicious=analytics['suspicious_list'][:6]
    )

@app.route('/dataset')
def dataset_management():
    """Dataset Management: Upload, preview first 10 rows, columns list, missing values check."""
    df = get_active_dataset()
    if df is None:
        return render_template('dataset.html', has_data=False)
        
    validation = validate_dataset(df)
    summary = get_dataset_summary(df)
    
    # Preview first 10 records
    preview_df = df.head(10).round(4)
    preview_records = preview_df.to_dict(orient='records')
    columns = list(df.columns)
    
    # Column datatypes summary
    col_types = [{'name': col, 'type': str(df[col].dtype), 'missing': int(df[col].isnull().sum())} for col in columns]

    return render_template(
        'dataset.html',
        has_data=True,
        validation=validation,
        summary=summary,
        columns=columns,
        col_types=col_types,
        preview_records=preview_records,
        filename=CURRENT_DATA_CACHE.get('filename')
    )

@app.route('/dataset/upload', methods=['POST'])
def upload_dataset():
    """Handles CSV upload, schema validation, and sets active dataset."""
    if 'dataset_file' not in request.files:
        flash("No file part provided in request.", "danger")
        return redirect(url_for('dataset_management'))
        
    file = request.files['dataset_file']
    if file.filename == '':
        flash("No file selected for upload.", "warning")
        return redirect(url_for('dataset_management'))
        
    if not file.filename.lower().endswith('.csv'):
        flash("Invalid file format. Please upload a standard CSV (.csv) file.", "danger")
        return redirect(url_for('dataset_management'))
        
    filename = secure_filename(file.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)
    
    try:
        df = pd.read_csv(save_path)
    except Exception as e:
        flash(f"Error parsing CSV file: {str(e)}", "danger")
        return redirect(url_for('dataset_management'))
        
    val = validate_dataset(df)
    if not val['is_valid']:
        err_msg = " ; ".join(val['errors'])
        flash(f"Dataset schema validation failed: {err_msg}", "danger")
        return redirect(url_for('dataset_management'))
        
    # Update cache
    CURRENT_DATA_CACHE['filepath'] = save_path
    CURRENT_DATA_CACHE['filename'] = filename
    CURRENT_DATA_CACHE['is_sample'] = False
    CURRENT_DATA_CACHE['df'] = df
    CURRENT_DATA_CACHE['results_df'] = None
    CURRENT_DATA_CACHE['summary'] = None
    CURRENT_DATA_CACHE['analytics'] = None
    
    refresh_predictions()
    
    msg = f"Dataset '{filename}' successfully uploaded and verified ({val['row_count']:,} records)."
    if val['warnings']:
        msg += f" Note: {' '.join(val['warnings'])}"
    flash(msg, "success")
    return redirect(url_for('dataset_management'))

@app.route('/dataset/load-sample', methods=['POST'])
def load_sample_dataset():
    """Switches active dataset to pre-installed realistic sample dataset."""
    if not os.path.exists(SAMPLE_DATASET_PATH):
        from generate_sample_data import generate_creditcard_dataset
        df = generate_creditcard_dataset(num_records=3000, fraud_ratio=0.035)
        os.makedirs(DATASET_FOLDER, exist_ok=True)
        df.to_csv(SAMPLE_DATASET_PATH, index=False)
        
    CURRENT_DATA_CACHE['filepath'] = SAMPLE_DATASET_PATH
    CURRENT_DATA_CACHE['filename'] = 'sample_creditcard.csv'
    CURRENT_DATA_CACHE['is_sample'] = True
    CURRENT_DATA_CACHE['df'] = pd.read_csv(SAMPLE_DATASET_PATH)
    CURRENT_DATA_CACHE['results_df'] = None
    CURRENT_DATA_CACHE['summary'] = None
    CURRENT_DATA_CACHE['analytics'] = None
    
    refresh_predictions()
    flash("Standard Kaggle-format sample credit card dataset loaded successfully!", "success")
    return redirect(url_for('dashboard'))

@app.route('/dataset/download-sample')
def download_sample_dataset():
    """Allows user to download the sample CSV dataset."""
    if not os.path.exists(SAMPLE_DATASET_PATH):
        from generate_sample_data import generate_creditcard_dataset
        df = generate_creditcard_dataset(num_records=3000, fraud_ratio=0.035)
        os.makedirs(DATASET_FOLDER, exist_ok=True)
        df.to_csv(SAMPLE_DATASET_PATH, index=False)
    return send_file(
        SAMPLE_DATASET_PATH,
        as_attachment=True,
        download_name='sample_creditcard_fraud.csv',
        mimetype='text/csv'
    )

@app.route('/train', methods=['POST'])
def train_model_route():
    """Triggers ML model training on the currently active dataset."""
    df = get_active_dataset()
    if df is None:
        flash("No active dataset to train on. Please upload or load a dataset first.", "warning")
        return redirect(url_for('dataset_management'))
        
    try:
        metadata = train_and_evaluate(df=df)
        reload_model_cache()
        # Invalidate old predictions so they are re-calculated with new model
        CURRENT_DATA_CACHE['results_df'] = None
        refresh_predictions()
        
        rf = metadata['primary_model']
        flash(
            f"Machine Learning model successfully retrained! "
            f"Recall: {rf['recall']*100:.2f}%, Precision: {rf['precision']*100:.2f}%, "
            f"PR-AUC: {rf['pr_auc']*100:.2f}%. Model saved to models/fraud_model.joblib.",
            "success"
        )
        return redirect(url_for('analysis'))
    except Exception as e:
        flash(f"Error during model training: {str(e)}", "danger")
        return redirect(url_for('dataset_management'))

@app.route('/detection')
def fraud_detection():
    """Fraud Detection: Predict whether each transaction is Genuine or Fraudulent with filters."""
    df = get_active_dataset()
    if df is None:
        return render_template('detection.html', has_data=False)
        
    if CURRENT_DATA_CACHE['results_df'] is None:
        refresh_predictions()
        
    results_df = CURRENT_DATA_CACHE['results_df']
    summary = CURRENT_DATA_CACHE['summary']
    
    if results_df is None:
        flash("Model is not ready to make predictions.", "danger")
        return render_template('detection.html', has_data=False)

    # Optional filter parameters: type = 'all', 'fraud', 'genuine', 'high_risk'
    filter_type = request.args.get('filter', 'all')
    search_q = request.args.get('q', '').strip()
    
    filtered_df = results_df.copy()
    if filter_type == 'fraud':
        filtered_df = filtered_df[filtered_df['Predicted_Class'] == 1]
    elif filter_type == 'genuine':
        filtered_df = filtered_df[filtered_df['Predicted_Class'] == 0]
    elif filter_type == 'high_risk':
        filtered_df = filtered_df[filtered_df['Risk_Level'] == 'High Risk']
    elif filter_type == 'medium_risk':
        filtered_df = filtered_df[filtered_df['Risk_Level'] == 'Medium Risk']
        
    if search_q:
        filtered_df = filtered_df[
            filtered_df['Transaction_ID'].str.contains(search_q, case=False) |
            filtered_df['Amount'].astype(str).str.contains(search_q)
        ]
        
    # Limit table display to first 200 rows for high responsiveness
    total_matches = len(filtered_df)
    display_records = filtered_df.head(200).to_dict(orient='records')

    return render_template(
        'detection.html',
        has_data=True,
        summary=summary,
        records=display_records,
        total_matches=total_matches,
        current_filter=filter_type,
        search_query=search_q
    )

@app.route('/detection/export-csv')
def export_predictions_csv():
    """Exports prediction results to downloadable CSV."""
    if CURRENT_DATA_CACHE['results_df'] is None:
        refresh_predictions()
        
    results_df = CURRENT_DATA_CACHE['results_df']
    if results_df is None:
        flash("No predictions available to export.", "warning")
        return redirect(url_for('fraud_detection'))
        
    export_cols = ['Transaction_ID', 'Time', 'Amount', 'Prediction', 'Fraud_Probability_Pct', 'Risk_Level']
    if 'Actual_Label' in results_df.columns:
        export_cols.insert(4, 'Actual_Label')
        
    csv_buffer = io.StringIO()
    results_df[export_cols].to_csv(csv_buffer, index=False)
    csv_bytes = io.BytesIO(csv_buffer.getvalue().encode('utf-8'))
    
    return send_file(
        csv_bytes,
        as_attachment=True,
        download_name='fraudguard_prediction_results.csv',
        mimetype='text/csv'
    )

@app.route('/detection/predict-single', methods=['POST'])
def predict_single_transaction():
    """API endpoint: Real-time inference on a custom or manually entered transaction."""
    try:
        data = request.get_json(force=True)
    except Exception:
        data = request.form.to_dict()
        
    artifacts = CURRENT_DATA_CACHE.get('artifacts')
    # Use load_model_artifacts
    from utils.prediction import load_model_artifacts
    artifacts = load_model_artifacts()
    if artifacts is None:
        return jsonify({'success': False, 'error': 'Trained ML model not found.'}), 400
        
    feature_names = artifacts['feature_names']
    model = artifacts['model']
    scaler = artifacts['scaler']
    
    # Construct single-row DataFrame
    row_dict = {}
    for feat in feature_names:
        val = data.get(feat, 0.0)
        try:
            row_dict[feat] = [float(val)]
        except ValueError:
            row_dict[feat] = [0.0]
            
    df_single = pd.DataFrame(row_dict)
    
    # Scale Amount and Time if in feature_names
    cols_to_scale = [c for c in ['Amount', 'Time'] if c in df_single.columns]
    if cols_to_scale and scaler is not None:
        df_single[cols_to_scale] = scaler.transform(df_single[cols_to_scale])
        
    X = df_single[feature_names]
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0, 1])
    risk_level = get_risk_category(prob)
    
    # Generate XAI explanation
    explanation_input = {k: v[0] for k, v in row_dict.items()}
    explanation_input['Fraud_Probability'] = prob
    explanation_input['Risk_Level'] = risk_level
    explanation_input['Amount'] = float(data.get('Amount', 0.0))
    
    xai = explain_transaction(explanation_input, artifacts)

    return jsonify({
        'success': True,
        'prediction': 'Fraudulent' if pred == 1 else 'Genuine',
        'predicted_class': pred,
        'fraud_probability': round(prob, 4),
        'fraud_probability_pct': round(prob * 100, 2),
        'risk_level': risk_level,
        'amount': float(data.get('Amount', 0.0)),
        'explanation': xai['summary'],
        'recommendation': xai['recommendation'],
        'top_factors': xai['top_factors']
    })

@app.route('/transaction/<int:row_index>')
def transaction_details(row_index):
    """Detailed view for a specific selected transaction with Explainable AI."""
    if CURRENT_DATA_CACHE['results_df'] is None:
        refresh_predictions()
        
    results_df = CURRENT_DATA_CACHE['results_df']
    if results_df is None or row_index < 0 or row_index >= len(results_df):
        flash("Transaction not found or dataset not loaded.", "warning")
        return redirect(url_for('fraud_detection'))
        
    row = results_df.iloc[row_index].to_dict()
    xai = explain_transaction(row)
    
    # Extract V1-V28 features for component inspection table
    v_features = [{'name': f'V{i}', 'value': round(float(row.get(f'V{i}', 0.0)), 4)} for i in range(1, 29)]

    return render_template(
        'transaction_details.html',
        tx=row,
        xai=xai,
        v_features=v_features,
        row_index=row_index,
        total_tx=len(results_df)
    )

@app.route('/analysis')
def fraud_analysis():
    """Fraud Analysis & Prevention: Patterns, metrics, PR-AUC, confusion matrix, recommendations."""
    if CURRENT_DATA_CACHE['results_df'] is None:
        refresh_predictions()
        
    summary = CURRENT_DATA_CACHE['summary']
    analytics = CURRENT_DATA_CACHE['analytics']
    metadata = load_metadata()
    recommendations = get_prevention_recommendations()

    return render_template(
        'analysis.html',
        has_data=summary is not None,
        summary=summary,
        analytics=analytics,
        metadata=metadata,
        recommendations=recommendations
    )

@app.route('/about')
def about_project():
    """About Project: Objectives, college mini-project specs, architecture, tech stack."""
    metadata = load_metadata()
    return render_template('about.html', metadata=metadata)

if __name__ == '__main__':
    # Initialize sample dataset if absent
    if not os.path.exists(SAMPLE_DATASET_PATH):
        from generate_sample_data import generate_creditcard_dataset
        os.makedirs(DATASET_FOLDER, exist_ok=True)
        df_init = generate_creditcard_dataset(num_records=3000, fraud_ratio=0.035)
        df_init.to_csv(SAMPLE_DATASET_PATH, index=False)

    # Initialize model if absent
    if not os.path.exists(os.path.join(MODELS_FOLDER, 'fraud_model.joblib')):
        train_and_evaluate(dataset_path=SAMPLE_DATASET_PATH)

    # Run predictions on sample data
    get_active_dataset()
    refresh_predictions()

    print("\n========================================================")
    print(" FraudGuard AI – Intelligent Credit Card Fraud System")
    print(" Server starting on http://127.0.0.1:5000")
    print("========================================================\n")
    app.run(host='127.0.0.1', port=5000, debug=True)
