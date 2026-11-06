"""
test_app.py
Comprehensive test suite verifying all routes, ML predictions,
Explainable AI, dataset handling, and export functionality.
"""

import os
import json
from app import app, CURRENT_DATA_CACHE

def run_tests():
    client = app.test_client()
    app.config['TESTING'] = True
    
    print("\n[1/7] Testing Dashboard Route...")
    res = client.get('/')
    assert res.status_code == 200, f"Dashboard returned {res.status_code}"
    assert b"Fraud Monitoring Overview" in res.data
    print("  -> Dashboard route passed.")

    print("\n[2/7] Testing Dataset Management Route...")
    res = client.get('/dataset')
    assert res.status_code == 200, f"Dataset page returned {res.status_code}"
    assert b"Dataset Management" in res.data
    print("  -> Dataset management route passed.")

    print("\n[3/7] Testing Fraud Detection Route & Filters...")
    res = client.get('/detection')
    assert res.status_code == 200
    assert b"Transaction Risk Detection" in res.data

    res_fraud = client.get('/detection?filter=fraud')
    assert res_fraud.status_code == 200
    
    res_gen = client.get('/detection?filter=genuine')
    assert res_gen.status_code == 200

    res_high = client.get('/detection?filter=high_risk')
    assert res_high.status_code == 200
    print("  -> Fraud detection route & all filters passed.")

    print("\n[4/7] Testing Transaction Forensic Details...")
    res = client.get('/transaction/0')
    assert res.status_code == 200
    assert b"Forensic Analysis" in res.data
    assert b"Explainable AI" in res.data
    print("  -> Transaction details & XAI passed.")

    print("\n[5/7] Testing Analysis & Prevention Page...")
    res = client.get('/analysis')
    assert res.status_code == 200
    assert b"Pattern Intelligence" in res.data
    assert b"Fraud Prevention Recommendations" in res.data
    assert b"Confusion Matrix" in res.data
    print("  -> Analysis & prevention route passed.")

    print("\n[6/7] Testing About Page...")
    res = client.get('/about')
    assert res.status_code == 200
    assert b"AI-Based Credit Card Fraud Detection" in res.data
    print("  -> About page passed.")

    print("\n[7/7] Testing Real-Time Single Transaction Prediction API...")
    payload = {
        'Amount': 850.0,
        'V14': -6.8,
        'V12': -4.5,
        'V10': -3.9,
        'V17': -4.2,
        'V4': 3.5,
        'V11': 3.1
    }
    res = client.post('/detection/predict-single',
                      data=json.dumps(payload),
                      content_type='application/json')
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data['success'] is True
    assert data['prediction'] in ['Fraudulent', 'Genuine']
    assert 'risk_level' in data
    assert 'explanation' in data
    print(f"  -> Single prediction API passed. Predicted: {data['prediction']}, Risk: {data['risk_level']}, Prob: {data['fraud_probability_pct']}%")

    print("\n[BONUS] Testing CSV Export & Sample Download...")
    res_export = client.get('/detection/export-csv')
    assert res_export.status_code == 200
    assert b"Transaction_ID" in res_export.data

    res_sample = client.get('/dataset/download-sample')
    assert res_sample.status_code == 200
    assert b"Time,V1,V2" in res_sample.data
    print("  -> CSV export & sample download passed.")

    print("\n==============================================")
    print(" ALL TESTS PASSED SUCCESSFULLY! (100% HEALTHY)")
    print("==============================================\n")

if __name__ == '__main__':
    run_tests()
