"""
test_workflow.py
Tests upload workflow, model retraining, and sample reset.
"""

import os
import io
import pandas as pd
from app import app, CURRENT_DATA_CACHE

def run_workflow_tests():
    client = app.test_client()
    app.config['TESTING'] = True

    print("\n[1/3] Testing Retrain Model POST /train...")
    res = client.post('/train', follow_redirects=True)
    assert res.status_code == 200
    assert b"Machine Learning model successfully retrained" in res.data
    print("  -> Model retrain endpoint successfully completed.")

    print("\n[2/3] Testing CSV Dataset Upload POST /dataset/upload...")
    # Create small valid test dataframe
    from generate_sample_data import generate_creditcard_dataset
    df_test = generate_creditcard_dataset(num_records=50, fraud_ratio=0.08)
    csv_bytes = io.BytesIO()
    df_test.to_csv(csv_bytes, index=False)
    csv_bytes.seek(0)

    data = {
        'dataset_file': (csv_bytes, 'test_upload_50.csv')
    }
    res = client.post('/dataset/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
    assert res.status_code == 200
    assert b"successfully uploaded and verified" in res.data
    assert CURRENT_DATA_CACHE['filename'] == 'test_upload_50.csv'
    print("  -> Dataset upload and dynamic cache update succeeded.")

    print("\n[3/3] Testing Reset to Sample Dataset POST /dataset/load-sample...")
    res = client.post('/dataset/load-sample', follow_redirects=True)
    assert res.status_code == 200
    assert CURRENT_DATA_CACHE['filename'] == 'sample_creditcard.csv'
    print("  -> Reset to sample dataset succeeded.")

    print("\nAll workflow and mutation tests passed cleanly!")

if __name__ == '__main__':
    run_workflow_tests()
