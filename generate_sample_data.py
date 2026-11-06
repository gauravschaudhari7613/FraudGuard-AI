"""
generate_sample_data.py
Generates a realistic credit card fraud detection dataset strictly conforming
to the standard Kaggle dataset schema:
Columns: Time, V1 to V28, Amount, Class
"""

import os
import numpy as np
import pandas as pd

def generate_creditcard_dataset(num_records=3000, fraud_ratio=0.032, random_state=42):
    np.random.seed(random_state)
    
    num_fraud = int(num_records * fraud_ratio)
    num_genuine = num_records - num_fraud
    
    # 1. Generate Genuine Transactions (Class = 0)
    # Time: distributed across 48 hours (0 to 172800 seconds) with diurnal pattern
    time_genuine = np.sort(np.random.uniform(0, 172800, num_genuine))
    
    # V1 to V28 for genuine transactions: PCA components near standard normal
    features_genuine = np.random.normal(loc=0.0, scale=1.0, size=(num_genuine, 28))
    
    # Amount for genuine: mostly smaller everyday transactions with occasional large ones
    # Log-normal distribution
    amount_genuine = np.random.lognormal(mean=3.2, sigma=1.2, size=num_genuine)
    amount_genuine = np.clip(amount_genuine, 0.50, 2400.00)
    amount_genuine = np.round(amount_genuine, 2)
    
    # 2. Generate Fraudulent Transactions (Class = 1)
    # Fraudsters often strike during night hours or in quick bursts
    time_fraud = np.sort(np.random.choice(
        np.concatenate([
            np.random.uniform(7200, 21600, int(num_fraud * 0.4)),    # 2am - 6am Day 1
            np.random.uniform(93600, 108000, int(num_fraud * 0.4)),  # 2am - 6am Day 2
            np.random.uniform(0, 172800, num_fraud - 2 * int(num_fraud * 0.4))
        ]),
        size=num_fraud,
        replace=False
    ))
    
    # V1 to V28 for fraudulent transactions:
    # Based on empirical Kaggle Credit Card Fraud PCA characteristics:
    # V14, V12, V10, V17 are heavily negative; V4, V11 are heavily positive.
    features_fraud = np.random.normal(loc=0.0, scale=1.0, size=(num_fraud, 28))
    # Apply characteristic fraud offsets
    features_fraud[:, 3]  += np.random.normal(loc=3.2, scale=1.1, size=num_fraud)   # V4
    features_fraud[:, 9]  += np.random.normal(loc=-3.8, scale=1.3, size=num_fraud)  # V10
    features_fraud[:, 10] += np.random.normal(loc=2.9, scale=1.0, size=num_fraud)   # V11
    features_fraud[:, 11] += np.random.normal(loc=-4.5, scale=1.4, size=num_fraud)  # V12
    features_fraud[:, 13] += np.random.normal(loc=-5.8, scale=1.5, size=num_fraud)  # V14
    features_fraud[:, 16] += np.random.normal(loc=-3.9, scale=1.2, size=num_fraud)  # V17
    features_fraud[:, 0]  += np.random.normal(loc=-2.1, scale=1.2, size=num_fraud)  # V1
    features_fraud[:, 1]  += np.random.normal(loc=2.0, scale=1.3, size=num_fraud)   # V2
    features_fraud[:, 2]  += np.random.normal(loc=-2.8, scale=1.4, size=num_fraud)  # V3
    
    # Amount for fraud: Bimodal (micro-test authorization amounts or high-value extraction)
    amount_types = np.random.choice(['micro', 'high', 'medium'], size=num_fraud, p=[0.25, 0.45, 0.30])
    amount_fraud = np.zeros(num_fraud)
    for i, atype in enumerate(amount_types):
        if atype == 'micro':
            amount_fraud[i] = np.random.uniform(1.00, 9.99)
        elif atype == 'high':
            amount_fraud[i] = np.random.uniform(350.00, 1850.00)
        else:
            amount_fraud[i] = np.random.uniform(45.00, 320.00)
    amount_fraud = np.round(amount_fraud, 2)
    
    # 3. Combine Genuine & Fraud DataFrames
    cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
    
    df_gen = pd.DataFrame(features_genuine, columns=[f'V{i}' for i in range(1, 29)])
    df_gen.insert(0, 'Time', time_genuine)
    df_gen['Amount'] = amount_genuine
    df_gen['Class'] = 0
    
    df_frd = pd.DataFrame(features_fraud, columns=[f'V{i}' for i in range(1, 29)])
    df_frd.insert(0, 'Time', time_fraud)
    df_frd['Amount'] = amount_fraud
    df_frd['Class'] = 1
    
    df = pd.concat([df_gen, df_frd], ignore_index=True)
    # Shuffle dataset to realistically interleave transactions
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    df['Time'] = np.round(df['Time'], 1)
    
    return df

if __name__ == '__main__':
    out_dir = os.path.join(os.path.dirname(__file__), 'dataset')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'sample_creditcard.csv')
    
    print(f"Generating realistic sample dataset ({out_path})...")
    df = generate_creditcard_dataset(num_records=3000, fraud_ratio=0.035, random_state=42)
    df.to_csv(out_path, index=False)
    print(f"Dataset successfully created! Total: {len(df)} records, Fraud: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
