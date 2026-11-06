# FraudGuard AI – Intelligent Credit Card Fraud Detection & Prevention System

> **B.Tech / College Mini Project**  
> An enterprise-grade, fintech-inspired cybersecurity web application that detects fraudulent credit card transactions, evaluates risk probabilities, provides Explainable AI (XAI) feature forensics, and recommends actionable cybersecurity prevention controls.

---

## 📌 Project Overview

**FraudGuard AI** uses Supervised Machine Learning to analyze credit card transactions in real-time or via batch CSV dataset ingestion. It is engineered with a dark navy/charcoal fintech cybersecurity user interface inspired by modern fraud-monitoring dashboards.

### Core Objectives:
1. **Detect Fraudulent Transactions**: Accurately flag fraudulent activities in highly imbalanced transaction streams.
2. **Distinguish Genuine vs. Fraudulent Transactions**: Minimize false alarms while maintaining high sensitivity to fraud attempts.
3. **Forensic Feature Explainability (XAI)**: Reveal *why* a transaction was flagged by evaluating PCA feature deviations against legitimate baselines.
4. **Interactive Fintech Analytics**: Visualize transaction distribution, threat levels, time velocity, and model confusion matrices.
5. **Practical Fraud Prevention**: Recommend cybersecurity directives (such as dynamic step-up 2FA, daily velocity thresholds, geofencing, and card freezing).

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **Backend Framework** | Python 3, Flask |
| **Machine Learning** | Scikit-Learn, Pandas, NumPy, Joblib |
| **Primary Algorithm** | Random Forest Classifier (`class_weight='balanced'`, `n_estimators=100`) |
| **Baseline Model** | Logistic Regression (Balanced) |
| **Frontend UI** | HTML5, Modern CSS3 (Dark Fintech Theme), Bootstrap 5, Font Awesome 6 |
| **Data Visualization** | Chart.js (Doughnut charts, timeline activity bars, feature importance charts) |
| **Dataset Format** | Standard Kaggle Credit Card Fraud Detection CSV (`Time`, `V1` to `V28`, `Amount`, `Class`) |

---

## 🚀 Key Features

### 1. Fraud Monitoring Overview (Dashboard)
- Real-time summary metric cards: **Total Transactions**, **Genuine Transactions**, **Fraudulent Transactions**, and **Fraud Detection Rate**.
- **Classification Doughnut Chart**: Interactive visual breakdown of genuine vs. fraudulent transactions.
- **Activity & Velocity Timeline Chart**: Temporal or amount-tier distribution of transactions.
- **Operational Fraud Insights**: Highlights the highest-risk transaction, average fraud ticket value, and mitigated loss volume.
- **High-Priority Suspicious Transactions**: Direct table preview of the most critical incidents with one-click forensic drill-down.

### 2. Dataset Management
- **CSV Drag & Drop / File Upload**: Upload custom datasets up to 100MB.
- **Automatic Schema Validation**: Validates numerical types, Kaggle PCA columns (`V1`–`V28`), `Amount`, and `Class`.
- **Data Quality & Null Analysis**: Checks and automatically imputes missing values using feature medians.
- **Dataset Preview**: First 10 records inspection table with horizontal scroll.
- **Sample Dataset Generation & Download**: Includes a built-in Kaggle-standard realistic 3,000-transaction sample dataset with 1-click download.
- **One-Click Model Retraining**: Retrain the Random Forest model directly on the uploaded dataset.

### 3. Fraud Detection Engine
- Comprehensive transaction classification table showing **Transaction ID**, **Amount**, **AI Prediction**, **Fraud Probability %**, and **Risk Category** (*Low Risk*, *Medium Risk*, *High Risk*).
- Status color coding: **Green** for Genuine, **Red** for Fraudulent.
- Filter toolbar: Filter by *All*, *Fraudulent Only*, *High Risk*, *Medium Risk*, or *Genuine Only*.
- Search bar to locate transactions by ID or Dollar Amount.
- **CSV Export**: Download predictions, risk scores, and classifications as a CSV file.
- **Real-Time Single Transaction Testing Modal**: Test custom parameters or select presets (*Legitimate Retail*, *High-Risk Fraud Vector*, *Micro-Authorization Probe*) with instant risk assessment.

### 4. Forensic Transaction Details & Explainable AI (XAI)
- Deep-dive into individual transactions.
- **"Why Was This Transaction Flagged?"**: Identifies top influential features contributing to the risk score using feature deviation from legitimate baselines weighted by Random Forest feature importances.
- Tailored **Security Directives**: Specific instructions on whether to hold, prompt 2FA, or clear the transaction.
- **Full PCA Component Vector**: Inspection grid of all 28 PCA features.

### 5. Fraud Analysis & Prevention Recommendations
- Detailed amount comparisons: Genuine vs. Fraudulent average, median, and maximum ticket values.
- Threat tier distribution charts.
- **Model Evaluation Suite**: Holdout test set metrics including **Recall**, **Precision**, **F1-Score**, **PR-AUC**, **ROC-AUC**, and an interactive **Confusion Matrix** (`TN`, `FP`, `FN`, `TP`).
- **Fraud Prevention Recommendations**: 7 actionable security countermeasures (2FA/3DS 2.0, Velocity Limits, Push Notifications, Micro-Probe Audits, Instant Card Freezing, Geofencing, and 24/7 Dispute Hotlines).
- Clear operational disclaimer clarifying decision-support scope.

---

## 📂 Project Structure

```text
fraudguard-ai/
│
├── app.py                      # Main Flask application and REST routes
├── train_model.py              # ML training, evaluation, and serialization script
├── generate_sample_data.py     # Generates Kaggle-compliant realistic sample data
├── test_app.py                 # Automated test suite for routes and API
├── requirements.txt            # Python package dependencies
├── README.md                   # Project documentation
│
├── dataset/
│   ├── .gitkeep
│   └── sample_creditcard.csv   # Pre-generated 3,000-record Kaggle-format CSV
│
├── models/
│   ├── .gitkeep
│   ├── fraud_model.joblib      # Serialized Random Forest model & scaler
│   └── model_metadata.json     # Benchmark metrics and feature importances
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py        # Schema validation, cleaning, and scaling
│   ├── prediction.py           # Model inference, risk classification, and XAI
│   └── analysis.py             # Statistical aggregation, chart data, recommendations
│
├── templates/
│   ├── base.html               # Master layout with sidebar, navbar, and test modal
│   ├── dashboard.html          # Main monitoring overview and summary cards
│   ├── dataset.html            # CSV upload, preview table, and retraining
│   ├── detection.html          # Detection results table, filters, and CSV export
│   ├── transaction_details.html# Forensic XAI breakdown and PCA inspector
│   ├── analysis.html           # Model metrics, confusion matrix, and 7 prevention rules
│   └── about.html              # College mini project documentation and architecture
│
├── static/
│   ├── css/
│   │   └── style.css           # Premium dark navy fintech cybersecurity theme
│   ├── js/
│   │   ├── dashboard.js        # Chart.js initialization (doughnuts, timelines, bars)
│   │   └── main.js             # Mobile toggles and single transaction AJAX tester
│   └── images/
│       └── .gitkeep
│
└── uploads/
    └── .gitkeep                # Destination for uploaded user CSV datasets
```

---

## 📊 Dataset Format Requirements

FraudGuard AI strictly supports the standard Kaggle Credit Card Fraud dataset format:
- **`Time`**: Seconds elapsed between this transaction and the first transaction in the dataset.
- **`V1` to `V28`**: Principal Component Analysis (PCA) numerical features representing confidential cardholder data.
- **`Amount`**: Transaction amount in US Dollars ($).
- **`Class`** *(Optional for prediction, required for training)*: Ground truth label (`0` = Genuine, `1` = Fraudulent).

---

## ⚙️ Installation & Setup (Windows)

### Step 1: Open PowerShell or Command Prompt
Navigate to the project root directory:
```powershell
cd "c:\Users\parag\OneDrive\Desktop\gaurav project"
```

### Step 2: (Optional) Create and Activate a Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Run Automated Verification Tests
Verify that all routes, models, and predictions are 100% operational:
```powershell
python test_app.py
```

### Step 5: Start the Web Application
```powershell
python app.py
```

### Step 6: Open in Your Browser
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🛡️ Model Performance Summary

| Metric | Random Forest (Balanced) | Logistic Regression (Baseline) |
|---|---|---|
| **Holdout Recall (Sensitivity)** | **100.00%** | 96.15% |
| **Precision** | **100.00%** | 92.59% |
| **F1-Score** | **100.00%** | 94.34% |
| **PR-AUC (Precision-Recall)** | **100.00%** | 97.48% |
| **ROC-AUC** | **100.00%** | 98.72% |

*Evaluated on a stratified 25% holdout test set with balanced class weighting to address extreme class imbalance.*

---

## ⚠️ Academic Disclaimer

FraudGuard AI is designed for academic demonstration and cybersecurity risk intelligence analysis. The application performs offline predictive scoring and forensic feature attribution on uploaded tabular datasets. It does not directly connect to live banking core settlement networks or decline live credit card authorizations.
