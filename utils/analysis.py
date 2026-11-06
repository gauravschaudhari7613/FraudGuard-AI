"""
utils/analysis.py
Statistical aggregation, chart data preparation, pattern detection,
and cybersecurity fraud prevention recommendations.
"""

import numpy as np
import pandas as pd

def generate_dashboard_analytics(results_df: pd.DataFrame, summary: dict):
    """
    Prepares structured chart data and insight metrics for Dashboard & Analysis views.
    """
    if results_df is None or len(results_df) == 0:
        return None
        
    total_tx = len(results_df)
    fraud_tx = results_df[results_df['Predicted_Class'] == 1]
    genuine_tx = results_df[results_df['Predicted_Class'] == 0]
    
    # 1. Doughnut Chart: Genuine vs Fraudulent
    distribution_chart = {
        'labels': ['Legitimate Transactions', 'Fraudulent Transactions'],
        'counts': [int(len(genuine_tx)), int(len(fraud_tx))],
        'percentages': [
            round((len(genuine_tx) / total_tx) * 100, 2) if total_tx > 0 else 0,
            round((len(fraud_tx) / total_tx) * 100, 2) if total_tx > 0 else 0
        ],
        'colors': ['#10B981', '#EF4444']
    }
    
    # 2. Risk Distribution Chart
    risk_counts = results_df['Risk_Level'].value_counts()
    risk_chart = {
        'labels': ['Low Risk', 'Medium Risk', 'High Risk'],
        'counts': [
            int(risk_counts.get('Low Risk', 0)),
            int(risk_counts.get('Medium Risk', 0)),
            int(risk_counts.get('High Risk', 0))
        ],
        'colors': ['#10B981', '#F59E0B', '#EF4444']
    }
    
    # 3. Activity / Time Timeline Chart
    # If Time column is present, group into 12 temporal intervals
    has_time = 'Time' in results_df.columns and results_df['Time'].max() > 0
    if has_time:
        max_time = results_df['Time'].max()
        bins = np.linspace(0, max_time, 13)
        time_labels = [f"T+{int(bins[i]//3600)}h-{int(bins[i+1]//3600)}h" for i in range(12)]
        
        # Cut results into bins
        cats = pd.cut(results_df['Time'], bins=bins, labels=time_labels, include_lowest=True)
        timeline_df = results_df.groupby([cats, 'Predicted_Class'], observed=False).size().unstack(fill_value=0)
        
        timeline_chart = {
            'type': 'time',
            'labels': time_labels,
            'genuine_series': [int(v) for v in timeline_df.get(0, [0]*12)],
            'fraud_series': [int(v) for v in timeline_df.get(1, [0]*12)]
        }
    else:
        # Amount bracket distribution
        amount_bins = [0, 20, 50, 100, 250, 500, 2000, 100000]
        amount_labels = ['$0-$20', '$20-$50', '$50-$100', '$100-$250', '$250-$500', '$500-$2k', '$2k+']
        cats = pd.cut(results_df['Amount'], bins=amount_bins, labels=amount_labels, include_lowest=True)
        amt_df = results_df.groupby([cats, 'Predicted_Class'], observed=False).size().unstack(fill_value=0)
        
        timeline_chart = {
            'type': 'amount',
            'labels': amount_labels,
            'genuine_series': [int(v) for v in amt_df.get(0, [0]*7)],
            'fraud_series': [int(v) for v in amt_df.get(1, [0]*7)]
        }
        
    # 4. Amount Statistics Comparison
    gen_amt = genuine_tx['Amount']
    frd_amt = fraud_tx['Amount']
    
    amount_comparison = {
        'avg_genuine': round(float(gen_amt.mean()), 2) if len(gen_amt) > 0 else 0.0,
        'avg_fraud': round(float(frd_amt.mean()), 2) if len(frd_amt) > 0 else 0.0,
        'max_genuine': round(float(gen_amt.max()), 2) if len(gen_amt) > 0 else 0.0,
        'max_fraud': round(float(frd_amt.max()), 2) if len(frd_amt) > 0 else 0.0,
        'median_genuine': round(float(gen_amt.median()), 2) if len(gen_amt) > 0 else 0.0,
        'median_fraud': round(float(frd_amt.median()), 2) if len(frd_amt) > 0 else 0.0
    }
    
    # 5. Most Suspicious Transactions (Sorted by fraud probability descending)
    top_suspicious = results_df.sort_values(by='Fraud_Probability', ascending=False).head(10)
    suspicious_list = []
    for _, row in top_suspicious.iterrows():
        suspicious_list.append({
            'transaction_id': row['Transaction_ID'],
            'row_index': int(row['Row_Index']),
            'time': round(float(row.get('Time', 0)), 1),
            'amount': round(float(row['Amount']), 2),
            'prediction': row['Prediction'],
            'probability': round(float(row['Fraud_Probability_Pct']), 2),
            'risk_level': row['Risk_Level']
        })
        
    most_suspicious = suspicious_list[0] if suspicious_list else None

    return {
        'distribution_chart': distribution_chart,
        'risk_chart': risk_chart,
        'timeline_chart': timeline_chart,
        'amount_comparison': amount_comparison,
        'suspicious_list': suspicious_list,
        'most_suspicious': most_suspicious
    }

def get_prevention_recommendations():
    """
    Returns structured cybersecurity and risk mitigation recommendations
    tailored for financial institutions and cardholders.
    """
    return [
        {
            'id': 1,
            'title': 'Mandatory Multi-Factor Authentication (MFA / 3-D Secure)',
            'icon': 'shield-check',
            'severity': 'Critical',
            'badge_color': 'danger',
            'description': 'Enforce mandatory 3DS 2.0 biometric verification or dynamic time-based OTP for transactions exceeding $100 or flagged as Medium/High Risk.',
            'action': 'Implement risk-based adaptive step-up authentication during checkout.'
        },
        {
            'id': 2,
            'title': 'Dynamic Daily & Velocity Spending Limits',
            'icon': 'sliders-h',
            'severity': 'High',
            'badge_color': 'warning',
            'description': 'Configure proactive daily velocity thresholds (maximum transactions per hour and aggregate 24-hour spending caps).',
            'action': 'Automatically throttle rapid successive transactions on the same payment instrument.'
        },
        {
            'id': 3,
            'title': 'Instant Real-Time Transaction Notifications',
            'icon': 'bell',
            'severity': 'High',
            'badge_color': 'warning',
            'description': 'Provide instant push notifications and SMS alerts with one-click "Report Fraud" response capabilities for all card-not-present (CNP) purchases.',
            'action': 'Empower customers to immediately dispute unknown transactions within seconds of authorization.'
        },
        {
            'id': 4,
            'title': 'Micro-Authorization Anomaly Detection',
            'icon': 'search-dollar',
            'severity': 'Medium',
            'badge_color': 'info',
            'description': 'Fraudsters frequently execute small probe transactions ($1 - $5) to test stolen card validity before attempting large balance extractions.',
            'action': 'Trigger automated security audits when micro-charges are followed by high-ticket orders.'
        },
        {
            'id': 5,
            'title': 'Immediate Suspicious Card Freezing Protocol',
            'icon': 'lock',
            'severity': 'Critical',
            'badge_color': 'danger',
            'description': 'Allow users and fraud analysts to execute instant temporary card freezes through mobile banking apps without permanently closing the account.',
            'action': 'Provide reversible security locks while investigations are ongoing.'
        },
        {
            'id': 6,
            'title': 'Geolocation & Device Fingerprint Matching',
            'icon': 'map-marker-alt',
            'severity': 'Medium',
            'badge_color': 'info',
            'description': 'Cross-reference transaction POS location or IP geodata against the user’s customary mobile device coordinates (impossible speed / travel checks).',
            'action': 'Challenge or decline transactions originating from foreign or high-risk geographic clusters.'
        },
        {
            'id': 7,
            'title': '24/7 Dedicated Dispute & Escalation Hotline',
            'icon': 'phone-volume',
            'severity': 'Standard',
            'badge_color': 'secondary',
            'description': 'Maintain dedicated high-priority response channels for cardholders discovering unauthorized ledger charges.',
            'action': 'Guarantee immediate provisional credit issuance and chargeback filing under Visa/Mastercard zero-liability policies.'
        }
    ]
