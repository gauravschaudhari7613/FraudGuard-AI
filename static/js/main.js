/**
 * FraudGuard AI - Main JavaScript Client Library
 */

document.addEventListener('DOMContentLoaded', function () {
    // Mobile sidebar toggle
    const toggleBtn = document.getElementById('sidebarToggle');
    const sidebar = document.querySelector('.app-sidebar');
    if (toggleBtn && sidebar) {
        toggleBtn.addEventListener('click', function () {
            sidebar.classList.toggle('show');
        });
    }

    // Single Transaction Tester Presets
    const presetSelect = document.getElementById('presetSelect');
    if (presetSelect) {
        presetSelect.addEventListener('change', function () {
            const val = this.value;
            if (val === 'fraud_sample') {
                document.getElementById('inputAmount').value = '485.50';
                document.getElementById('inputV14').value = '-6.85';
                document.getElementById('inputV12').value = '-4.20';
                document.getElementById('inputV10').value = '-3.95';
                document.getElementById('inputV17').value = '-4.10';
                document.getElementById('inputV4').value = '3.45';
                document.getElementById('inputV11').value = '3.10';
            } else if (val === 'genuine_sample') {
                document.getElementById('inputAmount').value = '42.80';
                document.getElementById('inputV14').value = '-0.12';
                document.getElementById('inputV12').value = '0.05';
                document.getElementById('inputV10').value = '0.02';
                document.getElementById('inputV17').value = '-0.08';
                document.getElementById('inputV4').value = '0.15';
                document.getElementById('inputV11').value = '-0.04';
            } else if (val === 'micro_fraud') {
                document.getElementById('inputAmount').value = '2.99';
                document.getElementById('inputV14').value = '-5.40';
                document.getElementById('inputV12').value = '-3.80';
                document.getElementById('inputV10').value = '-3.10';
                document.getElementById('inputV17').value = '-3.50';
                document.getElementById('inputV4').value = '2.90';
                document.getElementById('inputV11').value = '2.75';
            }
        });
    }

    // Handle Single Transaction Prediction Form Submission via AJAX
    const singleTxForm = document.getElementById('singleTxForm');
    if (singleTxForm) {
        singleTxForm.addEventListener('submit', function (e) {
            e.preventDefault();
            const submitBtn = document.getElementById('btnSubmitTxTest');
            const resultBox = document.getElementById('singleTxResult');
            
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i> Analyzing...';
            }

            const payload = {
                Amount: parseFloat(document.getElementById('inputAmount').value) || 0.0,
                V14: parseFloat(document.getElementById('inputV14').value) || 0.0,
                V12: parseFloat(document.getElementById('inputV12').value) || 0.0,
                V10: parseFloat(document.getElementById('inputV10').value) || 0.0,
                V17: parseFloat(document.getElementById('inputV17').value) || 0.0,
                V4: parseFloat(document.getElementById('inputV4').value) || 0.0,
                V11: parseFloat(document.getElementById('inputV11').value) || 0.0
            };

            fetch('/detection/predict-single', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            })
            .then(res => res.json())
            .then(data => {
                if (data.success) {
                    resultBox.classList.remove('d-none');
                    
                    const isFraud = data.prediction === 'Fraudulent';
                    const badgeClass = isFraud ? 'badge-prediction fraud' : 'badge-prediction genuine';
                    const riskClass = data.risk_level === 'High Risk' ? 'badge-risk high' : (data.risk_level === 'Medium Risk' ? 'badge-risk medium' : 'badge-risk low');

                    document.getElementById('resPredictionBadge').className = badgeClass;
                    document.getElementById('resPredictionBadge').innerHTML = isFraud ? '<i class="fas fa-exclamation-triangle"></i> Fraudulent' : '<i class="fas fa-check-circle"></i> Genuine';
                    
                    document.getElementById('resRiskBadge').className = riskClass;
                    document.getElementById('resRiskBadge').textContent = data.risk_level;

                    document.getElementById('resProbability').textContent = data.fraud_probability_pct + '%';
                    document.getElementById('resExplanation').textContent = data.explanation;
                    document.getElementById('resRecommendation').textContent = data.recommendation;
                } else {
                    alert('Prediction error: ' + (data.error || 'Unknown failure'));
                }
            })
            .catch(err => {
                console.error(err);
                alert('Network or server error executing prediction.');
            })
            .finally(() => {
                if (submitBtn) {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = '<i class="fas fa-microchip me-2"></i> Run AI Prediction';
                }
            });
        });
    }
});
