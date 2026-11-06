/**
 * FraudGuard AI - Chart.js Visualizations
 * Dark theme fintech cybersecurity charts
 */

function initDashboardCharts(distributionData, timelineData) {
    Chart.defaults.color = '#9ca3af';
    Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";

    // 1. Transaction Distribution Doughnut Chart
    const distCanvas = document.getElementById('distributionChart');
    if (distCanvas && distributionData) {
        new Chart(distCanvas, {
            type: 'doughnut',
            data: {
                labels: distributionData.labels,
                datasets: [{
                    data: distributionData.counts,
                    backgroundColor: ['#10b981', '#ef4444'],
                    borderColor: '#131b2e',
                    borderWidth: 3,
                    hoverOffset: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#e5e7eb',
                            boxWidth: 14,
                            padding: 18,
                            font: { size: 12, weight: '500' }
                        }
                    },
                    tooltip: {
                        backgroundColor: '#1f293d',
                        titleColor: '#fff',
                        bodyColor: '#e5e7eb',
                        borderColor: '#374151',
                        borderWidth: 1,
                        padding: 10,
                        callbacks: {
                            label: function (ctx) {
                                const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
                                const val = ctx.parsed;
                                const pct = ((val / total) * 100).toFixed(2);
                                return ` ${ctx.label}: ${val.toLocaleString()} (${pct}%)`;
                            }
                        }
                    }
                },
                cutout: '72%'
            }
        });
    }

    // 2. Transaction Activity Timeline Chart
    const actCanvas = document.getElementById('activityChart');
    if (actCanvas && timelineData) {
        new Chart(actCanvas, {
            type: 'bar',
            data: {
                labels: timelineData.labels,
                datasets: [
                    {
                        label: 'Legitimate Transactions',
                        data: timelineData.genuine_series,
                        backgroundColor: 'rgba(16, 185, 129, 0.65)',
                        borderColor: '#10b981',
                        borderWidth: 1,
                        borderRadius: 4
                    },
                    {
                        label: 'Fraudulent Transactions',
                        data: timelineData.fraud_series,
                        backgroundColor: 'rgba(239, 68, 68, 0.85)',
                        borderColor: '#ef4444',
                        borderWidth: 1,
                        borderRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#9ca3af', font: { size: 11 } }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#9ca3af', font: { size: 11 } }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: '#e5e7eb',
                            boxWidth: 12,
                            padding: 14,
                            font: { size: 12 }
                        }
                    },
                    tooltip: {
                        backgroundColor: '#1f293d',
                        borderColor: '#374151',
                        borderWidth: 1,
                        padding: 10
                    }
                }
            }
        });
    }
}

function initAnalysisCharts(riskData, featureData) {
    Chart.defaults.color = '#9ca3af';
    Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";

    // Risk Level Doughnut Chart
    const riskCanvas = document.getElementById('riskChart');
    if (riskCanvas && riskData) {
        new Chart(riskCanvas, {
            type: 'doughnut',
            data: {
                labels: riskData.labels,
                datasets: [{
                    data: riskData.counts,
                    backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
                    borderColor: '#131b2e',
                    borderWidth: 3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: '#e5e7eb', boxWidth: 12, padding: 14 }
                    }
                },
                cutout: '70%'
            }
        });
    }

    // Top Influential Features Horizontal Bar Chart
    const featCanvas = document.getElementById('featureImportanceChart');
    if (featCanvas && featureData) {
        const labels = featureData.map(f => f.feature);
        const values = featureData.map(f => f.importance_percent);

        new Chart(featCanvas, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Relative Importance (%)',
                    data: values,
                    backgroundColor: 'rgba(59, 130, 246, 0.75)',
                    borderColor: '#3b82f6',
                    borderWidth: 1,
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#9ca3af', callback: v => v + '%' }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: '#e5e7eb', font: { weight: '600' } }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: '#1f293d',
                        borderColor: '#374151',
                        borderWidth: 1,
                        callbacks: {
                            label: ctx => ` Importance: ${ctx.parsed.x.toFixed(2)}%`
                        }
                    }
                }
            }
        });
    }
}
