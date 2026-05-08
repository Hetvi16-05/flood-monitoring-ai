let mainChart = null;
let forecastChart = null;
let isStreaming = false;

document.addEventListener('DOMContentLoaded', () => {
    initTime();
    initChart();
    initUpload();
    
    // Add event listener for email input change
    const emailInput = document.getElementById('alert-email');
    if (emailInput) emailInput.addEventListener('change', updateSettings);
    
    // Start periodic status updates
    setInterval(updateStatus, 1000);
});

function initTime() {
    const timeEl = document.getElementById('current-time');
    const updateTime = () => {
        timeEl.innerText = new Date().toLocaleString();
    };
    updateTime();
    setInterval(updateTime, 1000);
}

function initChart() {
    const ctx = document.getElementById('mainChart').getContext('2d');
    mainChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({length: 50}, (_, i) => i),
            datasets: [
                {
                    label: 'Risk Score',
                    data: [],
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    fill: true,
                    tension: 0.4
                },
                {
                    label: 'Confidence',
                    data: [],
                    borderColor: '#38bdf8',
                    backgroundColor: 'rgba(56, 189, 248, 0.1)',
                    fill: true,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                },
                x: {
                    grid: { display: false }
                }
            },
            plugins: {
                legend: { labels: { color: '#94a3b8' } }
            }
        }
    });

    const fctx = document.getElementById('forecastChart').getContext('2d');
    forecastChart = new Chart(fctx, {
        type: 'bar',
        data: {
            labels: ['Now', '+1h', '+3h', '+6h'],
            datasets: [{
                label: 'Predicted Risk',
                data: [0, 0, 0, 0],
                backgroundColor: ['#38bdf8', '#38bdf8', '#f59e0b', '#ef4444'],
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true, max: 100, grid: { color: 'rgba(255, 255, 255, 0.05)' } },
                x: { grid: { display: false } }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

function updateStatus() {
    fetch('/status')
        .then(res => res.json())
        .then(data => {
            const tele = data.telemetry;
            if (!tele || Object.keys(tele).length === 0) return;

            // Update Cards
            document.getElementById('risk-level').innerText = tele.risk_level || 'N/A';
            document.getElementById('risk-score').innerText = tele.risk_score || '0';
            document.getElementById('risk-progress').style.width = (tele.risk_score || 0) + '%';
            document.getElementById('water-percent').innerText = (tele.water_p || 0).toFixed(1) + '%';
            document.getElementById('avg-depth').innerText = (tele.avg_water_depth || 0).toFixed(2) + 'm';
            document.getElementById('expansion-rate').innerText = ((tele.expansion_rate || 0) * 100).toFixed(1) + '%';
            document.getElementById('submersion-score').innerText = ((tele.submersion || 0) * 100).toFixed(1) + '%';
            
            // LSTM Prediction Update
            const lstmEl = document.getElementById('lstm-prediction');
            if (lstmEl) {
                if (tele.prediction && tele.prediction.lstm_score !== null && tele.prediction.lstm_score !== undefined) {
                    lstmEl.innerText = tele.prediction.lstm_score.toFixed(1);
                } else {
                    lstmEl.innerText = 'COLLECTING...';
                }
            }

            // Early Warning Update
            const warningBadge = document.getElementById('early-warning-badge');
            if (warningBadge) {
                if (tele.early_warning) {
                    warningBadge.innerText = tele.early_warning;
                    warningBadge.classList.remove('hidden');
                } else {
                    warningBadge.classList.add('hidden');
                }
            }

            // Emergency Alert Update
            const emergencyBanner = document.getElementById('emergency-alert');
            const emergencyMsg = document.getElementById('emergency-msg');
            if (emergencyBanner && emergencyMsg) {
                if (tele.emergency_alert) {
                    emergencyMsg.innerText = tele.emergency_alert;
                    emergencyBanner.classList.remove('hidden');
                } else {
                    emergencyBanner.classList.add('hidden');
                }
            }

            // Update Forecast Chart
            if (forecastChart && tele.prediction && tele.prediction.forecast_sequence) {
                const sequence = tele.prediction.forecast_sequence;
                forecastChart.data.datasets[0].data = [tele.risk_score, ...sequence];
                forecastChart.update('none');
            }

            // Risk Color Coding
            const rl = tele.risk_level;
            const rEl = document.getElementById('risk-level');
            if (rl === 'DANGEROUS' || rl === 'HIGH') rEl.style.color = '#ef4444';
            else if (rl === 'MEDIUM') rEl.style.color = '#f59e0b';
            else rEl.style.color = '#10b981';

            // Update Insights
            const insightsList = document.getElementById('insights-list');
            insightsList.innerHTML = '';
            if (tele.temporal_insights && tele.temporal_insights.length > 0) {
                tele.temporal_insights.forEach(insight => {
                    const item = document.createElement('div');
                    item.className = 'insight-item';
                    item.innerText = insight;
                    insightsList.appendChild(item);
                });
            } else {
                insightsList.innerHTML = '<p class="no-data">Monitoring in progress...</p>';
            }

            // Update Charts
            if (data.risk_history) {
                mainChart.data.datasets[0].data = data.risk_history;
            }
            if (data.conf_history) {
                mainChart.data.datasets[1].data = data.conf_history.map(c => c * 100);
            }
            mainChart.update('none'); // Update without animation for performance
        });
}

function toggleStream() {
    const feed = document.getElementById('video-feed');
    if (!isStreaming) {
        feed.src = '/video_feed?source=0'; // Default webcam
        isStreaming = true;
    } else {
        feed.src = '';
        isStreaming = false;
    }
}

function showSection(section, element) {
    document.querySelectorAll('.content-section').forEach(s => s.classList.add('hidden'));
    const targetSection = document.getElementById(`${section}-section`);
    if (targetSection) targetSection.classList.remove('hidden');
    
    document.querySelectorAll('.menu-item').forEach(m => m.classList.remove('active'));
    
    if (element) {
        element.classList.add('active');
    } else {
        // Programmatic fallback
        const sidebarItems = document.querySelectorAll('.menu-item');
        if (section === 'dashboard') sidebarItems[0].classList.add('active');
        else if (section === 'upload') sidebarItems[1].classList.add('active');
        else if (section === 'settings') sidebarItems[2].classList.add('active');
    }
}

function initUpload() {
    const zone = document.getElementById('upload-zone');
    // Drop logic only for image zone for simplicity
    zone.ondragover = (e) => { e.preventDefault(); zone.style.borderColor = '#38bdf8'; };
    zone.ondragleave = () => { zone.style.borderColor = 'rgba(255, 255, 255, 0.1)'; };
    zone.ondrop = (e) => {
        e.preventDefault();
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) handleImageUpload(file);
    };
}

function handleImageUpload(file) {
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);

    const resultSection = document.getElementById('upload-results');
    resultSection.classList.remove('hidden');
    document.getElementById('result-data').innerText = 'Analyzing Image...';

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            document.getElementById('result-data').innerHTML = `
                <div class="error-msg">❌ ${data.error}</div>
            `;
            return;
        }
        document.getElementById('result-img').src = 'data:image/jpeg;base64,' + data.result_image;
        document.getElementById('result-data').innerHTML = `
            <div class="result-telemetry">
                <p><strong>Risk:</strong> ${data.risk_level || 'N/A'} (${(data.risk_score || 0).toFixed(1)}/100)</p>
                <p><strong>Water Area:</strong> ${(data.water_p || 0).toFixed(2)}%</p>
                <p><strong>Confidence:</strong> ${((data.telemetry?.hybrid_conf || 0) * 100).toFixed(1)}%</p>
            </div>
        `;
    })
    .catch(err => {
        document.getElementById('result-data').innerText = 'Error analyzing image: ' + err;
    });
}

function handleVideoUpload(file) {
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);

    const resultData = document.getElementById('result-data');
    document.getElementById('upload-results').classList.remove('hidden');
    resultData.innerText = '📤 Uploading Video...';

    fetch('/upload_video', {
        method: 'POST',
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            resultData.innerText = '✅ Video Uploaded! Starting analysis...';
            showSection('dashboard');
            const feed = document.getElementById('video-feed');
            feed.src = '/video_feed?source=' + encodeURIComponent(data.path);
            isStreaming = true;
        } else {
            resultData.innerText = '❌ Upload failed: ' + data.error;
        }
    })
    .catch(err => {
        resultData.innerText = '❌ Error: ' + err;
    });
}

function processYoutube() {
    const url = document.getElementById('yt-url').value;
    const status = document.getElementById('cloud-status');
    
    if (!url) {
        status.innerText = 'Please enter a valid URL';
        return;
    }

    status.innerText = '📥 Downloading & Processing YouTube Stream...';
    
    fetch('/process_youtube', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: url })
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            status.innerText = '✅ Success! Processing: ' + data.title;
            // switch to dashboard and show the feed
            showSection('dashboard');
            const feed = document.getElementById('video-feed');
            feed.src = '/video_feed?source=' + encodeURIComponent(data.path);
            isStreaming = true;
        } else {
            status.innerText = '❌ Error: ' + (data.error || 'Unknown error');
        }
    })
    .catch(err => {
        status.innerText = '❌ Request failed: ' + err;
    });
}

function updateSettings() {
    const showYolo = document.getElementById('show-yolo').checked;
    const showMask = document.getElementById('show-mask').checked;
    const explainAi = document.getElementById('explain-ai').checked;
    const enableSound = document.getElementById('enable-sound').checked;
    const enableEmail = document.getElementById('enable-email').checked;
    const alertEmail = document.getElementById('alert-email').value;
    
    // Toggle email input visibility
    const emailGroup = document.getElementById('email-input-group');
    if (enableEmail) emailGroup.classList.remove('hidden');
    else emailGroup.classList.add('hidden');

    const settings = {
        show_yolo: showYolo,
        show_mask: showMask,
        explain_ai: explainAi,
        enable_sound: enableSound,
        enable_email: enableEmail,
        alert_email: alertEmail
    };
    
    fetch('/update_settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings)
    })
    .then(res => res.json())
    .then(data => console.log('Settings synced:', data))
    .catch(err => console.error('Settings sync failed:', err));
}
