let mainChart = null;
let forecastChart = null;
let isStreaming = false;
let currentLanguage = 'en';

const translations = {
    en: {
        dashboard: "Dashboard", imageAnalysis: "Image Analysis", settings: "Settings",
        riskLevel: "Risk Level", waterArea: "Water Area", insights: "Real-Time Insights",
        downloadReport: "📥 Download Disaster Audit Report", initializing: "INITIALIZING",
        low: "LOW (Safe)", medium: "MEDIUM", high: "HIGH", extreme: "EXTREME",
        predatorAlert: "🐊 PREDATOR DETECTED!"
    },
    gu: {
        dashboard: "ડેશબોર્ડ", imageAnalysis: "છબી વિશ્લેષણ", settings: "સેટિંગ્સ",
        riskLevel: "જોખમ સ્તર", waterArea: "પાણીનો વિસ્તાર", insights: "રીઅલ-ટાઇમ આંતરદૃષ્ટિ",
        downloadReport: "📥 આપત્તિ ઓડિટ રિપોર્ટ ડાઉનલોડ કરો", initializing: "પ્રારંભ થઈ રહ્યું છે",
        low: "ઓછું (સુરક્ષિત)", medium: "મધ્યમ", high: "ઉચ્ચ", extreme: "ગંભીર",
        predatorAlert: "🐊 મગર જોવા મળ્યો!"
    }
};

document.addEventListener('DOMContentLoaded', () => {
    initTime();
    initChart();
    initUpload();
    const emailInput = document.getElementById('alert-email');
    if (emailInput) emailInput.addEventListener('change', updateSettings);
    setInterval(updateStatus, 1000);
});

function initTime() {
    const timeEl = document.getElementById('current-time');
    const update = () => { if (timeEl) timeEl.innerText = new Date().toLocaleString(); };
    update(); setInterval(update, 1000);
}

function initChart() {
    const mCtx = document.getElementById('mainChart');
    if (mCtx) {
        mainChart = new Chart(mCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: Array.from({length: 50}, (_, i) => i),
                datasets: [
                    { label: 'Risk', data: [], borderColor: '#ef4444', fill: true, tension: 0.4 },
                    { label: 'Conf', data: [], borderColor: '#38bdf8', fill: true, tension: 0.4 }
                ]
            },
            options: { responsive: true, maintainAspectRatio: false }
        });
    }
    const fCtx = document.getElementById('forecastChart');
    if (fCtx) {
        forecastChart = new Chart(fCtx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: ['Now', '+1h', '+3h', '+6h'],
                datasets: [{ data: [0, 0, 0, 0], backgroundColor: ['#38bdf8', '#38bdf8', '#f59e0b', '#ef4444'] }]
            },
            options: { responsive: true, maintainAspectRatio: false }
        });
    }
}

function updateStatus() {
    fetch('/status').then(res => res.json()).then(data => {
        const tele = data.telemetry;
        if (!tele || Object.keys(tele).length === 0) return;

        const set = (id, val) => { const el = document.getElementById(id); if (el) el.innerText = val; };
        set('risk-level', tele.risk_level || 'N/A');
        set('risk-score', Math.round(tele.risk_score || 0));
        set('water-percent', (tele.water_p || 0).toFixed(1) + '%');
        set('expansion-rate', ((tele.expansion_rate || 0) * 100).toFixed(1) + '%');
        set('submersion-score', ((tele.submersion || 0) * 100).toFixed(1) + '%');
        
        const prog = document.getElementById('risk-progress');
        if (prog) prog.style.width = (tele.risk_score || 0) + '%';

        const buzzer = document.getElementById('buzzer');
        const soundBtn = document.getElementById('enable-sound');
        if (buzzer && soundBtn && soundBtn.checked) {
            const danger = (tele.risk_level || '').includes('HIGH') || (tele.risk_level || '').includes('EXTREME');
            if (danger) { if (buzzer.paused) buzzer.play().catch(() => {}); }
            else { buzzer.pause(); buzzer.currentTime = 0; }
        }

        if (mainChart && data.risk_history) {
            mainChart.data.datasets[0].data = data.risk_history;
            if (data.conf_history) mainChart.data.datasets[1].data = data.conf_history.map(c => c * 100);
            mainChart.update('none');
        }
        if (forecastChart && tele.prediction?.forecast_sequence) {
            forecastChart.data.datasets[0].data = [tele.risk_score, ...tele.prediction.forecast_sequence];
            forecastChart.update('none');
        }
    });
}

function setLanguage(lang) {
    currentLanguage = lang;
    const t = translations[lang];
    const spans = document.querySelectorAll('.menu-item span');
    if (spans.length >= 3) { spans[0].innerText = t.dashboard; spans[1].innerText = t.imageAnalysis; spans[2].innerText = t.settings; }
    const riskH3 = document.querySelector('.risk-card h3');
    if (riskH3) riskH3.innerText = t.riskLevel;
    const repBtn = document.querySelector('.report-zone button');
    if (repBtn) repBtn.innerText = t.downloadReport;
    document.querySelectorAll('.lang-toggle button').forEach(btn => {
        btn.classList.toggle('active', btn.innerText.toLowerCase().includes(lang === 'gu' ? 'ગુજ' : 'en'));
    });
}

function generateReport() {
    fetch('/status').then(res => res.json()).then(data => {
        const tele = data.telemetry;
        const report = `RAINWISE DISASTER AUDIT\nLocation: Vadodara\nRisk: ${tele.risk_level}\nWater: ${tele.water_p}%`;
        const blob = new Blob([report], { type: 'text/plain' });
        const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
        a.download = `Report_${Date.now()}.txt`; a.click();
    });
}

function initUpload() {
    const zone = document.getElementById('upload-zone');
    if (zone) {
        zone.ondragover = (e) => { e.preventDefault(); zone.style.borderColor = '#38bdf8'; };
        zone.ondrop = (e) => { e.preventDefault(); const file = e.dataTransfer.files[0]; if (file) handleImageUpload(file); };
    }
}

function handleImageUpload(file) {
    if (!file) return;
    const formData = new FormData(); formData.append('file', file);
    const resSec = document.getElementById('upload-results');
    const resData = document.getElementById('result-data');
    if (resSec) resSec.classList.remove('hidden');
    if (resData) resData.innerText = 'Analyzing...';

    fetch('/predict', { method: 'POST', body: formData }).then(res => res.json()).then(data => {
        const resImg = document.getElementById('result-img');
        if (resImg) resImg.src = 'data:image/jpeg;base64,' + data.result_image;
        if (resData) {
            const tele = data.telemetry || {};
            resData.innerHTML = `
                <div class="analysis-result-box">
                    <p><strong>Risk:</strong> ${data.risk_level} (${Math.round(data.risk_score)}/100)</p>
                    <p><strong>Water Area:</strong> ${data.water_p.toFixed(1)}%</p>
                    <p><strong>Submersion:</strong> ${((tele.submersion || 0) * 100).toFixed(1)}%</p>
                    ${tele.croc_detected ? '<p style="color:#ef4444; font-weight:bold; animation: pulse 1s infinite;">🐊 PREDATOR DETECTED!</p>' : ''}
                </div>
            `;
        }
    });
}

function handleVideoUpload(file) {
    if (!file) return;
    const formData = new FormData(); formData.append('file', file);
    const resData = document.getElementById('result-data');
    if (resData) resData.innerText = 'Uploading Video...';

    fetch('/upload_video', { method: 'POST', body: formData }).then(res => res.json()).then(data => {
        if (data.success) { showSection('dashboard'); document.getElementById('video-feed').src = '/video_feed?source=' + encodeURIComponent(data.path); isStreaming = true; }
    });
}

function processYoutube() {
    const url = document.getElementById('yt-url').value;
    const status = document.getElementById('cloud-status');
    if (!url) return;
    if (status) status.innerText = 'Downloading YouTube Stream...';
    fetch('/process_youtube', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ url: url }) })
    .then(res => res.json()).then(data => {
        if (data.success) { showSection('dashboard'); document.getElementById('video-feed').src = '/video_feed?source=' + encodeURIComponent(data.path); isStreaming = true; }
    });
}

function analyzeImageUrl() {
    const url = document.getElementById('img-url').value;
    if (!url) return;
    const resSec = document.getElementById('upload-results');
    const resData = document.getElementById('result-data');
    if (resSec) resSec.classList.remove('hidden');
    if (resData) resData.innerText = 'Fetching Image...';

    fetch('/analyze_url', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ url: url }) })
    .then(res => res.json()).then(data => {
        const resImg = document.getElementById('result-img');
        if (resImg) resImg.src = 'data:image/jpeg;base64,' + data.result_image;
        if (resData) resData.innerHTML = `<p>Risk: ${data.risk_level}</p>${data.telemetry?.croc_detected ? '<p style="color:red">🐊 CROC DETECTED!</p>' : ''}`;
    });
}

function toggleStream() {
    const feed = document.getElementById('video-feed');
    if (!feed) return;
    if (!isStreaming) { feed.src = '/video_feed?source=0'; isStreaming = true; }
    else { feed.src = ''; isStreaming = false; }
}

function showSection(section, element) {
    document.querySelectorAll('.content-section').forEach(s => s.classList.add('hidden'));
    const target = document.getElementById(`${section}-section`);
    if (target) target.classList.remove('hidden');
    document.querySelectorAll('.menu-item').forEach(m => m.classList.remove('active'));
    if (element) element.classList.add('active');
}

function updateSettings() {
    const settings = {
        show_yolo: document.getElementById('show-yolo')?.checked,
        enable_sound: document.getElementById('enable-sound')?.checked,
        alert_email: document.getElementById('alert-email')?.value
    };
    fetch('/update_settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(settings) });
}
