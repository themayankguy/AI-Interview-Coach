// scripts.js

// Global state
let currentSession = {
    duration: 10,
    questions: 5
};

// --- Duration Selection ---
function selectDuration(mins, questions) {
    currentSession.duration = mins;
    currentSession.questions = questions;

    // Update UI
    document.querySelectorAll('.duration-card').forEach(card => {
        card.classList.remove('selected');
    });
    event.currentTarget.classList.add('selected');
}

// --- Start Interview ---
async function startInterview() {
    window.location.href = 'interview.html';
}

// --- Polling Logic ---
let lastResultTimestamp = null;

async function startPollingForResults() {
    // Get current result state first to know when it changes
    try {
        const initial = await fetch('../../results.json?t=' + Date.now());
        if (initial.ok) {
            const data = await initial.json();
            // We use a simple hash of the data or just assume any change means new result
            lastResultTimestamp = JSON.stringify(data);
        }
    } catch (e) { }

    setInterval(async () => {
        try {
            const response = await fetch('../../results.json?t=' + Date.now());
            if (response.ok) {
                const data = await response.json();
                const currentDataStr = JSON.stringify(data);

                if (lastResultTimestamp !== null && currentDataStr !== lastResultTimestamp) {
                    // New result detected!
                    document.getElementById('sync-status').innerText = "Result Detected! Analyzing...";
                    setTimeout(() => {
                        window.location.href = 'dashboard.html?view=analytics';
                    }, 1000);
                }
            }
        } catch (error) {
            console.error('Polling error:', error);
        }
    }, 2000); // Check every 2 seconds
}

function resetToSetup() {
    document.getElementById('setup-section').style.display = 'block';
    document.getElementById('setup-section').style.opacity = '1';
    document.getElementById('analytics-section').style.display = 'none';
}

// --- Load Data ---
async function loadLatestResults() {
    try {
        // Fetch results.json from ../../results.json
        const response = await fetch('../../results.json?t=' + Date.now());
        if (!response.ok) throw new Error('Could not load results');

        const data = await response.json();
        updateDashboard(data);
        saveToHistory(data);
    } catch (error) {
        console.error('Error loading results:', error);
        // show fallback/empty state if needed
    }
}

function updateDashboard(data) {
    const confidence = Math.round(data.confidence || 0);
    const engagement = Math.round((data.avg_engagement || 0) * 100);
    const wpm = Math.round(data.wpm || 0);
    const fillers = data.fillers || 0;
    const stability = Math.round(data.stability || 75); // Fallback if missing

    // Update Scores
    document.getElementById('overall-confidence').innerText = confidence + '%';
    document.getElementById('confidence-fill').style.width = confidence + '%';
    document.getElementById('engagement-score').innerText = engagement + '%';
    document.getElementById('wpm-score').innerText = wpm;
    document.getElementById('fillers-count').innerText = fillers;
    document.getElementById('stability-score').innerText = stability + '%';

    // Update Placement Prediction
    const placementText = document.getElementById('placement-text');
    const placementFill = document.getElementById('placement-fill');

    if (confidence > 80) {
        placementText.innerText = "High placement probability";
        placementText.style.color = "var(--success)";
    } else if (confidence >= 60) {
        placementText.innerText = "Good chances";
        placementText.style.color = "var(--accent-glow)";
    } else if (confidence >= 40) {
        placementText.innerText = "Moderate probability";
        placementText.style.color = "var(--warning)";
    } else {
        placementText.innerText = "Needs improvement";
        placementText.style.color = "var(--danger)";
    }
    placementFill.style.width = confidence + '%';

    // Update AI Feedback
    const feedbackList = document.getElementById('feedback-list');
    feedbackList.innerHTML = '';

    const feedback = [];
    if (wpm < 130) feedback.push('• Speaking pace is slow - try to project more energy.');
    if (engagement > 60) feedback.push('• Good engagement and facial expression.');
    if (fillers > 2) feedback.push('• Reduce filler words for a more polished delivery.');
    if (stability > 70) feedback.push('• Strong vocal stability maintained throughout.');
    else feedback.push('• Focus on maintaining a consistent vocal tone.');

    feedback.forEach(item => {
        const li = document.createElement('li');
        li.innerText = item;
        feedbackList.appendChild(li);
    });

    // Initialize Charts
    initCharts(data);
}

// --- Charts ---
let charts = {};

function initCharts(data) {
    const ctxRadar = document.getElementById('radarChart')?.getContext('2d');
    const ctxBar = document.getElementById('barChart')?.getContext('2d');
    const ctxLine = document.getElementById('lineChart')?.getContext('2d');

    if (charts.radar) charts.radar.destroy();
    if (charts.bar) charts.bar.destroy();
    if (charts.line) charts.line.destroy();

    const confidence = Math.round(data.confidence || 0);
    const engagement = Math.round((data.avg_engagement || 0) * 100);
    const wpm = Math.round(data.wpm || 0);
    const fillers = data.fillers || 0;
    const stability = Math.round(data.stability || 75);

    if (ctxRadar) {
        charts.radar = new Chart(ctxRadar, {
            type: 'radar',
            data: {
                labels: ['Engagement', 'Speaking Pace', 'Voice Stability', 'Communication', 'Confidence'],
                datasets: [{
                    label: 'Performance',
                    data: [engagement, Math.min(100, (wpm / 150) * 100), stability, 80, confidence],
                    backgroundColor: 'rgba(94, 92, 230, 0.2)',
                    borderColor: '#5e5ce6',
                    pointBackgroundColor: '#5e5ce6',
                }]
            },
            options: {
                responsive: true,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        angleLines: { color: 'rgba(255,255,255,0.1)' }
                    }
                }
            }
        });
    }

    if (ctxBar) {
        charts.bar = new Chart(ctxBar, {
            type: 'bar',
            data: {
                labels: ['WPM', 'Engagement', 'Stability'],
                datasets: [{
                    label: 'Score',
                    data: [wpm, engagement, stability],
                    backgroundColor: ['#5e5ce6', '#34c759', '#ffcc00']
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: 'rgba(255,255,255,0.05)' }
                    }
                }
            }
        });
    }

    if (ctxLine) {
        // Generate simulated line data based on the average engagement
        const trendData = Array.from({ length: 10 }, () => Math.max(0, Math.min(100, engagement + (Math.random() * 20 - 10))));
        charts.line = new Chart(ctxLine, {
            type: 'line',
            data: {
                labels: ['1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m', '10m'],
                datasets: [{
                    label: 'Engagement Trend',
                    data: trendData,
                    borderColor: '#5e5ce6',
                    tension: 0.4,
                    fill: true,
                    backgroundColor: 'rgba(94, 92, 230, 0.1)'
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: 'rgba(255,255,255,0.05)' }
                    }
                }
            }
        });
    }
}

// --- History Management ---
function saveToHistory(data) {
    let history = JSON.parse(localStorage.getItem('interview_history') || '[]');

    const newEntry = {
        id: Date.now(),
        date: new Date().toLocaleDateString(),
        confidence: Math.round(data.confidence || 0),
        engagement: Math.round((data.avg_engagement || 0) * 100),
        wpm: Math.round(data.wpm || 0),
        fillers: data.fillers || 0,
        stability: Math.round(data.stability || 75)
    };

    // Check if this result is already the latest to prevent duplicates on refresh
    if (history.length > 0) {
        const last = history[0];
        if (last.confidence === newEntry.confidence && last.wpm === newEntry.wpm) return;
    }

    history.unshift(newEntry);
    localStorage.setItem('interview_history', JSON.stringify(history));
}

function loadHistory() {
    const list = document.getElementById('history-list');
    const noHistory = document.getElementById('no-history');
    if (!list) return;

    const history = JSON.parse(localStorage.getItem('interview_history') || '[]');

    if (history.length === 0) {
        noHistory.style.display = 'block';
        return;
    }

    list.innerHTML = '';
    history.forEach(session => {
        const card = document.createElement('div');
        card.className = 'history-card';
        card.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <span style="color: var(--text-secondary); font-size: 0.8rem;">${session.date}</span>
                <span style="color: var(--accent-glow); font-weight: 700;">${session.confidence}%</span>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 0.9rem;">
                <div><span style="color: var(--text-secondary);">Engagement:</span> ${session.engagement}%</div>
                <div><span style="color: var(--text-secondary);">WPM:</span> ${session.wpm}</div>
                <div><span style="color: var(--text-secondary);">Fillers:</span> ${session.fillers}</div>
                <div><span style="color: var(--text-secondary);">Stability:</span> ${session.stability}%</div>
            </div>
        `;
        list.appendChild(card);
    });
}

// Initializations
window.onload = () => {
    if (window.location.pathname.includes('dashboard.html')) {
        const urlParams = new URLsearchParams(window.location.search);
        if (urlParams.get('view') === 'analytics') {
            document.getElementById('setup-section').style.display = 'none';
            document.getElementById('analytics-section').style.display = 'block';
            loadLatestResults();
        }
    }

    if (window.location.pathname.includes('history.html')) {
        loadHistory();
    }
};
