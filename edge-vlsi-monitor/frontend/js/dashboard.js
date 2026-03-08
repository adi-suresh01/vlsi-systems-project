// Edge VLSI AI Monitor — Dashboard Frontend
// WebSocket-driven real-time monitoring

let ws = null;
let performanceChart, powerChart, latencyChart;
let latencyData = [];
const MAX_LATENCY_POINTS = 50;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeCharts();
    connectWebSocket();
    checkStatus();
});

// ── Charts ──────────────────────────────────────────────────────────

function initializeCharts() {
    const chartDefaults = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#aaa' } } },
        scales: {
            x: { ticks: { color: '#888' }, grid: { color: 'rgba(100,100,255,0.08)' } },
            y: { ticks: { color: '#888' }, grid: { color: 'rgba(100,100,255,0.08)' }, beginAtZero: true }
        }
    };

    // Performance bar chart
    performanceChart = new Chart(document.getElementById('performanceChart'), {
        type: 'bar',
        data: {
            labels: ['Execution Time (ms)', 'HW Time (us)', 'Throughput (s/s)'],
            datasets: [{
                label: 'Simulation Metrics',
                data: [0, 0, 0],
                backgroundColor: ['rgba(0,210,255,0.6)', 'rgba(123,47,247,0.6)', 'rgba(0,255,136,0.6)'],
                borderColor: ['#00d2ff', '#7b2ff7', '#00ff88'],
                borderWidth: 2
            }]
        },
        options: { ...chartDefaults, plugins: { ...chartDefaults.plugins, title: { display: true, text: 'Simulation Performance', color: '#c0c0ff' } } }
    });

    // Power doughnut
    powerChart = new Chart(document.getElementById('powerChart'), {
        type: 'doughnut',
        data: {
            labels: ['Base Power', 'Dynamic Power', 'Leakage Power'],
            datasets: [{
                data: [33, 33, 34],
                backgroundColor: ['rgba(0,210,255,0.6)', 'rgba(123,47,247,0.6)', 'rgba(255,170,0,0.6)'],
                borderColor: ['#00d2ff', '#7b2ff7', '#ffaa00'],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: { display: true, text: 'Power Distribution', color: '#c0c0ff' },
                legend: { position: 'bottom', labels: { color: '#aaa' } }
            }
        }
    });

    // Latency timeline
    latencyChart = new Chart(document.getElementById('latencyChart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Avg Latency (us)',
                data: [],
                borderColor: '#00d2ff',
                backgroundColor: 'rgba(0,210,255,0.1)',
                fill: true,
                tension: 0.3,
                pointRadius: 2
            }, {
                label: 'P99 Latency (us)',
                data: [],
                borderColor: '#ff4444',
                backgroundColor: 'rgba(255,68,68,0.05)',
                fill: false,
                tension: 0.3,
                pointRadius: 2
            }]
        },
        options: { ...chartDefaults, plugins: { ...chartDefaults.plugins, title: { display: true, text: 'Agent Latency Over Time', color: '#c0c0ff' } } }
    });
}

// ── WebSocket ───────────────────────────────────────────────────────

function connectWebSocket() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${proto}//${location.host}/ws`);

    ws.onopen = () => {
        document.getElementById('ws-status').className = 'status-dot connected';
        document.getElementById('status-text').textContent = 'Connected to Edge VLSI Monitor';
        logMessage('WebSocket connected', 'info');
    };

    ws.onclose = () => {
        document.getElementById('ws-status').className = 'status-dot disconnected';
        document.getElementById('status-text').textContent = 'Disconnected — reconnecting...';
        logMessage('WebSocket disconnected, retrying in 3s...', 'warn');
        setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = () => {
        logMessage('WebSocket error', 'error');
    };

    ws.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            handleWsMessage(msg);
        } catch (e) {
            console.error('Failed to parse WS message:', e);
        }
    };
}

function handleWsMessage(msg) {
    switch (msg.type) {
        case 'MetricsUpdate':
            updateAgentMetrics(msg.data);
            break;
        case 'SimulationComplete':
            updateSimulation(msg.data);
            logMessage('Simulation completed', 'info');
            break;
        case 'AgentStateChange':
            logMessage(`Agent ${msg.data.name || msg.data.id}: ${msg.data.state}`, 'info');
            refreshAgentList();
            break;
        case 'connected':
            logMessage(msg.data.message, 'info');
            break;
        case 'Error':
            logMessage(msg.data.message, 'error');
            break;
    }
}

// ── Agent Management ────────────────────────────────────────────────

async function spawnAgent() {
    const name = `agent-${Date.now().toString(36)}`;
    try {
        const resp = await fetch('/api/agents', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
        });
        if (resp.ok) {
            const info = await resp.json();
            logMessage(`Spawned agent: ${info.name} (${info.id})`, 'info');
            refreshAgentList();
        }
    } catch (e) {
        logMessage(`Failed to spawn agent: ${e.message}`, 'error');
    }
}

async function refreshAgentList() {
    try {
        const resp = await fetch('/api/agents');
        const agents = await resp.json();
        renderAgentList(agents);
    } catch (e) {
        // Ignore — will refresh on next metrics update
    }
}

function renderAgentList(agents) {
    const container = document.getElementById('agent-list');
    if (agents.length === 0) {
        container.innerHTML = '<p class="placeholder">No agents running. Click "Spawn Agent" to start.</p>';
        return;
    }

    container.innerHTML = agents.map(a => `
        <div class="agent-card">
            <div class="agent-info">
                <span class="agent-name">${a.name}</span>
                <span class="agent-state ${a.state}">${a.state}</span>
                <br>
                <small style="color:#666">Inferences: ${a.inference_count} | MAC: ${a.total_mac_ops.toLocaleString()} | Power: ${(a.power_w * 1000).toFixed(3)} mW</small>
            </div>
            <div class="agent-actions">
                ${a.state === 'running' ? `
                    <button onclick="agentAction('${a.id}', 'pause')">Pause</button>
                    <button onclick="agentSimulate('${a.id}')">Simulate</button>
                ` : ''}
                ${a.state === 'paused' ? `
                    <button onclick="agentAction('${a.id}', 'resume')">Resume</button>
                ` : ''}
                ${a.state !== 'terminated' ? `
                    <button onclick="agentAction('${a.id}', 'terminate')">Kill</button>
                ` : ''}
            </div>
        </div>
    `).join('');
}

async function agentAction(id, action) {
    const method = action === 'terminate' ? 'DELETE' : 'POST';
    const url = action === 'terminate' ? `/api/agents/${id}` : `/api/agents/${id}/${action}`;
    try {
        await fetch(url, { method });
        refreshAgentList();
    } catch (e) {
        logMessage(`Agent action failed: ${e.message}`, 'error');
    }
}

async function agentSimulate(id) {
    try {
        const resp = await fetch(`/api/agents/${id}/simulate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ samples: 5 })
        });
        if (resp.ok) {
            const data = await resp.json();
            updateSimulation(data);
            logMessage(`Agent simulation complete: ${data.hardware_details.mac_ops} MAC ops`, 'info');
        }
    } catch (e) {
        logMessage(`Simulation failed: ${e.message}`, 'error');
    }
}

// ── Benchmark ───────────────────────────────────────────────────────

async function runBenchmark() {
    const btn = document.getElementById('run-benchmark');
    btn.disabled = true;
    btn.innerHTML = '<span class="loading"></span>Running...';
    logMessage('Starting benchmark (10 samples)...', 'info');

    try {
        const resp = await fetch('/api/benchmark');
        const data = await resp.json();
        updateSimulation(data);
        logMessage('Benchmark completed', 'info');
    } catch (e) {
        logMessage(`Benchmark failed: ${e.message}`, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = 'Run Benchmark';
        document.getElementById('last-updated').textContent = new Date().toLocaleString();
    }
}

async function checkStatus() {
    try {
        const resp = await fetch('/api/status');
        const data = await resp.json();
        logMessage(`Status: ${data.message} (${data.agent_count} agents)`, 'info');
        refreshAgentList();
    } catch (e) {
        logMessage('Server not reachable', 'error');
    }
}

// ── Update Functions ────────────────────────────────────────────────

function updateSimulation(data) {
    // Performance chart
    performanceChart.data.datasets[0].data = [
        data.performance.hardware_per_sample,
        data.performance.theoretical_hw_time * 1e6,
        data.performance.throughput
    ];
    performanceChart.update();

    // Power chart
    const pb = data.power_breakdown;
    powerChart.data.datasets[0].data = [
        pb.base_power_w * 1e6,
        pb.dynamic_power_w * 1e6,
        pb.leakage_power_w * 1e6
    ];
    powerChart.update();

    // Power metrics
    const pw = data.power;
    const fmt = (v, unit) => `${v.toFixed(4)} ${unit}`;
    document.getElementById('total-power').textContent =
        pw.total_watts < 0.001 ? fmt(pw.total_microwatts, 'uW') :
        pw.total_watts < 1 ? fmt(pw.total_milliwatts, 'mW') :
        fmt(pw.total_watts, 'W');
    document.getElementById('energy-per-inference').textContent = `${pw.energy_per_inference_uj.toFixed(4)} uJ`;
    document.getElementById('base-power').textContent = fmt(pb.base_power_w * 1000, 'mW');
    document.getElementById('dynamic-power').textContent = fmt(pb.dynamic_power_w * 1e6, 'uW');
    document.getElementById('leakage-power').textContent = fmt(pb.leakage_power_w * 1000, 'mW');

    // Hardware metrics
    const hw = data.hardware_details;
    document.getElementById('clock-cycles').textContent = hw.cycles.toLocaleString();
    document.getElementById('total-operations').textContent = hw.operations.toLocaleString();
    document.getElementById('mac-operations').textContent = hw.mac_ops.toLocaleString();
    document.getElementById('relu-operations').textContent = hw.relu_ops.toLocaleString();
    document.getElementById('throughput').textContent = `${data.performance.throughput.toFixed(1)} samples/s`;
    document.getElementById('clock-freq').textContent = `${hw.clock_frequency_mhz} MHz`;

    // Thermal
    if (data.thermal) {
        const t = data.thermal;
        const pct = Math.min(100, Math.max(0, ((t.junction_temp_c - 25) / 60) * 100));
        document.getElementById('temp-fill').style.width = `${pct}%`;
        document.getElementById('current-temp').textContent = `${t.junction_temp_c.toFixed(1)} C`;
        document.getElementById('junction-temp').textContent = `${t.junction_temp_c.toFixed(1)} C`;
        document.getElementById('thermal-headroom').textContent = `${t.headroom_c.toFixed(1)} C`;
        document.getElementById('throttle-status').textContent = t.should_throttle ? 'THROTTLING' : 'Normal';
        document.getElementById('throttle-status').style.color = t.should_throttle ? '#ff4444' : '#00ff88';
    }
}

function updateAgentMetrics(snapshot) {
    if (snapshot.agents && snapshot.agents.length > 0) {
        renderAgentList(snapshot.agents);

        // Update latency timeline
        const now = new Date().toLocaleTimeString();
        const avgLatency = snapshot.agents.reduce((s, a) => s + a.avg_latency_us, 0) / snapshot.agents.length;
        const maxP99 = Math.max(...snapshot.agents.map(a => a.p99_latency_us));

        latencyData.push({ time: now, avg: avgLatency, p99: maxP99 });
        if (latencyData.length > MAX_LATENCY_POINTS) latencyData.shift();

        latencyChart.data.labels = latencyData.map(d => d.time);
        latencyChart.data.datasets[0].data = latencyData.map(d => d.avg);
        latencyChart.data.datasets[1].data = latencyData.map(d => d.p99);
        latencyChart.update('none');
    }
}

// ── Logging ─────────────────────────────────────────────────────────

function logMessage(message, level = '') {
    const container = document.getElementById('log-container');
    const timestamp = new Date().toLocaleTimeString();
    const p = document.createElement('p');
    p.textContent = `[${timestamp}] ${message}`;
    if (level) p.className = `log-${level}`;
    container.appendChild(p);
    container.scrollTop = container.scrollHeight;

    // Keep log size manageable
    while (container.children.length > 200) {
        container.removeChild(container.firstChild);
    }
}
