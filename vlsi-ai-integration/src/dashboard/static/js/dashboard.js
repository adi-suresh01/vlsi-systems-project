// Global variables for charts
let performanceChart, speedupChart, powerChart;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    initializeCharts();
    checkSystemStatus();
});

// Initialize all charts
function initializeCharts() {
    // Performance Comparison Chart
    const perfCtx = document.getElementById('performanceChart').getContext('2d');
    performanceChart = new Chart(perfCtx, {
        type: 'bar',
        data: {
            labels: ['CPU', 'TFLite', 'Hardware'],
            datasets: [{
                label: 'Time per Sample (ms)',
                data: [0, 0, 0],
                backgroundColor: [
                    'rgba(255, 99, 132, 0.7)',
                    'rgba(54, 162, 235, 0.7)',
                    'rgba(75, 192, 192, 0.7)'
                ],
                borderColor: [
                    'rgba(255, 99, 132, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(75, 192, 192, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Time (milliseconds)'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Inference Time Comparison'
                }
            }
        }
    });

    // Speedup Chart
    const speedupCtx = document.getElementById('speedupChart').getContext('2d');
    speedupChart = new Chart(speedupCtx, {
        type: 'bar',
        data: {
            labels: ['CPU vs Hardware', 'TFLite vs Hardware'],
            datasets: [{
                label: 'Speedup (x times faster)',
                data: [0, 0],
                backgroundColor: [
                    'rgba(255, 206, 86, 0.7)',
                    'rgba(153, 102, 255, 0.7)'
                ],
                borderColor: [
                    'rgba(255, 206, 86, 1)',
                    'rgba(153, 102, 255, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Speedup Factor'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Hardware Acceleration Speedup'
                }
            }
        }
    });

    // Power Chart (Donut)
    const powerCtx = document.getElementById('powerChart').getContext('2d');
    powerChart = new Chart(powerCtx, {
        type: 'doughnut',
        data: {
            labels: ['Base Power', 'Dynamic Power'],
            datasets: [{
                data: [50, 50],
                backgroundColor: [
                    'rgba(255, 159, 64, 0.7)',
                    'rgba(255, 99, 132, 0.7)'
                ],
                borderColor: [
                    'rgba(255, 159, 64, 1)',
                    'rgba(255, 99, 132, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Power Distribution'
                },
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

// Check system status
async function checkSystemStatus() {
    try {
        logMessage('Checking system status...');
        const response = await fetch('/api/status');
        const data = await response.json();
        
        const statusText = document.getElementById('status-text');
        const runButton = document.getElementById('run-benchmark');
        
        if (data.ready) {
            statusText.textContent = '✅ System Ready - All files available';
            statusText.className = 'status-ready';
            runButton.disabled = false;
            logMessage('✅ System ready for benchmarking');
        } else {
            statusText.textContent = '❌ System Not Ready - Missing files';
            statusText.className = 'status-error';
            runButton.disabled = true;
            logMessage('❌ Missing required files. Please run training script first.');
            
            // Log missing files
            Object.entries(data.files).forEach(([name, info]) => {
                if (!info.exists) {
                    logMessage(`❌ Missing: ${name} (${info.path})`);
                }
            });
        }
    } catch (error) {
        console.error('Status check failed:', error);
        logMessage('❌ Failed to check system status');
    }
}

// Run benchmark
async function runBenchmark() {
    const runButton = document.getElementById('run-benchmark');
    const statusText = document.getElementById('status-text');
    
    // Disable button and show loading
    runButton.disabled = true;
    runButton.innerHTML = '<span class="loading"></span>Running Benchmark...';
    statusText.textContent = '🔄 Running benchmark...';
    statusText.className = 'status-loading';
    
    logMessage('🚀 Starting benchmark...');
    
    try {
        const response = await fetch('/api/benchmark');
        const data = await response.json();
        
        // Update UI with results
        updateDashboard(data);
        
        statusText.textContent = '✅ Benchmark completed successfully!';
        statusText.className = 'status-ready';
        logMessage('✅ Benchmark completed successfully!');
        
    } catch (error) {
        console.error('Benchmark failed:', error);
        statusText.textContent = '❌ Benchmark failed';
        statusText.className = 'status-error';
        logMessage(`❌ Benchmark failed: ${error.message}`);
    } finally {
        // Re-enable button
        runButton.disabled = false;
        runButton.innerHTML = 'Run Benchmark';
        document.getElementById('last-updated').textContent = new Date().toLocaleString();
    }
}

// Update dashboard with benchmark results
function updateDashboard(data) {
    logMessage('📊 Updating dashboard with results...');
    
    // Update performance chart
    performanceChart.data.datasets[0].data = [
        data.performance.cpu_per_sample,
        data.performance.tflite_per_sample,
        data.performance.hardware_per_sample
    ];
    performanceChart.update();
    
    // Update speedup chart
    speedupChart.data.datasets[0].data = [
        data.speedup.cpu_vs_hardware,
        data.speedup.tflite_vs_hardware
    ];
    speedupChart.update();
    
    // Update power metrics
    const power = data.power;
    let powerDisplay, powerUnit;
    
    if (power.total_watts < 0.001) {
        powerDisplay = power.total_microwatts.toFixed(1);
        powerUnit = 'µW';
    } else if (power.total_watts < 1) {
        powerDisplay = power.total_milliwatts.toFixed(3);
        powerUnit = 'mW';  
    } else {
        powerDisplay = power.total_watts.toFixed(6);
        powerUnit = 'W';
    }
    
    document.getElementById('total-power').textContent = `${powerDisplay} ${powerUnit}`;
    document.getElementById('energy-per-inference').textContent = `${power.energy_per_inference_uj.toFixed(3)} µJ`;
    
    // Update hardware metrics
    document.getElementById('clock-cycles').textContent = data.hardware_details.cycles.toLocaleString();
    document.getElementById('total-operations').textContent = data.hardware_details.operations.toLocaleString();
    document.getElementById('mac-operations').textContent = data.hardware_details.mac_ops.toLocaleString();
    document.getElementById('relu-operations').textContent = data.hardware_details.relu_ops.toLocaleString();
    document.getElementById('throughput').textContent = `${data.hardware_details.throughput.toFixed(2)} samples/sec`;
    
    // Update accuracy metrics
    document.getElementById('cpu-prediction').textContent = data.accuracy.cpu_sample_prediction;
    document.getElementById('tflite-prediction').textContent = data.accuracy.tflite_sample_prediction;
    document.getElementById('predictions-match').textContent = data.accuracy.predictions_match ? '✅ Yes' : '❌ No';
    
    // Log key results
    logMessage(`⏱️ CPU: ${data.performance.cpu_per_sample.toFixed(2)}ms, TFLite: ${data.performance.tflite_per_sample.toFixed(2)}ms, Hardware: ${data.performance.hardware_per_sample.toFixed(2)}ms`);
    logMessage(`⚡ Speedup - CPU vs HW: ${data.speedup.cpu_vs_hardware.toFixed(2)}x, TFLite vs HW: ${data.speedup.tflite_vs_hardware.toFixed(2)}x`);
    logMessage(`🔋 Power: ${powerDisplay} ${powerUnit}, Energy: ${power.energy_per_inference_uj.toFixed(3)} µJ/inference`);
}

// Log messages
function logMessage(message) {
    const logContainer = document.getElementById('log-container');
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = document.createElement('p');
    logEntry.textContent = `[${timestamp}] ${message}`;
    logContainer.appendChild(logEntry);
    logContainer.scrollTop = logContainer.scrollHeight;
}
