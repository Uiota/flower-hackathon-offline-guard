// Dashboard JavaScript for Federated Learning Web App

class FederatedLearningDashboard {
    constructor() {
        this.socket = null;
        this.charts = {};
        this.isConnected = false;
        this.trainingData = {
            rounds: [],
            serverLoss: [],
            serverAccuracy: [],
            clientMetrics: {}
        };

        this.init();
    }

    init() {
        this.setupSocketConnection();
        this.setupEventListeners();
        this.setupCharts();
        this.loadConfiguration();
    }

    setupSocketConnection() {
        // Connect to the WebSocket server
        this.socket = io();

        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.isConnected = true;
            this.updateConnectionStatus(true);
            this.socket.emit('request_status');
        });

        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.isConnected = false;
            this.updateConnectionStatus(false);
        });

        this.socket.on('status_update', (data) => {
            this.updateStatus(data);
        });

        this.socket.on('metrics_update', (data) => {
            this.updateMetrics(data);
        });

        this.socket.on('training_started', (data) => {
            this.onTrainingStarted(data);
        });

        this.socket.on('training_stopped', (data) => {
            this.onTrainingStopped(data);
        });

        this.socket.on('connect_error', (error) => {
            console.error('Connection error:', error);
            this.addLog('Connection error: ' + error.message, 'error');
        });
    }

    setupEventListeners() {
        // Start/Stop buttons
        document.getElementById('start-btn').addEventListener('click', () => {
            this.startTraining();
        });

        document.getElementById('stop-btn').addEventListener('click', () => {
            this.stopTraining();
        });

        // Configuration form
        document.getElementById('config-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.updateConfiguration();
        });

        // Clear logs button
        document.getElementById('clear-logs').addEventListener('click', () => {
            this.clearLogs();
        });

        // Request status every 30 seconds
        setInterval(() => {
            if (this.isConnected) {
                this.socket.emit('request_status');
                this.socket.emit('request_metrics');
            }
        }, 30000);
    }

    setupCharts() {
        // Loss Chart
        const lossCtx = document.getElementById('loss-chart').getContext('2d');
        this.charts.loss = new Chart(lossCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Server Loss',
                    data: [],
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    tension: 0.1,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Training Loss Over Rounds'
                    },
                    legend: {
                        display: true
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Round'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Loss'
                        },
                        beginAtZero: false
                    }
                }
            }
        });

        // Accuracy Chart
        const accuracyCtx = document.getElementById('accuracy-chart').getContext('2d');
        this.charts.accuracy = new Chart(accuracyCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Server Accuracy',
                    data: [],
                    borderColor: 'rgb(54, 162, 235)',
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    tension: 0.1,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Model Accuracy Over Rounds'
                    },
                    legend: {
                        display: true
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Round'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Accuracy'
                        },
                        beginAtZero: true,
                        max: 1.0
                    }
                }
            }
        });
    }

    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connection-status');
        if (connected) {
            statusElement.innerHTML = '<i class="fas fa-circle text-success"></i> Connected';
        } else {
            statusElement.innerHTML = '<i class="fas fa-circle text-danger"></i> Disconnected';
        }
    }

    updateStatus(data) {
        // Update training status
        const statusElement = document.getElementById('training-status');
        const indicatorElement = document.getElementById('training-indicator');
        const startBtn = document.getElementById('start-btn');
        const stopBtn = document.getElementById('stop-btn');

        if (data.is_running) {
            statusElement.textContent = 'Running';
            statusElement.className = 'badge bg-success me-2';
            indicatorElement.classList.remove('d-none');
            startBtn.disabled = true;
            stopBtn.disabled = false;
        } else {
            statusElement.textContent = 'Stopped';
            statusElement.className = 'badge bg-secondary me-2';
            indicatorElement.classList.add('d-none');
            startBtn.disabled = false;
            stopBtn.disabled = true;
        }

        // Update configuration display
        if (data.config) {
            this.displayConfiguration(data.config);
        }
    }

    updateMetrics(data) {
        // Process server metrics
        if (data.server_loss) {
            this.processServerMetrics(data.server_loss, 'loss');
        }

        if (data.server_eval_accuracy) {
            this.processServerMetrics(data.server_eval_accuracy, 'accuracy');
        }

        // Update client table
        this.updateClientTable(data);

        // Update summary metrics
        this.updateSummaryMetrics(data);
    }

    processServerMetrics(metrics, type) {
        const rounds = [];
        const values = [];

        metrics.forEach(metric => {
            rounds.push(metric.round);
            values.push(metric.value);
        });

        // Update chart
        if (this.charts[type]) {
            this.charts[type].data.labels = rounds;
            this.charts[type].data.datasets[0].data = values;
            this.charts[type].update('none');
        }

        // Update current value display
        if (values.length > 0) {
            const latestValue = values[values.length - 1];
            const latestRound = rounds[rounds.length - 1];

            if (type === 'loss') {
                document.getElementById('current-loss').textContent = latestValue.toFixed(4);
            }

            document.getElementById('current-round').textContent = latestRound;
        }
    }

    updateClientTable(data) {
        const tbody = document.querySelector('#clients-table tbody');
        tbody.innerHTML = '';

        const clientData = this.aggregateClientData(data);

        if (Object.keys(clientData).length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="6" class="text-center text-muted">
                        No clients connected
                    </td>
                </tr>
            `;
            document.getElementById('connected-clients').textContent = '0';
            return;
        }

        document.getElementById('connected-clients').textContent = Object.keys(clientData).length;

        Object.entries(clientData).forEach(([clientId, client]) => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${clientId}</td>
                <td>
                    <span class="badge ${client.status === 'active' ? 'bg-success' : 'bg-secondary'}">
                        ${client.status}
                    </span>
                </td>
                <td>${client.lastSeen ? new Date(client.lastSeen * 1000).toLocaleTimeString() : '-'}</td>
                <td>${client.samples || '-'}</td>
                <td>${client.loss ? client.loss.toFixed(4) : '-'}</td>
                <td>${client.accuracy ? (client.accuracy * 100).toFixed(2) + '%' : '-'}</td>
            `;
            tbody.appendChild(row);
        });
    }

    aggregateClientData(data) {
        const clients = {};

        // Process all metric types to find unique clients
        Object.entries(data).forEach(([metricName, metrics]) => {
            if (Array.isArray(metrics)) {
                metrics.forEach(metric => {
                    if (metric.client_id) {
                        if (!clients[metric.client_id]) {
                            clients[metric.client_id] = {
                                status: 'active',
                                lastSeen: metric.timestamp,
                                samples: null,
                                loss: null,
                                accuracy: null
                            };
                        }

                        // Update with latest timestamp
                        if (metric.timestamp > clients[metric.client_id].lastSeen) {
                            clients[metric.client_id].lastSeen = metric.timestamp;
                        }

                        // Store specific metrics
                        if (metricName === 'client_samples') {
                            clients[metric.client_id].samples = metric.value;
                        } else if (metricName === 'client_loss') {
                            clients[metric.client_id].loss = metric.value;
                        } else if (metricName === 'client_eval_accuracy') {
                            clients[metric.client_id].accuracy = metric.value;
                        }
                    }
                });
            }
        });

        // Mark clients as inactive if not seen recently (5 minutes)
        const now = Date.now() / 1000;
        Object.values(clients).forEach(client => {
            if (now - client.lastSeen > 300) {
                client.status = 'inactive';
            }
        });

        return clients;
    }

    updateSummaryMetrics(data) {
        // This is already handled by processServerMetrics and updateClientTable
        // Additional summary updates can be added here
    }

    startTraining() {
        if (!this.isConnected) {
            this.addLog('Cannot start training: not connected to server', 'error');
            return;
        }

        this.addLog('Starting federated training...', 'info');

        fetch('/api/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                this.addLog('Failed to start training: ' + data.error, 'error');
            } else {
                this.addLog('Training started successfully', 'success');
            }
        })
        .catch(error => {
            this.addLog('Error starting training: ' + error.message, 'error');
        });
    }

    stopTraining() {
        if (!this.isConnected) {
            this.addLog('Cannot stop training: not connected to server', 'error');
            return;
        }

        this.addLog('Stopping federated training...', 'info');

        fetch('/api/stop', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                this.addLog('Failed to stop training: ' + data.error, 'error');
            } else {
                this.addLog('Training stopped successfully', 'success');
            }
        })
        .catch(error => {
            this.addLog('Error stopping training: ' + error.message, 'error');
        });
    }

    updateConfiguration() {
        const form = document.getElementById('config-form');
        const formData = new FormData(form);
        const config = {};

        for (let [key, value] of formData.entries()) {
            // Convert numeric values
            if (key === 'num_rounds' || key === 'min_fit_clients') {
                config[key] = parseInt(value);
            } else {
                config[key] = value;
            }
        }

        this.addLog('Updating configuration...', 'info');

        fetch('/api/config', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                this.addLog('Failed to update configuration: ' + data.error, 'error');
            } else {
                this.addLog('Configuration updated successfully', 'success');
                this.displayConfiguration(data.config);
            }
        })
        .catch(error => {
            this.addLog('Error updating configuration: ' + error.message, 'error');
        });
    }

    loadConfiguration() {
        fetch('/api/config')
        .then(response => response.json())
        .then(data => {
            if (!data.error) {
                this.displayConfiguration(data);
            }
        })
        .catch(error => {
            console.error('Error loading configuration:', error);
        });
    }

    displayConfiguration(config) {
        // Update form fields with current configuration
        if (config.dataset) {
            document.getElementById('dataset-select').value = config.dataset;
        }
        if (config.model) {
            document.getElementById('model-select').value = config.model;
        }
        if (config.num_rounds) {
            document.getElementById('num-rounds').value = config.num_rounds;
        }
        if (config.min_fit_clients) {
            document.getElementById('min-clients').value = config.min_fit_clients;
        }
    }

    onTrainingStarted(data) {
        this.addLog('Federated learning training has started', 'success');
        this.clearCharts();
    }

    onTrainingStopped(data) {
        this.addLog('Federated learning training has stopped', 'info');
    }

    clearCharts() {
        // Clear chart data
        Object.values(this.charts).forEach(chart => {
            chart.data.labels = [];
            chart.data.datasets[0].data = [];
            chart.update('none');
        });

        // Reset display values
        document.getElementById('current-round').textContent = '0';
        document.getElementById('current-loss').textContent = '-';
    }

    addLog(message, type = 'info') {
        const logsContainer = document.getElementById('logs-container');
        const timestamp = new Date().toLocaleTimeString();

        const logEntry = document.createElement('div');
        logEntry.className = `log-entry log-${type}`;
        logEntry.innerHTML = `<strong>${timestamp}</strong> ${message}`;

        logsContainer.appendChild(logEntry);
        logsContainer.scrollTop = logsContainer.scrollHeight;

        // Limit log entries to prevent memory issues
        const logEntries = logsContainer.children;
        if (logEntries.length > 100) {
            logsContainer.removeChild(logEntries[0]);
        }
    }

    clearLogs() {
        const logsContainer = document.getElementById('logs-container');
        logsContainer.innerHTML = '<div class="text-muted">Logs cleared</div>';
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    new FederatedLearningDashboard();
});