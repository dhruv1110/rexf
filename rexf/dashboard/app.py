"""Simple working dashboard with charts."""

import json
import socket
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from ..backends.intelligent_storage import IntelligentStorage
from ..intelligence.insights import InsightsEngine


class DashboardHandler(BaseHTTPRequestHandler):
    """Simple HTTP request handler for the experiment dashboard."""

    def __init__(self, storage: IntelligentStorage, *args, **kwargs):
        self.storage = storage
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        query_params = parse_qs(parsed_path.query)

        if path == "/" or path == "/dashboard":
            self._serve_dashboard()
        elif path == "/api/experiments":
            self._serve_experiments_api(query_params)
        elif path == "/api/stats":
            self._serve_stats_api()
        elif path == "/api/chart_data":
            self._serve_chart_data_api(query_params)
        else:
            self._send_404()

    def _serve_dashboard(self):
        """Serve the main dashboard HTML."""
        html = self._generate_dashboard_html()
        self._send_response(200, html, "text/html")

    def _serve_experiments_api(self, query_params):
        """Serve experiments data as JSON."""
        try:
            experiments = self.storage.list_experiments(limit=50)
            
            experiments_data = []
            for exp in experiments:
                exp_data = {
                    "run_id": exp.run_id,
                    "experiment_name": exp.experiment_name,
                    "status": exp.status,
                    "start_time": exp.start_time.isoformat(),
                    "duration": exp.duration,
                    "parameters": exp.parameters,
                    "metrics": exp.metrics,
                }
                experiments_data.append(exp_data)

            self._send_json_response({"experiments": experiments_data})

        except Exception as e:
            self._send_json_response({"error": str(e)}, status=500)

    def _serve_stats_api(self):
        """Serve storage statistics as JSON."""
        try:
            stats = self.storage.get_storage_stats()
            self._send_json_response({"stats": stats})
        except Exception as e:
            self._send_json_response({"error": str(e)}, status=500)

    def _serve_chart_data_api(self, query_params):
        """Serve chart data for visualization."""
        try:
            experiments = self.storage.list_experiments(limit=100)
            
            # Prepare chart data
            chart_data = {
                "metrics_over_time": [],
                "parameter_space": [],
                "available_metrics": set(),
                "available_parameters": set()
            }
            
            for exp in experiments:
                # Collect available metrics and parameters
                for metric in (exp.metrics or {}):
                    chart_data["available_metrics"].add(metric)
                for param in (exp.parameters or {}):
                    chart_data["available_parameters"].add(param)
                
                # Metrics over time data
                for metric_name, metric_value in (exp.metrics or {}).items():
                    chart_data["metrics_over_time"].append({
                        "experiment": exp.experiment_name,
                        "run_id": exp.run_id[:8],
                        "time": exp.start_time.isoformat(),
                        "metric_name": metric_name,
                        "metric_value": metric_value,
                        "parameters": exp.parameters
                    })
                
                # Parameter space data
                if exp.parameters and exp.metrics:
                    for param_name, param_value in exp.parameters.items():
                        for metric_name, metric_value in exp.metrics.items():
                            chart_data["parameter_space"].append({
                                "experiment": exp.experiment_name,
                                "run_id": exp.run_id[:8],
                                "param_name": param_name,
                                "param_value": param_value,
                                "metric_name": metric_name,
                                "metric_value": metric_value
                            })
            
            # Convert sets to lists for JSON serialization
            chart_data["available_metrics"] = list(chart_data["available_metrics"])
            chart_data["available_parameters"] = list(chart_data["available_parameters"])
            
            self._send_json_response(chart_data)
            
        except Exception as e:
            self._send_json_response({"error": str(e)}, status=500)

    def _send_404(self):
        """Send 404 Not Found response."""
        self._send_response(404, "Not Found", "text/plain")

    def _send_response(self, status: int, content: str, content_type: str):
        """Send HTTP response."""
        self.send_response(status)
        self.send_header("Content-type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content.encode("utf-8"))

    def _send_json_response(self, data: dict, status: int = 200):
        """Send JSON response."""
        content = json.dumps(data, indent=2, default=str)
        self._send_response(status, content, "application/json")

    def _generate_dashboard_html(self) -> str:
        """Generate the simple dashboard HTML with working charts."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RexF Experiment Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f7fa;
            color: #333;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            margin: 0;
            font-size: 1.8rem;
        }
        
        .container {
            max-width: 1400px;
            margin: 2rem auto;
            padding: 0 1rem;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .stat-card {
            background: white;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.07);
            text-align: center;
            border: 1px solid #e2e8f0;
        }
        
        .stat-number {
            font-size: 2.5rem;
            font-weight: 700;
            color: #667eea;
            margin-bottom: 0.5rem;
        }
        
        .stat-label {
            color: #64748b;
            font-weight: 500;
            text-transform: uppercase;
            font-size: 0.85rem;
        }
        
        .charts-section {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-bottom: 2rem;
        }
        
        .chart-card {
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.07);
            border: 1px solid #e2e8f0;
            overflow: hidden;
        }
        
        .chart-header {
            padding: 1.5rem;
            background: #f8fafc;
            border-bottom: 1px solid #e2e8f0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .chart-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #1e293b;
        }
        
        .chart-controls {
            display: flex;
            gap: 0.5rem;
        }
        
        .chart-select {
            padding: 0.5rem;
            border: 1px solid #d1d5db;
            border-radius: 6px;
            background: white;
            font-size: 0.85rem;
        }
        
        .chart-container {
            padding: 1.5rem;
            height: 400px;
            position: relative;
        }
        
        .experiments-section {
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.07);
            border: 1px solid #e2e8f0;
            padding: 1.5rem;
        }
        
        .experiments-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }
        
        .experiments-table th,
        .experiments-table td {
            padding: 1rem 0.75rem;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .experiments-table th {
            background: #f8fafc;
            font-weight: 600;
            color: #374151;
        }
        
        .status-badge {
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        
        .status-completed {
            background: #d1fae5;
            color: #065f46;
        }
        
        .status-failed {
            background: #fee2e2;
            color: #991b1b;
        }
        
        .status-running {
            background: #fef3c7;
            color: #92400e;
        }
        
        .loading {
            text-align: center;
            padding: 3rem;
            color: #64748b;
            font-size: 1.1rem;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧪 RexF Experiment Dashboard</h1>
    </div>
    
    <div class="container">
        <!-- Statistics Cards -->
        <div class="stats-grid" id="stats-grid">
            <div class="stat-card">
                <div class="stat-number" id="total-experiments">-</div>
                <div class="stat-label">Total Experiments</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="completed-experiments">-</div>
                <div class="stat-label">Completed</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="success-rate">-</div>
                <div class="stat-label">Success Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="avg-duration">-</div>
                <div class="stat-label">Avg Duration (s)</div>
            </div>
        </div>
        
        <!-- Charts Section -->
        <div class="charts-section">
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-title">📊 Metrics Over Time</div>
                    <div class="chart-controls">
                        <select id="metric-select" class="chart-select" onchange="updateMetricsChart()">
                            <option value="">Select metric...</option>
                        </select>
                    </div>
                </div>
                <div class="chart-container">
                    <canvas id="metrics-chart"></canvas>
                </div>
            </div>
            
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-title">🎯 Parameter vs Metric</div>
                    <div class="chart-controls">
                        <select id="param-select" class="chart-select" onchange="updateParameterChart()">
                            <option value="">Select parameter...</option>
                        </select>
                        <select id="metric-select-2" class="chart-select" onchange="updateParameterChart()">
                            <option value="">Select metric...</option>
                        </select>
                    </div>
                </div>
                <div class="chart-container">
                    <canvas id="parameter-chart"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Experiments Table -->
        <div class="experiments-section">
            <h2>Recent Experiments</h2>
            <div id="experiments-content">
                <div class="loading">Loading experiments...</div>
            </div>
        </div>
    </div>
    
    <script>
        let chartData = {};
        let metricsChart = null;
        let parameterChart = null;
        
        async function loadData() {
            try {
                // Load stats
                const statsResponse = await fetch('/api/stats');
                const statsData = await statsResponse.json();
                updateStatsDisplay(statsData.stats);
                
                // Load experiments
                const experimentsResponse = await fetch('/api/experiments');
                const experimentsData = await experimentsResponse.json();
                updateExperimentsDisplay(experimentsData.experiments);
                
                // Load chart data
                const chartResponse = await fetch('/api/chart_data');
                chartData = await chartResponse.json();
                
                // Populate dropdowns
                populateDropdowns();
                
                // Auto-select first options
                if (chartData.available_metrics.length > 0) {
                    document.getElementById('metric-select').value = chartData.available_metrics[0];
                    document.getElementById('metric-select-2').value = chartData.available_metrics[0];
                    updateMetricsChart();
                }
                
                if (chartData.available_parameters.length > 0) {
                    document.getElementById('param-select').value = chartData.available_parameters[0];
                    updateParameterChart();
                }
                
            } catch (error) {
                console.error('Error loading data:', error);
            }
        }
        
        function updateStatsDisplay(stats) {
            document.getElementById('total-experiments').textContent = stats.total_experiments || 0;
            
            const byStatus = stats.by_status || {};
            const completed = byStatus.completed || 0;
            const total = stats.total_experiments || 0;
            
            document.getElementById('completed-experiments').textContent = completed;
            
            const successRate = total > 0 ? ((completed / total) * 100).toFixed(1) : 0;
            document.getElementById('success-rate').textContent = successRate + '%';
            
            document.getElementById('avg-duration').textContent = '0.00';  // Simplified for now
        }
        
        function updateExperimentsDisplay(experiments) {
            const container = document.getElementById('experiments-content');
            
            if (experiments.length === 0) {
                container.innerHTML = '<div class="loading">No experiments found</div>';
                return;
            }
            
            const table = document.createElement('table');
            table.className = 'experiments-table';
            
            table.innerHTML = `
                <thead>
                    <tr>
                        <th>Name</th>
                        <th>Status</th>
                        <th>Duration</th>
                        <th>Key Metrics</th>
                        <th>Start Time</th>
                    </tr>
                </thead>
                <tbody>
                    ${experiments.slice(0, 10).map(exp => `
                        <tr>
                            <td>${exp.experiment_name}</td>
                            <td>
                                <span class="status-badge status-${exp.status}">
                                    ${exp.status}
                                </span>
                            </td>
                            <td>${(exp.duration || 0).toFixed(2)}s</td>
                            <td>${Object.keys(exp.metrics || {}).slice(0, 2).join(', ')}</td>
                            <td>${new Date(exp.start_time).toLocaleString()}</td>
                        </tr>
                    `).join('')}
                </tbody>
            `;
            
            container.innerHTML = '';
            container.appendChild(table);
        }
        
        function populateDropdowns() {
            // Metrics dropdowns
            const metricSelect = document.getElementById('metric-select');
            const metricSelect2 = document.getElementById('metric-select-2');
            
            const metricOptions = chartData.available_metrics.map(metric => 
                `<option value="${metric}">${metric}</option>`
            ).join('');
            
            metricSelect.innerHTML = '<option value="">Select metric...</option>' + metricOptions;
            metricSelect2.innerHTML = '<option value="">Select metric...</option>' + metricOptions;
            
            // Parameters dropdown
            const paramSelect = document.getElementById('param-select');
            const paramOptions = chartData.available_parameters.map(param => 
                `<option value="${param}">${param}</option>`
            ).join('');
            
            paramSelect.innerHTML = '<option value="">Select parameter...</option>' + paramOptions;
        }
        
        function updateMetricsChart() {
            const selectedMetric = document.getElementById('metric-select').value;
            
            if (!selectedMetric || !chartData.metrics_over_time) {
                console.log('No metric selected or no data available');
                return;
            }
            
            console.log('Updating metrics chart for:', selectedMetric);
            
            const ctx = document.getElementById('metrics-chart').getContext('2d');
            
            if (metricsChart) {
                metricsChart.destroy();
            }
            
            // Filter data for selected metric
            const data = chartData.metrics_over_time
                .filter(item => item.metric_name === selectedMetric)
                .map((item, index) => ({
                    x: index,
                    y: item.metric_value,
                    label: item.experiment + ' (' + item.run_id + ')'
                }));
            
            metricsChart = new Chart(ctx, {
                type: 'scatter',
                data: {
                    datasets: [{
                        label: selectedMetric,
                        data: data,
                        backgroundColor: '#667eea',
                        borderColor: '#667eea',
                        pointRadius: 6,
                        pointHoverRadius: 8
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: selectedMetric + ' Over Time'
                        },
                        legend: {
                            display: false
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const point = context.parsed || context.raw;
                                    return context.dataset.label + ': ' + point.y;
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            type: 'linear',
                            title: {
                                display: true,
                                text: 'Experiment Index'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: selectedMetric
                            }
                        }
                    }
                }
            });
        }
        
        function updateParameterChart() {
            const selectedParam = document.getElementById('param-select').value;
            const selectedMetric = document.getElementById('metric-select-2').value;
            
            if (!selectedParam || !selectedMetric || !chartData.parameter_space) {
                console.log('Missing parameter or metric selection, or no data available');
                return;
            }
            
            console.log('Updating parameter chart for:', selectedParam, 'vs', selectedMetric);
            
            const ctx = document.getElementById('parameter-chart').getContext('2d');
            
            if (parameterChart) {
                parameterChart.destroy();
            }
            
            // Filter data for selected parameter and metric
            const data = chartData.parameter_space
                .filter(item => item.param_name === selectedParam && item.metric_name === selectedMetric)
                .map(item => ({
                    x: item.param_value,
                    y: item.metric_value,
                    label: item.experiment + ' (' + item.run_id + ')'
                }));
            
            parameterChart = new Chart(ctx, {
                type: 'scatter',
                data: {
                    datasets: [{
                        label: `${selectedMetric} vs ${selectedParam}`,
                        data: data,
                        backgroundColor: '#f093fb',
                        borderColor: '#f093fb',
                        pointRadius: 6,
                        pointHoverRadius: 8
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: selectedMetric + ' vs ' + selectedParam
                        },
                        legend: {
                            display: false
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const point = context.parsed || context.raw;
                                    return context.dataset.label + ': (' + point.x + ', ' + point.y + ')';
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: selectedParam
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: selectedMetric
                            }
                        }
                    }
                }
            });
        }
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            loadData();
            
            // Auto-refresh every 30 seconds
            setInterval(loadData, 30000);
        });
    </script>
</body>
</html>
        """.strip()

    def log_message(self, format, *args):
        """Override to suppress log messages."""
        return


def run_dashboard(
    storage_path: str = "experiments.db",
    host: str = "localhost",
    port: int = 8080,
    open_browser: bool = True,
) -> None:
    """Run the simple experiment dashboard web server."""
    
    # Find an available port
    actual_port = port
    while True:
        try:
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.bind((host, actual_port))
            test_socket.close()
            break
        except OSError:
            actual_port += 1
            if actual_port > port + 50:
                raise RuntimeError(f"No available ports found starting from {port}")

    # Create server
    storage = IntelligentStorage(storage_path)
    
    def handler_factory(*args, **kwargs):
        return DashboardHandler(storage, *args, **kwargs)
    
    server = HTTPServer((host, actual_port), handler_factory)

    url = f"http://{host}:{actual_port}"
    print(f"🚀 Starting Simple RexF Dashboard at {url}")
    print("📊 Features:")
    print("  • Working interactive charts")
    print("  • Metrics over time visualization")
    print("  • Parameter vs metric analysis")
    print("  • Real-time experiment data")
    print()
    print("Press Ctrl+C to stop the server")

    # Open browser
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down dashboard...")
        server.shutdown()


if __name__ == "__main__":
    run_dashboard()
