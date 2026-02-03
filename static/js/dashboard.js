document.addEventListener('DOMContentLoaded', () => {
    const runBtn = document.getElementById('run-eval');
    const loader = document.getElementById('loader');

    let charts = {};

    const initCharts = () => {
        // Main Flow Chart
        const flowCtx = document.getElementById('flowChart').getContext('2d');
        charts.flow = new Chart(flowCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'Demand', data: [], borderColor: '#ff453a', borderWidth: 2, fill: false, tension: 0.3 },
                    { label: 'Solar', data: [], borderColor: '#ffb340', borderWidth: 2, fill: true, backgroundColor: 'rgba(255, 179, 64, 0.1)', tension: 0.3 },
                    { label: 'Wind', data: [], borderColor: '#32d74b', borderWidth: 2, fill: true, backgroundColor: 'rgba(50, 215, 75, 0.1)', tension: 0.3 },
                    { label: 'Net Grid', data: [], borderColor: '#5e5ce6', borderWidth: 3, borderDash: [5, 5], tension: 0.3 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { ticks: { color: '#8e8e93' }, grid: { display: false } },
                    y: { ticks: { color: '#8e8e93' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                },
                plugins: { legend: { labels: { color: '#fff', font: { family: 'Outfit' } } } }
            }
        });

        // SOC Chart
        const socCtx = document.getElementById('socChart').getContext('2d');
        charts.soc = new Chart(socCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{ label: 'Battery SOC', data: [], borderColor: '#0071e3', backgroundColor: 'rgba(0, 113, 227, 0.2)', fill: true, tension: 0.4 }]
            },
            options: {
                responsive: true,
                scales: { y: { min: 0, max: 1 } },
                plugins: { legend: { display: false } }
            }
        });

        // Action Chart
        const actionCtx = document.getElementById('actionChart').getContext('2d');
        charts.action = new Chart(actionCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{ label: 'Action (-1 Discard, 1 Charge)', data: [], backgroundColor: 'rgba(94, 92, 230, 0.6)' }]
            },
            options: {
                responsive: true,
                scales: { y: { min: -1, max: 1 } },
                plugins: { legend: { display: false } }
            }
        });
    };

    const updateDashboard = (data) => {
        // Update Stats
        document.getElementById('peak-load').textContent = `${Math.max(...data.load).toFixed(0)} MW`;

        const totalRenewable = data.solar.reduce((a, b) => a + b, 0) + data.wind.reduce((a, b) => a + b, 0);
        const totalLoad = data.load.reduce((a, b) => a + b, 0);
        document.getElementById('renewable-purity').textContent = `${((totalRenewable / totalLoad) * 100).toFixed(1)}%`;

        const meanImbalance = data.net_grid.map(v => Math.abs(v)).reduce((a, b) => a + b, 0) / data.net_grid.length;
        document.getElementById('grid-imbalance').textContent = `${meanImbalance.toFixed(1)} MW`;

        // Count cycles (simple heuristic: sign changes in action)
        let cycles = 0;
        for (let i = 1; i < data.actions.length; i++) {
            if (data.actions[i] * data.actions[i - 1] < 0) cycles++;
        }
        document.getElementById('battery-cycles').textContent = Math.floor(cycles / 2);

        // Update Charts
        charts.flow.data.labels = data.timestamps;
        charts.flow.data.datasets[0].data = data.load;
        charts.flow.data.datasets[1].data = data.solar;
        charts.flow.data.datasets[2].data = data.wind;
        charts.flow.data.datasets[3].data = data.net_grid;
        charts.flow.update();

        charts.soc.data.labels = data.timestamps;
        charts.soc.data.datasets[0].data = data.soc;
        charts.soc.update();

        charts.action.data.labels = data.timestamps;
        charts.action.data.datasets[0].data = data.actions;
        charts.action.update();
    };

    const runEvaluation = async () => {
        loader.classList.remove('hidden');
        try {
            const response = await fetch('/api/evaluate');
            const data = await response.json();
            if (data.error) {
                alert(data.error);
            } else {
                updateDashboard(data);
            }
        } catch (err) {
            console.error(err);
            alert("Failed to connect to simulation server.");
        } finally {
            loader.classList.add('hidden');
        }
    };

    initCharts();
    runBtn.addEventListener('click', runEvaluation);

    // Run initial eval
    runEvaluation();
});
