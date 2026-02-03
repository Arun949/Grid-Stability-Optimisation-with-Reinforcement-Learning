# Grid Stability Optimisation with Reinforcement Learning

This project implements an advanced Reinforcement Learning (RL) agent designed to optimize energy storage and grid interaction in a power system with high renewable energy penetration. It features a custom Gymnasium environment, PPO-based training, and a modern web dashboard for real-time visualization.

## 🚀 Overview
As renewable energy sources like wind and solar become dominant, grid stability becomes increasingly complex due to their intermittent nature. This project demonstrates how an AI agent can manage a grid-scale battery system to:
- **Perform Peak Shaving**: Reduce demand on the grid during peak hours.
- **Support Renewable Integration**: Store excess clean energy and discharge it when production is low.
- **Enhance Grid Stability**: Minimize the squared deviation of net grid exchange to ensure a "flat" demand profile.

## 🛠️ Tech Stack
- **AI/RL**: `Stable-Baselines3` (PPO), `Gymnasium`, `NumPy`, `Pandas`
- **Backend**: `Flask`, `Flask-CORS`
- **Frontend**: `Vanilla CSS`, `Chart.js`, `Google Fonts (Outfit)`
- **Experiment Tracking**: `TensorBoard`

## 📂 Project Structure
```text
├── agents/             # RL Training and Evaluation scripts
├── data/               # Processed ENTSO-E grid data
├── environment/        # Custom Gymnasium environment and Battery model
├── models/             # Saved RL models and normalization statistics
├── static/             # Dashboard styles and JavaScript
├── templates/          # HTML templates for the dashboard
├── app.py              # Flask server and API entry point
└── requirements.txt    # Project dependencies
```

## ⚙️ Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd "Grid Stability Optimisation"
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Mac/Linux
   # .venv\Scripts\activate   # Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📈 Usage

### 1. Training the Agent
Run the training script to train the PPO agent on the historical grid data.
```bash
python agents/train_agent.py
```
*Note: Logs are saved to `results/logs/` and can be viewed via TensorBoard.*

### 2. Manual Evaluation
To run a 1-week evaluation and generate performance plots:
```bash
python agents/evaluate_agent.py
```
Plots will be saved to `results/evaluation_plot.png`.

### 3. Web Dashboard
Launch the interactive AI dashboard to visualize the agent's performance in real-time.
```bash
python app.py
```
Access the dashboard at `http://localhost:5001`.

## 🧠 Methodology

### The Environment (`SmartGridEnv`)
The agent receives an observation vector containing:
- Current Grid Load
- Solar Generation
- Wind Generation
- Battery State of Charge (SoC)
- Time of Day (sin/cos encoding)

### Reward Function
The agent is optimized via a reward function that penalizes:
1. **Grid Imbalance**: High penalties for large power draws or injections.
2. **Battery Wear**: Small penalties for every charge/discharge cycle to promote efficient usage.

---
Developed as a demonstration of Reinforcement Learning for Sustainable Infrastructure.
