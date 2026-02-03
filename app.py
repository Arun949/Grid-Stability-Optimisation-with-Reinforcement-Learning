from flask import Flask, render_template, jsonify
from flask_cors import CORS
import sys
import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.grid_env import SmartGridEnv

app = Flask(__name__)
CORS(app)

# Load Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned_grid_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'rl_models', 'ppo_smartgrid.zip')
STATS_PATH = os.path.join(BASE_DIR, 'models', 'rl_models', 'vec_normalize.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/evaluate')
def evaluate():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model not found. Please train the agent first."}), 404

    try:
        # Instantiate Env
        env = SmartGridEnv(data_path=DATA_PATH)
        env = DummyVecEnv([lambda: env])
        
        # Load Normalization Stats
        env = VecNormalize.load(STATS_PATH, env)
        env.training = False
        env.norm_reward = False

        # Load Model
        model = PPO.load(MODEL_PATH, env=env)

        # Run Episode (approx 3 days for faster loading)
        # 4 steps/hour * 24 hours * 3 days = 288 steps
        steps = 288
        obs = env.reset()
        
        results = {
            "timestamps": [],
            "soc": [],
            "net_grid": [],
            "solar": [],
            "wind": [],
            "load": [],
            "actions": []
        }

        for i in range(steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            info_dict = info[0]
            # Get raw observation from the environment (un-normalized)
            # The observation order in grid_env.py: [load, solar, wind_total, soc, t_sin, t_cos]
            raw_obs = env.get_original_obs()[0]
            
            results["load"].append(float(raw_obs[0]))
            results["solar"].append(float(raw_obs[1]))
            results["wind"].append(float(raw_obs[2]))
            results["soc"].append(float(info_dict['soc']))
            results["net_grid"].append(float(info_dict['net_grid_exchange']))
            results["actions"].append(float(action[0][0]))
            
            # Simulated timestamp for frontend
            results["timestamps"].append(i)

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
