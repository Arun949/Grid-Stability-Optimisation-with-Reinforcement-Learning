import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.grid_env import SmartGridEnv

def evaluate():
    model_path = "models/rl_models/ppo_smartgrid.zip"
    stats_path = "models/rl_models/vec_normalize.pkl"
    
    if not os.path.exists(model_path):
        print("Model not found. Train first.")
        return

    # Create Env
    env = SmartGridEnv(data_path='data/processed/cleaned_grid_data.csv')
    env = DummyVecEnv([lambda: env])
    
    # Load Normalization Stats
    # We must use the same stats as training to interpret observations correctly
    env = VecNormalize.load(stats_path, env)
    # Disable training mode for normalization (don't update stats during eval)
    env.training = False
    env.norm_reward = False

    # Load Model
    model = PPO.load(model_path, env=env)

    print("Evaluating model...")
    obs = env.reset()
    
    socs = []
    net_grid = []
    rewards = []
    actions = []

    # Run for 1 week (4 * 24 * 7 = 672 steps)
    steps = 672
    for i in range(steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # VecEnv returns array of infos
        info_dict = info[0]
        
        socs.append(info_dict['soc'])
        net_grid.append(info_dict['net_grid_exchange'])
        rewards.append(reward[0])
        actions.append(action[0][0])
        
    # Plotting
    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # SOC
    ax[0].plot(socs, color='blue', label='SOC')
    ax[0].set_ylabel('Battery SOC')
    ax[0].set_ylim(0, 1)
    ax[0].legend()
    ax[0].set_title('Trained Agent Evaluation (1 Week)')
    
    # Net Grid
    ax[1].plot(np.array(net_grid), color='red', label='Net Grid Exchange (MW)', alpha=0.7)
    ax[1].axhline(0, color='black', linestyle='--')
    ax[1].set_ylabel('Power (MW)')
    ax[1].legend()
    
    # Actions
    ax[2].plot(actions, color='green', label='Battery Action (Norm)')
    ax[2].set_ylabel('Action [-1, 1]')
    ax[2].set_xlabel('Steps (15 min)')
    ax[2].legend()
    
    plt.tight_layout()
    output_plot = 'results/evaluation_plot.png'
    plt.savefig(output_plot)
    print(f"Evaluation plot saved to {output_plot}")
    print(f"Mean Reward: {np.mean(rewards):.2f}")

if __name__ == "__main__":
    evaluate()
