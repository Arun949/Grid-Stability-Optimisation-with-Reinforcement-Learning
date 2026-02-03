import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.grid_env import SmartGridEnv

def train():
    # Create log dir
    log_dir = "results/logs/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("models/rl_models", exist_ok=True)

    # Instantiate the env
    # Note: We use the class directly to avoid registration issues if not installed as package
    env = SmartGridEnv(data_path='data/processed/cleaned_grid_data.csv')
    
    # Wrap in Monitor to track stats
    env = Monitor(env, log_dir)
    
    # Vectorize and Normalize
    # VecNormalize is important for PPO to normalize observations and rewards
    # We use DummyVecEnv for single process training
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # Initialize Agent
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=log_dir
    )

    print("Starting training...")
    # Train for 500k timesteps (adjustable)
    # Reduced to 100k for initial quick run, can be increased
    TIMESTEPS = 100000 
    model.learn(total_timesteps=TIMESTEPS)
    
    # Save Model
    model_path = "models/rl_models/ppo_smartgrid"
    model.save(model_path)
    
    # Save Normalization Stats (crucial for loading later)
    env.save("models/rl_models/vec_normalize.pkl")
    
    print(f"Training complete. Model saved to {model_path}")

if __name__ == "__main__":
    train()
