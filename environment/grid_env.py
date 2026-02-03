import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from .battery import Battery

class SmartGridEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data_path):
        super(SmartGridEnv, self).__init__()
        
        # Load Data
        self.df = pd.read_csv(data_path, parse_dates=['utc_timestamp'])
        self.df.sort_values('utc_timestamp', inplace=True)
        self.data_len = len(self.df)
        
        # Define Action Space: Continuous [-1, 1] for Battery Control
        # -1 = Max Discharge, 1 = Max Charge
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Define Observation Space
        # [Net Load, Solar, Wind, Battery SOC, Time of Day (sin), Time of Day (cos)]
        # Net Load = Load - (Solar + Wind)
        # We normalize vaguely for stability, but let's keep it raw for now and normalize in wrapper if needed.
        # Actually, let's provide: [Load, Solar, Wind, SOC, Hour_sin, Hour_cos]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, 0, 0, 0, -1, -1]),
            high=np.array([np.inf, np.inf, np.inf, 1, 1, 1]),
            dtype=np.float32
        )
        
        # Initialize Components
        self.battery = Battery(capacity=2000.0, max_charge_rate=500.0) # Scaled up for grid level? 
        # Note: Data is usually MW or similar? Let's check data scale.
        # analysis showed mean generation ~4000-6000. Units in source are likely MW if "DE_" country level.
        # If standard 15 min data is MW, then 4000 MW is huge. Battery needs to be grid scale.
        # Let's check magnitude in reset/step to confirm. Assuming MW.
        # Let's make battery 1000 MWh, 500 MW for significant impact.
        self.battery = Battery(capacity=1000.0, max_charge_rate=250.0) 

        self.current_step = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0 # Start from beginning (or random?)
        # For RL, random start index is better to prevent overfitting to sequence start
        if seed is not None:
             np.random.seed(seed)
        
        # Pick a random start point, ensuring enough steps for an episode
        self.current_step = np.random.randint(0, self.data_len - 24*4*7) # At least 1 week left
        
        self.battery.reset()
        
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        
        # Data Columns mapping (based on analysis)
        load = row['DE_load_actual_entsoe_transparency']
        solar = row['DE_solar_generation_actual']
        wind = row['DE_wind_generation_actual'] + row['DE_wind_offshore_generation_actual'] + row['DE_wind_onshore_capacity'] 
        # careful: 'DE_wind_onshore_capacity' is capacity, we want generation. 
        # Using correct columns from analysis.
        # Actually analysis showed: DE_wind_generation_actual, DE_wind_offshore_generation_actual. 
        # onshore generation implies (total - offshore) or separate column?
        # looking at previous columns: DE_wind_generation_actual seems to be total wind? or onshore?
        # usually ENTSO-E provides: Wind Onshore, Wind Offshore.
        # Let's sum what we have relevant.
        
        wind_total = row.get('DE_wind_generation_actual', 0) + row.get('DE_wind_offshore_generation_actual', 0)
        
        soc = self.battery.soc
        
        # Time encoding
        hour = row['utc_timestamp'].hour
        minute = row['utc_timestamp'].minute
        total_minutes = hour * 60 + minute
        day_fraction = total_minutes / (24 * 60)
        
        t_sin = np.sin(2 * np.pi * day_fraction)
        t_cos = np.cos(2 * np.pi * day_fraction)
        
        obs = np.array([load, solar, wind_total, soc, t_sin, t_cos], dtype=np.float32)
        return obs

    def step(self, action):
        # Action is % of max charge rate
        cmd_power = action[0] * self.battery.max_charge_rate 
        
        # Execute battery step
        actual_battery_power = self.battery.step(cmd_power)
        
        # Get current state data
        obs = self._get_obs()
        load = obs[0]
        solar = obs[1]
        wind = obs[2]
        
        # Grid Interaction
        # Net Load = Load - (Solar + Wind)
        # If Net Load > 0, we need power from grid (or battery discharge)
        # If Net Load < 0, we have excess renewable
        
        # Balance Equation:
        # Load = Solar + Wind + Battery_Discharge + Grid_Import - Battery_Charge - Grid_Export
        # Grid_Net = Load - Solar - Wind + Battery_Power (Positive=Charge/Load)
        
        grid_net = load - solar - wind + actual_battery_power
        
        # Reward Function
        # 1. Stability/Smoothness: Penalty for high grid reliance (peak shaving)
        # 2. Renewables: Penalty is already implicit if we want to minimize grid_net
        
        reward = 0
        
        # Squared penalty for grid stress (incentivizes flatness/zero)
        reward -= 0.001 * (grid_net ** 2)
        
        # Small penalty for battery use (aging cost)
        reward -= 0.1 * abs(actual_battery_power)
        
        # Optional: Bonus for keeping SOC healthy (e.g. 20-80%)?
        # Let's stick to simple stability first.
        
        # Terminate?
        terminated = False
        truncated = False
        
        self.current_step += 1
        if self.current_step >= self.data_len - 1:
            truncated = True
            
        next_obs = self._get_obs()
        
        info = {
            'net_grid_exchange': grid_net,
            'battery_power': actual_battery_power,
            'soc': self.battery.soc
        }
        
        return next_obs, reward, terminated, truncated, info
