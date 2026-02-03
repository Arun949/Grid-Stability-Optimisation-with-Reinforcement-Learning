import numpy as np

class Battery:
    def __init__(self, capacity=100.0, max_charge_rate=50.0, efficiency=0.95, initial_soc=0.5):
        """
        Initialize battery parameters.
        
        Args:
            capacity (float): Total energy capacity in kWh.
            max_charge_rate (float): Max charge/discharge power in kW.
            efficiency (float): Round-trip efficiency (0 to 1).
            initial_soc (float): Initial state of charge (0 to 1).
        """
        self.capacity = capacity
        self.max_charge_rate = max_charge_rate
        self.efficiency = efficiency
        
        # State
        self.soc = initial_soc # State of Charge (0.0 to 1.0)
        self.current_energy = self.soc * self.capacity

    def step(self, power):
        """
        Execute a time step with requested power.
        Positive power = Charge
        Negative power = Discharge
        
        Args:
            power (float): Requested power in kW.
            
        Returns:
            actual_power (float): The actual power effectively processed (accounting for limits).
        """
        # 1. Clamp power to max rates
        power = np.clip(power, -self.max_charge_rate, self.max_charge_rate)
        
        # 2. Calculate potential energy change (kWh) for 15 min step (0.25 hours)
        dt = 0.25 
        
        if power >= 0:
            # Charging: Effective energy entering battery is reduced by efficiency (sqrt approx for one-way)
            # Typically efficiency is round-trip. We can apply sqrt(eff) on charge and discharge
            # or apply full eff on one side. Let's apply on charge for simplicity or standard modeling.
            # Common model: E_new = E_old + P * dt * eff_charge
            # Here assuming efficiency is round trip, let's say eff_charge = eff_discharge = sqrt(efficiency)
            eff_factor = np.sqrt(self.efficiency)
            energy_change = power * dt * eff_factor
        else:
            # Discharging: Energy leaving battery must be higher to deliver requested power
            # E_new = E_old + P * dt / eff_discharge
            eff_factor = np.sqrt(self.efficiency)
            energy_change = power * dt / eff_factor

        # 3. Check physical limits
        new_energy = self.current_energy + energy_change
        
        # Clamp to capacity
        if new_energy > self.capacity:
            new_energy = self.capacity
            # Back-calculate actual power possible
            # capacity = current + actual_power * dt * eff
            # actual_power = (capacity - current) / (dt * eff)
            if power > 0:
                 actual_power = (self.capacity - self.current_energy) / (dt * eff_factor)
            else:
                 # Should not happen if logic is correct (limit checks) but good for safety
                 actual_power = power 
        elif new_energy < 0:
            new_energy = 0
            # Back-calculate actual power
            # 0 = current + actual_power * dt / eff
            # actual_power = -current * eff / dt
            if power < 0:
                actual_power = -self.current_energy * eff_factor / dt
            else:
                actual_power = power
        else:
            actual_power = power

        self.current_energy = new_energy
        self.soc = self.current_energy / self.capacity
        
        return actual_power

    def reset(self):
        self.soc = 0.5
        self.current_energy = self.soc * self.capacity
