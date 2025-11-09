import gym
from gym import spaces
import numpy as np

class SatelliteEnvironment(gym.Env):
    def __init__(self):
        super(SatelliteEnvironment, self).__init__()
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        
        self.mu = 398600.4418
        self.dt = 60.0
        self.max_steps = 1000
        self.current_step = 0
        
        self.reset()
    
    def reset(self):
        self.current_step = 0
        altitude = 6778 + np.random.uniform(-100, 100)
        inclination = np.radians(51.6)
        
        self.state = np.array([
            altitude, 0, 0,
            0, np.sqrt(self.mu/altitude) * np.cos(inclination), np.sqrt(self.mu/altitude) * np.sin(inclination)
        ])
        
        self.target_orbit = np.array([6778, 0, 0, 0, np.sqrt(self.mu/6778), 0])
        self.debris_objects = self.generate_debris()
        
        return self.get_observation()
    
    def step(self, action):
        self.apply_thrust(action)
        self.propagate_orbit()
        
        reward = self.calculate_reward()
        done = self.current_step >= self.max_steps
        
        self.current_step += 1
        
        return self.get_observation(), reward, done, {}
    
    def apply_thrust(self, action):
        thrust = action * 0.1
        self.state[3:] += thrust
    
    def propagate_orbit(self):
        state = self.state
        k1 = self.orbital_dynamics(state)
        k2 = self.orbital_dynamics(state + 0.5 * self.dt * k1)
        k3 = self.orbital_dynamics(state + 0.5 * self.dt * k2)
        k4 = self.orbital_dynamics(state + self.dt * k3)
        
        self.state = state + (self.dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    
    def orbital_dynamics(self, state):
        x, y, z, vx, vy, vz = state
        r = np.array([x, y, z])
        r_norm = np.linalg.norm(r)
        
        ax = -self.mu * x / r_norm**3
        ay = -self.mu * y / r_norm**3
        az = -self.mu * z / r_norm**3
        
        return np.array([vx, vy, vz, ax, ay, az])
    
    def calculate_reward(self):
        position_error = np.linalg.norm(self.state[:3] - self.target_orbit[:3])
        velocity_error = np.linalg.norm(self.state[3:] - self.target_orbit[3:])
        
        collision_penalty = 0
        for debris in self.debris_objects:
            dist = np.linalg.norm(self.state[:3] - debris[:3])
            if dist < 100:
                collision_penalty -= 100
        
        fuel_penalty = -0.01 * np.linalg.norm(self.state[3:] - self.target_orbit[3:])
        
        return -position_error - velocity_error + collision_penalty + fuel_penalty
    
    def get_observation(self):
        obs = np.concatenate([self.state, self.target_orbit])
        return obs.astype(np.float32)
    
    def generate_debris(self):
        debris = []
        for _ in range(5):
            alt_variation = np.random.uniform(-200, 200)
            angle_variation = np.random.uniform(0, 2*np.pi)
            
            debris_state = np.array([
                6778 + alt_variation,
                0,
                0,
                0,
                np.sqrt(self.mu/(6778 + alt_variation)) * np.cos(angle_variation),
                np.sqrt(self.mu/(6778 + alt_variation)) * np.sin(angle_variation)
            ])
            debris.append(debris_state)
        
        return debris