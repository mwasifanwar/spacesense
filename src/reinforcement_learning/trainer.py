import numpy as np
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class RLTrainer:
    def __init__(self, env, agent):
        self.config = Config()
        self.env = env
        self.agent = agent
        self.buffer = ReplayBuffer(self.config.get('reinforcement_learning.buffer_size'))
        self.batch_size = self.config.get('reinforcement_learning.batch_size')
    
    def train(self, episodes=1000):
        rewards_history = []
        
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                self.buffer.push(state, action, reward, next_state, done)
                
                if len(self.buffer) >= self.batch_size:
                    states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
                    self.agent.update(states, actions, rewards, next_states, dones)
                
                state = next_state
                episode_reward += reward
            
            rewards_history.append(episode_reward)
            
            if episode % 100 == 0:
                print(f"mwasifanwar Episode {episode}, Reward: {episode_reward:.2f}")
        
        return rewards_history