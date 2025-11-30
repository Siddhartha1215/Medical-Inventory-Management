# agent/dqn_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple
from agent.replay_buffer import ReplayBuffer
from utils.helpers import state_to_vector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, env, lr=1e-3, gamma=0.99, buffer_size=50000, batch_size=64, target_update=1000, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=5000):
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.input_dim = 7  # 4 category one-hot + 3 demand one-hot
        self.output_dim = len(env.actions)
        self.policy_net = QNetwork(self.input_dim, self.output_dim).to(device)
        self.target_net = QNetwork(self.input_dim, self.output_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        # epsilon schedule
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

    def select_action(self, state_tuple):
        state_v = state_to_vector(state_tuple).astype(np.float32)
        self.steps_done += 1

        # linear decay
        self.epsilon = max(self.epsilon_end, self.epsilon_start - (self.steps_done / self.epsilon_decay)*(self.epsilon_start - self.epsilon_end))
        
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.output_dim)
        else:
            with torch.no_grad():
                s = torch.from_numpy(state_v).unsqueeze(0).to(device)
                qs = self.policy_net(s)
                return int(torch.argmax(qs, dim=1).item())

    def push_experience(self, s, a, r, s_next, done):
        s_v = state_to_vector(s).astype(np.float32)
        s_next_v = state_to_vector(s_next).astype(np.float32)
        self.replay_buffer.push(s_v, a, r, s_next_v, done)

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states_tensor = torch.from_numpy(states).to(device)
        actions_tensor = torch.from_numpy(actions).long().to(device)
        rewards_tensor = torch.from_numpy(rewards).float().to(device)
        next_states_tensor = torch.from_numpy(next_states).to(device)
        dones_tensor = torch.from_numpy(dones).float().to(device)
        q_values = self.policy_net(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states_tensor).max(1)[0]
            expected_q = rewards_tensor + (1.0 - dones_tensor) * self.gamma * next_q_values
        loss = nn.functional.mse_loss(q_values, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str):
        self.policy_net.load_state_dict(torch.load(path, map_location=device))
        self.update_target()