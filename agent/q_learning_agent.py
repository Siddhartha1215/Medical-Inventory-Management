# agent/q_learning_agent.py

import numpy as np
from typing import Tuple

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((len(env.states), len(env.actions)))

    def choose_action(self, state_idx: int):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.env.actions))
        return int(np.argmax(self.Q[state_idx]))

    def update(self, s, a, r, s_next):
        best_next = np.max(self.Q[s_next])
        self.Q[s, a] += self.alpha * (r + self.gamma * best_next - self.Q[s, a])

    def save(self, path: str):
        np.save(path, self.Q)

    def load(self, path: str):
        self.Q = np.load(path)