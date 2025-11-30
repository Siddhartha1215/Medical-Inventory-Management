# env/inventory_env.py
import random
from typing import Tuple, List
class MedicalInventoryEnv:

    """

    Discrete environment based on your Phase-I MDP.

    State: (Category, PrevDemand)

      - Category in ["A","L","O","E"]

      - PrevDemand in ["Low","Medium","High"]

    Actions: ["NoOrder","SmallOrder","MediumOrder","LargeOrder"]

    """
    def __init__(self, episode_length: int = 30, seed: int = 0):
        random.seed(seed)
        self.categories = ["A", "L", "O", "E"]
        self.demands = ["Low", "Medium", "High"]
        self.actions = ["NoOrder", "SmallOrder", "MediumOrder", "LargeOrder"]
        self.states = [(c, d) for c in self.categories for d in self.demands]  # 12 states
        self.reward_map = {"A": 10, "L": 5, "O": -10, "E": -15}
        # Transition probabilities as specified in your doc
        self.transition_prob = {
            "NoOrder": {
                "Low":    {"A":0.60, "L":0.30, "O":0.10, "E":0.00},
                "Medium": {"A":0.20, "L":0.50, "O":0.25, "E":0.05},
                "High":   {"A":0.05, "L":0.20, "O":0.60, "E":0.15},
            },
            "SmallOrder": {
                "Low":    {"A":0.75, "L":0.20, "O":0.05, "E":0.00},
                "Medium": {"A":0.40, "L":0.40, "O":0.15, "E":0.05},
                "High":   {"A":0.15, "L":0.35, "O":0.40, "E":0.10},
            },
            "MediumOrder": {
                "Low":    {"A":0.90, "L":0.10, "O":0.00, "E":0.00},
                "Medium": {"A":0.60, "L":0.30, "O":0.05, "E":0.05},
                "High":   {"A":0.30, "L":0.50, "O":0.15, "E":0.05},
            },
            "LargeOrder": {
                "Low":    {"A":0.98, "L":0.02, "O":0.00, "E":0.00},
                "Medium": {"A":0.85, "L":0.10, "O":0.03, "E":0.02},
                "High":   {"A":0.60, "L":0.30, "O":0.05, "E":0.05},
            }

        }

        # episode management
        self.episode_length = episode_length
        self.t = 0
        self.state = None
        self.seed = seed
    def reset(self, init_state: Tuple[str,str]=("A","Low")) -> Tuple[str,str]:
        """Reset environment. Optionally supply initial state tuple."""
        self.state = init_state
        self.t = 0
        return self.state
    
    def _sample_next_category(self, action: str, prev_demand: str) -> str:
        probs = self.transition_prob[action][prev_demand]
        categories = list(probs.keys())
        weights = list(probs.values())
        # random.choices for sampling with weights
        return random.choices(categories, weights=weights, k=1)[0]
    
    def _sample_next_demand(self, prev_demand: str) -> str:
        # Simple demand dynamics: small chance of staying same, or change randomly.
        # You can refine this later.
        r = random.random()
        if r < 0.6:
            # demand tends to persist with some noise
            return prev_demand
        else:
            return random.choice(self.demands)

    def step(self, action: str):

        """

        Perform action (string) and return next_state, reward, done, info

        """
        assert action in self.actions, f"Unknown action {action}"
        category, prev_demand = self.state
        next_category = self._sample_next_category(action, prev_demand)
        next_demand = self._sample_next_demand(prev_demand)
        next_state = (next_category, next_demand)
        reward = self.reward_map[next_category]
        self.state = next_state
        self.t += 1
        done = False

        # Terminal: episode_length reached OR 3 consecutive failure days can be implemented externally.
        if self.t >= self.episode_length:
            done = True
        info = {}
        return next_state, reward, done, info

    def render(self):
        print(f"t={self.t}, state={self.state}")
    # small helpers for indices
    
    def state_index(self, state):
        return self.states.index(state)

    def action_index(self, action):
        return self.actions.index(action)

