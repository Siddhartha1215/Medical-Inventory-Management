# utils/helpers.py

import numpy as np

def one_hot_category(category: str):
    mapping = {"A":0, "L":1, "O":2, "E":3}
    vec = np.zeros(4, dtype=float)
    vec[mapping[category]] = 1.0
    return vec

def one_hot_demand(demand: str):
    mapping = {"Low":0, "Medium":1, "High":2}
    vec = np.zeros(3, dtype=float)
    vec[mapping[demand]] = 1.0
    return vec

def state_to_vector(state):
    """
    Encode (Category, Demand) -> vector of length 7.
    """
    cat, dem = state
    return np.concatenate([one_hot_category(cat), one_hot_demand(dem)])
