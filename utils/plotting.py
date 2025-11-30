# utils/plotting.py

import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(rewards, title="Episode Rewards", savepath=None):
    plt.figure(figsize=(8,4))
    plt.plot(rewards, alpha=0.7)

    # smooth
    if len(rewards) >= 10:
        smoothed = np.convolve(rewards, np.ones(10)/10, mode='valid')
        plt.plot(range(len(smoothed)), smoothed, label='smoothed', linewidth=2)

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(title)
    plt.grid(True)

    if savepath:
        plt.savefig(savepath, bbox_inches='tight')

    plt.show()
