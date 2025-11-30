import numpy as np
from env.inventory_env import MedicalInventoryEnv
from agent.q_learning_agent import QLearningAgent
from utils.plotting import plot_rewards

def train_qlearning(episodes=5000, episode_length=30):
    env = MedicalInventoryEnv(episode_length=episode_length)
    agent = QLearningAgent(env, alpha=0.1, gamma=0.9, epsilon=0.1)
    reward_history = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            s_idx = env.state_index(state)
            a_idx = agent.choose_action(s_idx)
            action = env.actions[a_idx]
            next_state, reward, done, _ = env.step(action)
            s_next_idx = env.state_index(next_state)
            agent.update(s_idx, a_idx, reward, s_next_idx)
            total_reward += reward
            state = next_state
        reward_history.append(total_reward)

        if (ep+1) % 500 == 0:
            print(f"[Q] Episode {ep+1}/{episodes} total_reward={total_reward:.1f}")

    agent.save("results/q_table.npy")
    plot_rewards(reward_history, title="Q-Learning Rewards", savepath="results/q_learning_rewards.png")
    print("Q-Learning training done.")

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    train_qlearning(episodes=3000, episode_length=30)

