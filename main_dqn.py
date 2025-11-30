import os
import numpy as np
from env.inventory_env import MedicalInventoryEnv
from agent.dqn_agent import DQNAgent
from utils.plotting import plot_rewards

def train_dqn(episodes=2000, episode_length=30):
    env = MedicalInventoryEnv(episode_length=episode_length)
    agent = DQNAgent(env, lr=1e-3, gamma=0.99, buffer_size=20000, batch_size=64, target_update=500, epsilon_decay=10000)
    rewards = []
    losses = []
    total_steps = 0

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action_idx = agent.select_action(state)
            action = env.actions[action_idx]
            next_state, reward, done, _ = env.step(action)
            agent.push_experience(state, action_idx, reward, next_state, done)
            loss = agent.learn()

            if loss is not None:
                losses.append(loss)

            total_reward += reward
            state = next_state
            total_steps += 1

            # update target network periodically
            if total_steps % agent.target_update == 0:
                agent.update_target()
        rewards.append(total_reward)

        if (ep+1) % 100 == 0:
            avg_recent = np.mean(rewards[-100:])
            print(f"[DQN] Episode {ep+1}/{episodes} reward={total_reward:.1f} avg100={avg_recent:.2f} eps={agent.epsilon:.3f}")

    os.makedirs("results", exist_ok=True)
    agent.save("results/dqn_policy.pth")
    np.save("results/dqn_rewards.npy", rewards)
    plot_rewards(rewards, title="DQN Rewards", savepath="results/dqn_rewards.png")
    print("DQN training done.")

if __name__ == "__main__":
    train_dqn(episodes=1500, episode_length=30)
    