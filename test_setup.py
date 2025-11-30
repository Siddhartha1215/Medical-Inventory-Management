# test_setup.py - Quick test to verify all imports work

print("Testing project setup...")

try:
    print("1. Testing numpy...")
    import numpy as np
    print("   ✓ numpy imported successfully")
    
    print("2. Testing matplotlib...")
    import matplotlib.pyplot as plt
    print("   ✓ matplotlib imported successfully")
    
    print("3. Testing torch...")
    import torch
    print(f"   ✓ torch imported successfully (device: {'CUDA' if torch.cuda.is_available() else 'CPU'})")
    
    print("4. Testing environment...")
    from env.inventory_env import MedicalInventoryEnv
    env = MedicalInventoryEnv()
    print("   ✓ Environment imported and created successfully")
    
    print("5. Testing Q-Learning agent...")
    from agent.q_learning_agent import QLearningAgent
    agent = QLearningAgent(env)
    print("   ✓ Q-Learning agent imported and created successfully")
    
    print("6. Testing DQN agent...")
    from agent.dqn_agent import DQNAgent
    dqn_agent = DQNAgent(env)
    print("   ✓ DQN agent imported and created successfully")
    
    print("7. Testing utilities...")
    from utils.helpers import state_to_vector
    from utils.plotting import plot_rewards
    test_state = ("A", "Low")
    vec = state_to_vector(test_state)
    print(f"   ✓ Utilities work (state vector shape: {vec.shape})")
    
    print("\n✅ All tests passed! Project is ready to run.")
    print("\nNext steps:")
    print("  1. Run Q-Learning: python main_q_learning.py")
    print("  2. Run DQN: python main_dqn.py")
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print("Please install dependencies: pip install -r requirements.txt")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

