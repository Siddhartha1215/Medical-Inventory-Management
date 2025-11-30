# Quick Start Guide - Medical Inventory RL

## Step-by-Step Execution Instructions

### Step 1: Install Dependencies

Open your terminal/PowerShell in the project directory and run:

```bash
pip install -r requirements.txt
```

**Note for PyTorch:**
- If you have CUDA/GPU: `pip install torch` (will install GPU version)
- If CPU-only: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

### Step 2: Run Q-Learning (Tabular Method)

This is faster and simpler - good for initial testing:

```bash
python main_q_learning.py
```

**What happens:**
- Trains for 3000 episodes
- Prints progress every 500 episodes
- Saves Q-table to `results/q_table.npy`
- Generates reward plot: `results/q_learning_rewards.png`
- Shows plot window

**Expected output:**
```
[Q] Episode 500/3000 total_reward=150.0
[Q] Episode 1000/3000 total_reward=180.0
...
Q-Learning training done.
```

### Step 3: Run DQN (Deep Learning Method)

This uses neural networks and takes longer:

```bash
python main_dqn.py
```

**What happens:**
- Trains for 1500 episodes
- Prints progress every 100 episodes
- Saves model to `results/dqn_policy.pth`
- Saves rewards to `results/dqn_rewards.npy`
- Generates reward plot: `results/dqn_rewards.png`

**Expected output:**
```
[DQN] Episode 100/1500 reward=120.0 avg100=95.50 eps=0.950
[DQN] Episode 200/1500 reward=150.0 avg100=110.25 eps=0.900
...
DQN training done.
```

## Troubleshooting

### Import Errors
If you get `ModuleNotFoundError`, make sure you're in the project root directory:
```bash
cd "C:\Users\vasur\OneDrive\Desktop\RL\Inventory Management"
```

### PyTorch Installation Issues
If PyTorch fails to install, try:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Plot Window Not Showing
If plots don't display, the images are still saved in `results/` folder. You can open them manually.

## Results Location

All outputs are saved in the `results/` folder:
- `q_table.npy` - Trained Q-table (Q-Learning)
- `q_learning_rewards.png` - Reward plot (Q-Learning)
- `dqn_policy.pth` - Trained neural network (DQN)
- `dqn_rewards.npy` - Reward history (DQN)
- `dqn_rewards.png` - Reward plot (DQN)

## Quick Test

To verify everything works, run a quick test:
```bash
python -c "from env.inventory_env import MedicalInventoryEnv; env = MedicalInventoryEnv(); print('Environment loaded successfully!')"
```

