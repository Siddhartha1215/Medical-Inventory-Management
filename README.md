# Medical Inventory RL (Q-Learning + DQN)

## Setup

1. Create virtualenv (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   (If you use CPU-only PyTorch install via `pip install torch --index-url https://download.pytorch.org/whl/cpu` or regular `pip install torch` if your environment supports CUDA.)

## Run Q-Learning

```bash
python main_q_learning.py
```

## Run DQN

```bash
python main_dqn.py
```

## Results

Trained artifacts and reward plots are saved under `./results/`

## Notes, tips & recommended hyperparameter experiments

- The environment is intentionally simple and discrete (12 states). Q-Learning should converge quickly for baseline results.
- The DQN is provided to show how you would scale to richer state encodings (e.g., including numerical stock counts, time-to-expiry, continuous demand features).
- Tune episodes, batch_size, lr, epsilon_decay, target_update based on results. Try longer training for DQN if rewards are noisy.
- You can modify `MedicalInventoryEnv._sample_next_demand` to make demand patterns more realistic (autoregressive, or seasonal spikes).
- Add logging: track stock-out frequency, expired frequency (count episodes / days with category O/E) — can be collected from environment during training loops.

# Medical-Inventory-management
