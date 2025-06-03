
# Adaptive Script Scheduler (MLP + Q-Learning Agent)

This project implements an **agentic, self-learning scheduler** for running Python scripts. It combines a **Multi-Layer Perceptron (MLP)** model with a **Q-learning reinforcement learning agent** to learn the ideal script run frequency based on real-world execution patterns and user interactions.

## Key Features

- **Hybrid Learning Agent** — MLP for stable baseline prediction + Q-learning for adaptive self-improvement.
- **Interval-Based Scheduler** — Runs Python scripts on intelligent, dynamic schedules.
- **User Feedback Integration** — Accepts real-time user preferences and manual triggers.
- **Continuous Reinforcement Learning** — Agent evolves from experience via RL-based policy updates.
- **CLI Dashboard** — Track all scripts, their intervals, and learning behavior.

## Project Structure

```
Qsentry/
└── src/
    ├── sample1.py / sample2.py / sample3.py   # Example scripts
    ├── runner.py              # Executes scripts on a schedule
    ├── agent.py               # Core hybrid MLP + Q-learning agent
    ├── user_input.py          # Accepts run/increase/decrease feedback
    ├── cli_dashboard.py       # Terminal interface for monitoring
    ├── train_dummy_mlp.py     # Generates initial dummy MLP model
    ├── retrain.py             # Retrains MLP from live usage data
    ├── interval_model.pkl     # Persisted MLP model
    ├── schedule.json          # Dynamic script interval config
    ├── user_pref.json         # Feedback from increase/decrease
    ├── user_feedback.log      # Timestamps of manual runs
    ├── log.txt                # Historical run logs
    ├── training_data.csv      # Learning data
```

## How the Scheduler Works

1. **`runner.py`** executes each script according to the `schedule.json` intervals.
2. **`agent.py`** runs every 30 seconds:
   - Calculates average interval of each script (based on `log.txt`)
   - Collects manual run frequency and preference (`user_feedback.log`, `user_pref.json`)
   - Predicts a **baseline interval** using the **MLP model**
   - Adjusts the interval using **Q-learning** for fine-tuned adaptation
   - Logs training data and updates `schedule.json`
3. Every 10 loops (~5 minutes), **Q-model retrains** using collected state-action-reward experiences.

## Reinforcement Learning (Q-learning) Integration

The RL agent (`QModel`) takes state inputs:

- `avg_interval` — Time gap between last script runs
- `user_pref` — Whether user asked to increase/decrease frequency
- `manual_count` — How many times user manually ran the script

The agent chooses an action:

- `0` — Decrease interval by 15%
- `1` — Keep interval unchanged
- `2` — Increase interval by 15%

The reward system is:

- `+1.0` if user recently triggered the script manually
- `+0.5` if a preference (increase/decrease) exists
- `-0.3` penalty if interval hits the minimum threshold
- `+0.2` if new interval is close to recent average (stability bonus)

The Q-model stores (state, action, reward, next_state) tuples and retrains every 10 updates.

This allows the agent to **autonomously improve its scheduling policy over time**.

## Supported Commands

Use `python user_input.py` to issue feedback:

- `run sample2.py` — Manually trigger a script run
- `increase sample3.py` — Ask for more frequent runs
- `decrease sample1.py` — Ask for less frequent runs

These commands influence learning by affecting future interval predictions.

## MLP Training

Initial training is done with synthetic data:

```bash
python train_dummy_mlp.py
```

It simulates scheduling behavior with:

```python
target = max(3, avg - manual - pref * 1.5 + random_noise)
```

MLP model used:
```python
MLPRegressor(hidden_layer_sizes=(32,), max_iter=500)
```

The model is retrained later with real data via:
```bash
python retrain.py
```

## Live Learning & Logging

Each update cycle logs a training entry to `training_data.csv` with:

- Script name
- Average run interval
- User preference
- Manual run count
- Target interval

This allows both models to learn:

- MLP: retrained periodically on real usage patterns
- Q-learning: continually updates policy from experience

## Getting Started

### Step 1: Install Requirements
```bash
pip install scikit-learn joblib pandas
```

### Step 2: Train Dummy MLP
```bash
cd src/
python train_dummy_mlp.py
```

### Step 3: Start the System (in 4 terminals)

- Terminal 1:
  ```bash
  python runner.py
  ```

- Terminal 2:
  ```bash
  python agent.py
  ```

- Terminal 3:
  ```bash
  python user_input.py
  ```

- Terminal 4 (optional):
  ```bash
  python cli_dashboard.py
  ```

