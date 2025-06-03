
# Adaptive Script Scheduler (MLP + Q-Learning Agent)

This project implements a self-learning scheduler for running Python scripts. It combines a Multi-Layer Perceptron (MLP) model with a Q-learning reinforcement learning agent to learn the ideal script run frequency based on real-world execution patterns and user interactions.

## Features

- Hybrid Learning Agent — MLP for stable baseline prediction + Q-learning for adaptive self-improvement.
- Interval-Based Scheduler — Runs Python scripts on intelligent, dynamic schedules.
- User Feedback Integration — Accepts real-time user preferences and manual triggers.
- Continuous Reinforcement Learning — Agent evolves from experience via RL-based policy updates.
- CLI Dashboard — Track all scripts, their intervals, and learning behavior.

## Project Structure

```
src/
├── agent.py              # Main self-learning scheduler (MLP + Q-learning)
├── cli_dashboard.py      # Terminal dashboard for monitoring
├── interval_model.pkl    # Pretrained MLP model
├── log.txt               # Script run log
├── q_experience.csv      # Q-learning experience log
├── q_model.pkl           # Q-learning model file
├── q_model.py            # Q-learning logic
├── retrain.py            # Periodic MLP retraining
├── runner.py             # Runs scripts based on the schedule
├── sample1.py / sample2.py / sample3.py   # Sample scripts
├── schedule.json         # Live schedule configuration
├── train_dummy_mlp.py    # Script to generate initial MLP model
├── training_data.csv     # Training data for MLP
├── user_feedback.log     # Manual run events
├── user_input.py         # CLI tool for user commands
├── user_pref.json        # Stores user preferences
```

## How the Scheduler Works

1. `runner.py` executes scripts based on `schedule.json`.
2. `agent.py` runs every 30 seconds and:
   - Computes average run intervals from `log.txt`
   - Collects manual run frequency and user preferences
   - Predicts a base interval using an MLP model
   - Chooses an action using Q-learning (decrease, keep, or increase)
   - Applies a small adjustment and updates the interval
   - Logs training data
3. Every 10 update loops, the Q-model retrains on collected experiences from `q_experience.csv`.

## Reinforcement Learning (Q-learning)

The agent's state includes:

- `avg_interval`: Average time between script executions
- `user_pref`: User’s recent preference (increase/decrease)
- `manual_count`: Number of recent manual runs

The Q-agent chooses one of 3 actions:

- `0` → decrease interval by 15%
- `1` → keep interval unchanged
- `2` → increase interval by 15%

Reward signals:

- `+1.0` for a recent manual run
- `+0.5` for any preference input
- `-0.3` if the interval is at its minimum
- `+0.2` if the prediction is stable (near average)
- `0.0` otherwise

Each cycle generates a `(state, action, reward, next_state)` tuple logged to `q_experience.csv`.

## Supported Commands

Run the agent and use:

```bash
python user_input.py increase sample1.py
python user_input.py decrease sample2.py
python user_input.py run sample3.py
```

These commands affect future scheduling decisions through both MLP and Q-learning.

## MLP Training

Initially, the MLP model is trained on synthetic data using:

```bash
python train_dummy_mlp.py
```

Features:
- `avg_interval`, `user_pref`, `manual_count`

Target interval is generated with:
```python
target = max(3, avg - manual - pref * 1.5 + random_noise)
```

Retraining on real data is triggered by:
```bash
python retrain.py
```

It uses the most recent records from `training_data.csv`.

## Logging & Learning

- `training_data.csv` logs interval prediction inputs
- `q_experience.csv` logs reinforcement learning experiences
- `log.txt` keeps a history of script executions
- `user_feedback.log` records all manual run commands
- `user_pref.json` stores user-stated frequency adjustments

## Getting Started

### Step 1: Install Dependencies
```bash
pip install scikit-learn joblib pandas
```

### Step 2: Train Dummy MLP
```bash
cd src/
python train_dummy_mlp.py
```

### Step 3: Launch Components (Each in a separate terminal)

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

