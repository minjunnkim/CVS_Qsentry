# Adaptive Script Scheduler (MLP-Based Agent)

This project implements a self-learning, agentic scheduling system for Python scripts using a Multi-Layer Perceptron (MLP) model. The agent dynamically adjusts how frequently each script should run based on execution history and user input.

## Features

- Hot-reloading scheduler — runs multiple scripts based on live-updated intervals.
- MLP-based decision engine — predicts new intervals based on behavior patterns.
- User feedback — allows manual commands to influence future scheduling.
- Continuous self-learning — logs real data and periodically retrains the model.
- CLI dashboard — monitor script activity and schedule changes in real time.

## Project Structure

```
Qsentry/
└── src/
    ├── sample1.py / sample2.py / sample3.py   # Dummy scripts
    ├── runner.py              # Executes scripts per schedule
    ├── agent.py               # Predicts and updates schedule with learning
    ├── user_input.py          # Accepts commands (run/increase/decrease)
    ├── cli_dashboard.py       # Live terminal dashboard
    ├── train_dummy_mlp.py     # Initial synthetic MLP training
    ├── retrain.py             # Periodic retraining from collected data
    ├── interval_model.pkl     # Pretrained MLP model
    ├── schedule.json          # Active script schedule (sec)
    ├── user_pref.json         # Saved user preferences
    ├── user_feedback.log      # Manual run history
    ├── log.txt                # Script execution log
    ├── training_data.csv      # Collected training data (for retraining)
```

## How It Works

1. `runner.py` runs each script based on its interval from `schedule.json`
2. `agent.py` runs every 15 seconds and:
   - Computes recent average run interval for each script
   - Gathers user preference and recent manual run count
   - Predicts next interval using a loaded MLP model (`interval_model.pkl`)
   - Updates `schedule.json` and logs a training sample
3. After every 20 loops (5 mins), it retrains the model using `training_data.csv`
4. Manual commands via `user_input.py` affect long-term behavior

## Supported Commands

Use `python user_input.py` to issue commands:

- `run sample2.py` — Manually trigger script
- `increase sample3.py` — Tell agent to run it more frequently
- `decrease sample1.py` — Reduce its frequency

These commands influence `user_pref.json` and help the agent adapt.

## How the MLP Model is Trained

The model is a basic neural network trained using synthetic data in `train_dummy_mlp.py`. It learns to predict ideal run intervals using:

- `avg_interval`: recent average time between runs
- `user_pref`: user-specified bias (via increase/decrease)
- `manual_trigger_count`: how often the script was manually run

```python
MLPRegressor(hidden_layer_sizes=(32,), max_iter=500)
```

Target values during training are synthesized as:
```python
target = max(3, avg - manual - pref * 1.5 + random_noise)
```

**Note**: This is a placeholder. In the real application, we'd train on actual usage data. This is just a simple demonstration of the idea.

To retrain:
```bash
python train_dummy_mlp.py
```

## Continuous Self-Learning

As the agent runs, it collects and logs real data:

- After each schedule update, a record is appended to `training_data.csv`
- The MLP model is retrained every 20 updates (5 minutes)
- The model is automatically reloaded and used for future predictions

Each training entry contains:
- `script`
- `avg_interval`
- `user_pref`
- `manual_count`
- `target_interval`

This allows the scheduler to gradually adapt and personalize over time.

## Getting Started

### 1. Install Dependencies
```bash
pip install scikit-learn joblib
```

### 2. Train Initial MLP Model
```bash
cd src/
python train_dummy_mlp.py
```

### 3. Start the System
Open 4 terminals:

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

### 4. Issue Commands
Try:
```bash
increase sample1.py
decrease sample3.py
run sample2.py
```

## Notes

- The current learning is based on synthetic training and live user interactions.
- Real-world improvements require real usage logs and longer-running training sessions.