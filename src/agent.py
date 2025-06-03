import json
import time
from collections import defaultdict
import joblib
from datetime import datetime
import os
import csv
import pandas as pd

from q_model import QModel

model_path = "src/interval_model.pkl"
model = joblib.load(model_path)

TRAINING_DATA = "src/training_data.csv"
RETRAIN_EVERY = 20
iteration = 0

qmodel = QModel()

def load_logs():
    logs = defaultdict(list)
    try:
        with open("src/log.txt") as f:
            for line in f:
                ts, script = line.strip().split(",")
                logs[script].append(float(ts))
    except:
        pass
    return logs

def compute_avg_interval(logs):
    avg = {}
    for script, ts_list in logs.items():
        ts_list = sorted(ts_list)[-6:]  # last 5 intervals
        if len(ts_list) >= 2:
            intervals = [ts_list[i+1] - ts_list[i] for i in range(len(ts_list)-1)]
            avg[script] = sum(intervals) / len(intervals)
    return avg

def load_user_pref():
    try:
        with open("src/user_pref.json") as f:
            return json.load(f)
    except:
        return {}

def load_user_feedback():
    counts = defaultdict(int)
    now = time.time()
    try:
        with open("src/user_feedback.log") as f:
            for line in f:
                ts, script, action = line.strip().split(",")
                ts = float(ts)
                if action == "manual_run" and now - ts < 180:
                    counts[script] += 1
    except:
        pass
    return counts

def log_training_sample(script, avg, pref, feedback, pred):
    if not os.path.exists(TRAINING_DATA):
        with open(TRAINING_DATA, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["script", "avg_interval", "user_pref", "manual_count", "target_interval"])
    with open(TRAINING_DATA, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([script, avg, pref, feedback, round(pred, 2)])

def update_schedule():
    global model

    try:
        with open("src/schedule.json") as f:
            schedule = json.load(f)
    except:
        print("[agent] Failed to load schedule.")
        return

    logs = load_logs()
    avg_intervals = compute_avg_interval(logs)
    prefs = load_user_pref()
    feedback_counts = load_user_feedback()

    for script in schedule:
        avg = avg_intervals.get(script, schedule[script])
        pref = prefs.get(script, 0)
        feedback = feedback_counts.get(script, 0)

        # Current state
        state = [avg, pref, feedback]
        features = pd.DataFrame([state], columns=["avg_interval", "user_pref", "manual_count"])
        base_pred = max(3, model.predict(features)[0])  # MLP baseline

        # RL action
        action = qmodel.select_action(state)
        if action == 0:
            pred = base_pred * 0.85
        elif action == 2:
            pred = base_pred * 1.15
        else:
            pred = base_pred

        pred = round(max(3, pred))
        schedule[script] = pred

        log_training_sample(script, avg, pref, feedback, pred)

        # Reward shaping
        if feedback > 0:
            reward = 1.0
        elif pref != 0:
            reward = 0.5
        elif pred <= 3:
            reward = -0.3
        else:
            reward = 0.1  # low positive for stability

        # Log RL experience
        next_logs = load_logs()
        next_avg = compute_avg_interval(next_logs).get(script, pred)
        next_state = [next_avg, pref, feedback]

        qmodel.log_experience(state, action, reward, next_state)

    with open("src/schedule.json", "w") as f:
        json.dump(schedule, f, indent=2)

    print(f"[agent] Schedule updated at {datetime.now()}: {schedule}")

def maybe_retrain():
    qmodel.train_from_log()
    print("[agent] Q-learning model retrained")

if __name__ == "__main__":
    while True:
        update_schedule()
        iteration += 1
        if iteration % RETRAIN_EVERY == 0:
            maybe_retrain()
        time.sleep(30)
