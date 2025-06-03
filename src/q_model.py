import os
import random
import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

ACTIONS = {0: "decrease", 1: "keep", 2: "increase"}

class QModel:
    def __init__(self, model_path="src/q_model.pkl", log_path="src/q_experience.csv"):
        self.model_path = model_path
        self.log_path = log_path

        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            self.model = MLPRegressor(hidden_layer_sizes=(32,), max_iter=500)
            # Warm-start with dummy data so .predict() works
            X_dummy = np.random.rand(5, 3)
            y_dummy = np.random.rand(5, 3)
            self.model.fit(X_dummy, y_dummy)
            joblib.dump(self.model, model_path)

    def select_action(self, state, epsilon=0.2):
        """Epsilon-greedy policy"""
        if random.random() < epsilon:
            return random.choice([0, 1, 2])  # explore
        q_values = self.model.predict([state])[0]
        return int(np.argmax(q_values))     # exploit

    def log_experience(self, state, action, reward, next_state):
        row = state + [action, reward] + next_state
        columns = [
            "avg", "pref", "manual", "action", "reward",
            "next_avg", "next_pref", "next_manual"
        ]
        df = pd.DataFrame([row], columns=columns)
        df.to_csv(self.log_path, mode="a", header=not os.path.exists(self.log_path), index=False)

    def train_from_log(self, alpha=0.1, gamma=0.9):
        if not os.path.exists(self.log_path):
            return

        df = pd.read_csv(self.log_path)
        X = df[["avg", "pref", "manual"]].values
        actions = df["action"].values
        rewards = df["reward"].values
        next_X = df[["next_avg", "next_pref", "next_manual"]].values

        q_values = self.model.predict(X)
        next_q_values = self.model.predict(next_X)

        for i in range(len(X)):
            a = actions[i]
            r = rewards[i]
            max_next = np.max(next_q_values[i])
            q_values[i][a] = (1 - alpha) * q_values[i][a] + alpha * (r + gamma * max_next)

        self.model.fit(X, q_values)
        joblib.dump(self.model, self.model_path)
