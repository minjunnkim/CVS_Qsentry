import random
import joblib
from sklearn.neural_network import MLPRegressor
import pandas as pd

rows = []
for _ in range(1000):
    avg = random.uniform(4, 12)
    manual = random.randint(0, 4)
    pref = random.randint(-4, 4)

    target = max(3, avg - manual - pref * 1.5 + random.uniform(-0.5, 0.5))
    rows.append({"avg_interval": avg, "user_pref": pref, "manual_count": manual, "target_interval": target})

df = pd.DataFrame(rows)
X = df[["avg_interval", "user_pref", "manual_count"]]
y = df["target_interval"]

model = MLPRegressor(hidden_layer_sizes=(32,), max_iter=500)
model.fit(X, y)

joblib.dump(model, "src/interval_model.pkl")
print("[mlp] Dummy pretrained model saved to interval_model.pkl")
