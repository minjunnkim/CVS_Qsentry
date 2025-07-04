import pandas as pd
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from pathlib import Path

def retrain_mlp_from_csv(log_file_path, model_file_path):
    log_file = Path(log_file_path)
    model_file = Path(model_file_path)

    if not log_file.exists():
        print("Training log file does not exist.")
        return

    df = pd.read_csv(log_file)
    if df.shape[0] < 10:
        print("Not enough data to retrain.")
        return

    X = df[["avg_interval", "user_pref", "manual_count"]]
    y = df["target_interval"]

    identity = FunctionTransformer(lambda x: x)

    pipeline = make_pipeline(identity, MLPRegressor(hidden_layer_sizes=(32,), max_iter=500, random_state=42))
    pipeline.fit(X, y)
    joblib.dump(pipeline, model_file)
    
    print(f"Model retrained and saved to {model_file_path}")