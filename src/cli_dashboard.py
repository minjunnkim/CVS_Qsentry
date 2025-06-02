import time
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import os

SCHEDULE_PATH = Path("src/schedule.json")
LOG_PATH = Path("src/log.txt")

def load_schedule():
    try:
        with open(SCHEDULE_PATH) as f:
            return json.load(f)
    except:
        return {}

def load_last_run_times():
    last_run = defaultdict(lambda: "Never")
    if not LOG_PATH.exists():
        return last_run

    with open(LOG_PATH) as f:
        for line in f:
            ts, script = line.strip().split(",")
            last_run[script] = float(ts)
    return last_run

def format_time(ts):
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S") if isinstance(ts, float) else ts

def print_dashboard():
    os.system("cls" if os.name == "nt" else "clear")
    print("───────────── SCHEDULER STATUS ─────────────")
    print(f"{'Script':<15}{'Last Run':<20}{'Next Run':<20}{'Interval (s)':<15}")
    print("-" * 65)

    schedule = load_schedule()
    last_runs = load_last_run_times()
    now = time.time()

    for script, interval in schedule.items():
        last_run = last_runs.get(script, "Never")
        if isinstance(last_run, float):
            next_run = last_run + interval
        else:
            next_run = "Unknown"

        print(f"{script:<15}{format_time(last_run):<20}{format_time(next_run):<20}{int(interval):<15}")

    print("-" * 65)
    print(f"Updated at {datetime.now().strftime('%H:%M:%S')}")

def main():
    while True:
        print_dashboard()
        time.sleep(3)

if __name__ == "__main__":
    main()
