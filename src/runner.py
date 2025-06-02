import json
import time
import subprocess
from threading import Thread, Lock
import os
from pathlib import Path

schedule_path = Path("src/schedule.json")
log_path = Path("src/log.txt")

intervals = {}
interval_lock = Lock()

def load_schedule():
    try:
        with open(schedule_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"[runner] Failed to load schedule: {e}")
        return {}

def monitor_schedule():
    last_mtime = 0
    global intervals
    while True:
        try:
            current_mtime = os.path.getmtime(schedule_path)
            if current_mtime != last_mtime:
                last_mtime = current_mtime
                new_schedule = load_schedule()
                with interval_lock:
                    intervals = new_schedule
                print(f"[runner] Reloaded schedule: {new_schedule}")
        except Exception as e:
            print(f"[runner] Error monitoring schedule: {e}")
        time.sleep(2)  # fast check

def run_script_periodically(script_name):
    next_run = time.time()
    while True:
        with interval_lock:
            interval = intervals.get(script_name)
        if interval is None:
            time.sleep(0.5)
            continue
        now = time.time()
        if now >= next_run:
            subprocess.run(["python", f"src/{script_name}"])
            with open(log_path, "a") as f:
                f.write(f"{time.time()},{script_name}\n")
            next_run = now + interval
        else:
            time.sleep(0.5)

def main():
    global intervals
    intervals = load_schedule()
    Thread(target=monitor_schedule, daemon=True).start()

    threads = []
    for script in intervals:
        t = Thread(target=run_script_periodically, args=(script,))
        t.daemon = True
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

if __name__ == "__main__":
    main()
