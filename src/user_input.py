import subprocess
import time
import json

def log_manual_trigger(script_name, action):
    with open("src/user_feedback.log", "a") as f:
        f.write(f"{time.time()},{script_name},{action}\n")

def update_user_pref(script, change):
    pref_path = "src/user_pref.json"
    try:
        with open(pref_path) as f:
            prefs = json.load(f)
    except:
        prefs = {}
    prefs[script] = prefs.get(script, 0) + change
    with open(pref_path, "w") as f:
        json.dump(prefs, f, indent=2)

def main():
    print("Commands: run <script.py>, increase <script.py>, decrease <script.py>")
    while True:
        cmd = input(">> ").strip()
        if cmd == "exit":
            break
        tokens = cmd.split()
        if len(tokens) != 2:
            continue
        action, script = tokens
        if script not in ["sample1.py", "sample2.py", "sample3.py"]:
            continue

        if action == "run":
            subprocess.run(["python", f"src/{script}"])
            log_manual_trigger(script, "manual_run")
            with open("src/log.txt", "a") as f:
                f.write(f"{time.time()},{script}\n")

        elif action == "increase":
            log_manual_trigger(script, action)
            update_user_pref(script, 2)
        elif action == "decrease":
            log_manual_trigger(script, action)
            update_user_pref(script, -2)

if __name__ == "__main__":
    main()
