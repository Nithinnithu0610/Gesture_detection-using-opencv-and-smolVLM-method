#!/usr/bin/env python3
import subprocess, time, os, signal, sys, csv, platform
from statistics import mean

def parse_log(path):
    cpu, mem, fps = [], [], []
    if not os.path.exists(path):
        return cpu, mem, fps
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("CPU Usage:"):
                try:
                    cpu.append(float(line.split("CPU Usage:")[1].strip().rstrip('%')))
                except:
                    pass
            if line.startswith("Memory Usage:"):
                try:
                    mem.append(float(line.split("Memory Usage:")[1].strip().split()[0]))
                except:
                    pass
            if line.startswith("Frames Processed Per Second:"):
                try:
                    fps.append(float(line.split("Frames Processed Per Second:")[1].strip()))
                except:
                    pass
    return cpu, mem, fps

def run_process(cmd, duration):
    p = subprocess.Popen(cmd)
    try:
        time.sleep(duration)
        if platform.system() == "Windows":
            p.terminate()  # clean stop for Windows
        else:
            p.send_signal(signal.SIGINT)  # works on Linux/macOS
    except Exception as e:
        print(f"Error stopping process: {e}")
        p.kill()
    try:
        p.wait(timeout=5)
    except:
        p.kill()

def main():
    dur = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    python = sys.executable
    modules = [
        ("opencv", "opencv/gesture_outputs.log"),
        ("smolvlm", "smolvlm/gesture_outputs.log")
    ]
    results = []

    for mod, log in modules:
        try:
            open(log, 'w').close()  # reset log
        except:
            pass
        print(f"Running {mod} for {dur}s... ensure camera is free")
        run_process([python, f"{mod}/gesture_{mod}.py"], dur)
        cpu, mem, fps = parse_log(log)
        results.append({
            "module": mod,
            "samples": len(cpu),
            "cpu_mean": round(mean(cpu), 2) if cpu else "",
            "mem_mean_mb": round(mean(mem), 2) if mem else "",
            "fps_mean": round(mean(fps), 2) if fps else ""
        })

    os.makedirs("benchmarks", exist_ok=True)
    with open("benchmarks/results_summary.csv", "w", newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=["module", "samples", "cpu_mean", "mem_mean_mb", "fps_mean"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print("✅ Saved benchmarks/results_summary.csv")

if __name__ == '__main__':
    main()
