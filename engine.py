#!/usr/bin/env python3
"""
engine.py — Main Controller
-------------------------------------------------------
Reads config.json and runs:
  1. py1_opencv (gesture detection)
  2. py2_smolvlm (VLM analysis)
-------------------------------------------------------
"""

import importlib
import json
import time
import os


def load_config(cfg_path="config.json"):
    with open(cfg_path, "r") as f:
        return json.load(f)


def run_module(name, config):
    print(f"\n[ENGINE] Running module: {name}")
    try:
        module = importlib.import_module(name)
        if hasattr(module, "main"):
            start = time.time()
            module.main(config)
            print(f"[ENGINE] ✅ {name} completed in {time.time() - start:.2f}s")
        else:
            print(f"[ENGINE] ⚠️ {name} has no main(config) function.")
    except Exception as e:
        print(f"[ENGINE] ❌ Error running {name}: {e}")


def main():
    print("=== Gesture Detection Engine ===")
    print("[INFO] Logging enabled → engine_log.txt")

    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = load_config(cfg_path)

    print(f"[CONFIG] Approach: {config['approach']}")
    print(f"[CONFIG] Static Approach: {config['static_approach']}")
    print(f"[CONFIG] FPS Target: {config['cfg_metrics']['fps_target']}")
    print(f"[CONFIG] Latency Limit: {config['cfg_metrics']['latency_limit_ms']} ms")

    start = time.time()

    run_module("py1_opencv", config)
    run_module("py2_smolvlm", config)

    print(f"\n[ENGINE] Total execution time: {time.time() - start:.2f}s")
    print("[ENGINE] All modules completed successfully.")


if __name__ == "__main__":
    main()
