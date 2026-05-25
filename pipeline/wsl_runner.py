"""
WSL GPU Bridge -- invoked by the Celery worker when GPU models are requested.

Usage (inside WSL):
    python pipeline/wsl_runner.py --job-file=/tmp/job.json --job-id=<id>

The script:
1. Reads job config from a JSON file
2. Sets env vars (MODEL_LIST, SEEDS, etc.)
3. Runs the main_cli.py backtest pipeline
4. Prints [WSL_PROGRESS:...] lines to stdout for the parent process
5. Writes results JSON to a temp file for the parent process to read
"""

import json
import os
import sys
import time
import traceback
import argparse
import signal

_wsl_job_id = None
_wsl_results_path = None
_wsl_forced_stop = False


def _handle_stop(*_):
    global _wsl_forced_stop
    _wsl_forced_stop = True


signal.signal(signal.SIGTERM, _handle_stop)
signal.signal(signal.SIGINT, _handle_stop)


def _emit_progress(event: str, data: dict):
    if _wsl_job_id:
        print(f"[WSL_PROGRESS:{_wsl_job_id}:{event}:{json.dumps(data)}]", flush=True)


def _emit_event(event: str, msg: str):
    if _wsl_job_id:
        print(f"[WSL_EVENT:{_wsl_job_id}:{event}:{msg}]", flush=True)


def main():
    global _wsl_job_id, _wsl_results_path

    parser = argparse.ArgumentParser(description="WSL GPU Backtest Bridge")
    parser.add_argument("--job-file", required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()

    _wsl_job_id = args.job_id

    with open(args.job_file, "r") as f:
        job_config = json.load(f)

    pair = job_config.get("pair", "EURUSD")
    models = job_config.get("models", ["logistic"])
    start = job_config.get("start_date") or None
    end = job_config.get("end_date") or None
    months = job_config.get("months", 3)
    repeats = job_config.get("repeats", 1)
    seed = job_config.get("seed", 42)
    hpo_intensity = job_config.get("hpo_intensity", "quick")
    n_trials_override = job_config.get("n_trials")
    trading_costs = job_config.get("trading_costs", True)
    config_overrides = job_config.get("config_overrides", {})

    os.environ["MODEL_LIST"] = ",".join(models)
    os.environ["SEEDS"] = str(seed)
    os.environ["REPEATS"] = str(repeats)
    os.environ["N_MONTHS"] = str(months)
    os.environ["PAIR"] = pair
    os.environ["TRADING_COSTS"] = "1" if trading_costs else "0"
    os.environ["SMOKE_TEST"] = "0"
    os.environ["MLB_THREADS"] = "1"

    _wsl_results_path = os.path.join(job_config.get("results_dir", "."),
                                     f"wsl_result_{_wsl_job_id}.json")

    _emit_event("info", f"Starting WSL backtest: {len(models)} models, {months} months, {repeats} repeats")
    _emit_progress("started", {"pair": pair, "models": models, "months": months})

    t0 = time.time()

    try:
        from pipeline.main_cli import main as cli_main

        cli_main()

        elapsed = time.time() - t0
        _emit_progress("completed", {"elapsed_sec": round(elapsed, 1)})
        _emit_event("info", f"WSL backtest complete in {elapsed:.1f}s")

    except Exception as e:
        elapsed = time.time() - t0
        _emit_progress("failed", {"error": str(e), "elapsed_sec": round(elapsed, 1)})
        _emit_event("error", f"WSL backtest failed: {e}")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
