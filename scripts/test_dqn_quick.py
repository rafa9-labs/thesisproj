"""Quick DQN-only smoke test."""
import subprocess, sys, time
t0 = time.time()
result = subprocess.run(
    [sys.executable, "tests/smoke_all_models.py", "--models", "dqn"],
    capture_output=True, text=True, encoding="utf-8", errors="replace"
)
print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
if result.stderr:
    # Filter TF noise
    for line in result.stderr.splitlines():
        if "RewardProcessWrapper" in line or "Error" in line or "FAIL" in line or "PASS" in line:
            print(f"STDERR: {line}")
print(f"\nExit code: {result.returncode} | Time: {time.time()-t0:.1f}s")