# Redis Connection Helper
# WSL2 IP changes on reboot. Run this to update .env:
#   python scripts\update_redis_ip.py
#
# Or manually: find WSL IP with `wsl hostname -I` and update API_REDIS_URL in .env

import subprocess
import re
import os

def get_wsl_ip():
    try:
        result = subprocess.run(
            ["wsl", "-d", "Ubuntu", "--", "hostname", "-I"],
            capture_output=True, text=True, timeout=10
        )
        ip = result.stdout.strip().split()[0]
        return ip
    except Exception as e:
        print(f"Error getting WSL IP: {e}")
        return None

def ensure_redis_running(wsl_ip):
    try:
        subprocess.run(
            ["wsl", "-d", "Ubuntu", "--", "bash", "-c",
             f"redis-cli -h {wsl_ip} ping"],
            capture_output=True, text=True, timeout=5
        )
        return True
    except Exception:
        return False

def start_redis(wsl_ip):
    print("Starting Redis in WSL...")
    try:
        subprocess.run(
            ["wsl", "-d", "Ubuntu", "--", "bash", "-c",
             "redis-server --bind 0.0.0.0 --daemonize yes && "
             "redis-cli config set bind '0.0.0.0' && "
             "redis-cli config set protected-mode no"],
            capture_output=True, text=True, timeout=15
        )
        return True
    except Exception as e:
        print(f"Error starting Redis: {e}")
        return False

def verify_connection(redis_url):
    try:
        import redis
        r = redis.from_url(redis_url)
        return r.ping()
    except Exception as e:
        print(f"Connection failed: {e}")
        return False

def main():
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    wsl_ip = get_wsl_ip()
    if not wsl_ip:
        print("Could not determine WSL IP. Is WSL running?")
        return 1

    redis_url = f"redis://{wsl_ip}:6379/0"
    print(f"WSL IP: {wsl_ip}")

    if not verify_connection(redis_url):
        print("Redis not reachable, starting...")
        start_redis(wsl_ip)
        import time
        time.sleep(2)
        if not verify_connection(redis_url):
            print("FATAL: Could not connect to Redis after starting it.")
            return 1

    print(f"Redis OK at {redis_url}")

    with open(env_path, "r") as f:
        lines = f.readlines()

    found = False
    new_lines = []
    for line in lines:
        if line.startswith("API_REDIS_URL="):
            new_lines.append(f"API_REDIS_URL={redis_url}\n")
            found = True
        else:
            new_lines.append(line)

    if not found:
        new_lines.append(f"\nAPI_REDIS_URL={redis_url}\n")

    with open(env_path, "w") as f:
        f.writelines(new_lines)

    print(f"Updated .env with API_REDIS_URL={redis_url}")
    print("\nNow restart your services (uvicorn + celery) to pick up the change.")
    return 0

if __name__ == "__main__":
    exit(main())