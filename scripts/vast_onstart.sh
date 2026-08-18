#!/bin/bash
# KodaQuant dedicated compute node — Vast.ai onstart script.
# Paste this into the Vast.ai launch dialog ("On-start script" field).
# Idempotent (safe on re-start) and retries Docker Hub pulls (flaky on Vast).
#
# REQUIRED: fill ALLOW_IP with your public IP (curl ifconfig.me) so the
# firewall opens port 8000 only for you. If left empty, ufw is skipped and
# the API port is PUBLIC — anyone who scans the instance can use it.
#
# Every step is logged to /var/log/onstart.log — after a failure, read it:
#   tail -100 /var/log/onstart.log
set -e
ALLOW_IP="89.180.47.218"
log() { echo "[KodaQuant] $(date -u +%H:%M:%S) $*"; }
exec > >(tee -a /var/log/onstart.log) 2>&1
trap 'log "FAILED at line $LINENO (last exit code: $?)"' ERR

log "bootstrapping dedicated compute node (log: /var/log/onstart.log)"
export DEBIAN_FRONTEND=noninteractive

# 0. Baseline checks
log "image: $(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME || echo unknown)"
nvidia-smi >/dev/null 2>&1 && log "GPU visible on host: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)" || log "WARNING: nvidia-smi not found (host GPU check failed)"
which python3 && python3 --version

# 1. Base tools + docker (apt docker.io first, fallback to get.docker.com)
apt-get update -qq
if ! apt-get install -y -qq git curl gpg docker.io docker-compose-plugin; then
  log "apt docker.io failed — trying get.docker.com"
  curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
  sh /tmp/get-docker.sh
fi

# 2. NVIDIA container toolkit (required by the compose GPU reservation)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update -qq
apt-get install -y -qq nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
log "nvidia toolkit configured"

# 3. Ensure the Docker daemon is running (Vast instances often have no systemd)
systemctl restart docker 2>/dev/null || true
if ! docker info >/dev/null 2>&1; then
  nohup dockerd > /var/log/dockerd.log 2>&1 &
  sleep 10
fi
docker info >/dev/null 2>&1 || { log "dockerd failed to start — see /var/log/dockerd.log"; exit 1; }
docker info 2>/dev/null | grep -i "runtime" || true
log "docker daemon up"

# 3b. Firewall — keep SSH open, restrict API port 8000 to your IP
if [ -n "$ALLOW_IP" ]; then
  apt-get install -y -qq ufw || true
  ufw allow 22/tcp
  ufw allow from "$ALLOW_IP" to any port 8000 proto tcp
  ufw --force enable || true
  log "ufw enabled: ssh + port 8000 from $ALLOW_IP"
else
  log "WARNING: ALLOW_IP empty — skipping ufw (API port 8000 is PUBLIC)"
fi

# 4. Repo (idempotent — onstart re-runs on every instance start)
if [ -d /root/thesisproj/.git ]; then
  cd /root/thesisproj
  git pull --ff-only || true
else
  git clone https://github.com/rafa9-labs/thesisproj.git /root/thesisproj
  cd /root/thesisproj
fi
git log --oneline -1 || true
log "repo ready"

# 5. Start the stack with retries (Docker Hub resets are common on Vast)
for attempt in 1 2 3 4 5; do
  log "compose up attempt $attempt"
  if docker compose up -d api worker redis; then
    log "compose up succeeded"
    break
  fi
  log "attempt $attempt failed — retrying in 20s"
  sleep 20
done

# 5b. Report container state
docker ps -a --format 'table {{.Names}}\t{{.Status}}' || true

# 6. Wait for the API to become healthy (first boot may need up to ~8 min)
for i in $(seq 1 96); do
  if curl -sf http://localhost:8000/api/v1/health >/dev/null; then
    log "API healthy on port 8000"
    exit 0
  fi
  sleep 5
done

log "WARNING: API not healthy within 480s — check 'docker compose logs api'"
docker compose logs --tail 50 api || true
exit 1