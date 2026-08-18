#!/bin/bash
# KodaQuant dedicated compute node — Vast.ai onstart script.
# Paste this into the Vast.ai launch dialog ("On-start script" field).
# Idempotent (safe on re-start) and retries Docker Hub pulls (flaky on Vast).
set -e
log() { echo "[KodaQuant] $(date -u +%H:%M:%S) $*"; }

log "bootstrapping dedicated compute node"
export DEBIAN_FRONTEND=noninteractive

# 1. Base tools + docker
apt-get update -qq
apt-get install -y -qq git curl gpg docker.io docker-compose-plugin

# 2. NVIDIA container toolkit (required by the compose GPU reservation)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update -qq
apt-get install -y -qq nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker

# 3. Ensure the Docker daemon is running (Vast instances often have no systemd)
systemctl restart docker 2>/dev/null || true
if ! docker info >/dev/null 2>&1; then
  nohup dockerd > /var/log/dockerd.log 2>&1 &
  sleep 10
fi
docker info >/dev/null 2>&1 || { log "dockerd failed to start"; exit 1; }
log "docker daemon up"

# 4. Repo (idempotent — onstart re-runs on every instance start)
if [ -d /root/thesisproj/.git ]; then
  cd /root/thesisproj
  git pull --ff-only || true
else
  git clone https://github.com/rafa9-labs/thesisproj.git /root/thesisproj
  cd /root/thesisproj
fi
log "repo ready"

# 5. Start the stack with retries (Docker Hub resets are common on Vast)
for attempt in 1 2 3 4 5; do
  log "compose up attempt $attempt"
  if docker compose up -d api worker redis; then
    break
  fi
  log "attempt $attempt failed — retrying in 20s"
  sleep 20
done

# 6. Wait for the API to become healthy
for i in $(seq 1 36); do
  if curl -sf http://localhost:8000/api/v1/health >/dev/null; then
    log "API healthy on port 8000"
    exit 0
  fi
  sleep 5
done

log "WARNING: API not healthy within 180s — check 'docker compose logs api'"
exit 1
