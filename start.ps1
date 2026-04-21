<# 
.SYNOPSIS
    FX ML Backtester — launcher
.DESCRIPTION
    Starts the full stack (Docker or native) and provides a menu for
    common operations: backtests, logs, health checks, teardown.
.USAGE
    .\start.ps1              # interactive menu
    .\start.ps1 docker       # launch Docker stack directly  
    .\start.ps1 native       # launch native (venv) stack directly
    .\start.ps1 stop         # stop all services
    .\start.ps1 status       # show service health
#>
param(
    [ValidateSet("docker","native","stop","status","logs","test","")]
    [string]$Action = ""
)

$ErrorActionPreference = "SilentlyContinue"

# ─── colours ───────────────────────────────────────────────────────
function C($t,$c="White")  { Write-Host $t -ForegroundColor $c -NoNewline }
function H($t)              { Write-Host ""; Write-Host "  $t" -ForegroundColor Cyan }
function OK($t)             { C "  OK  " Green;   Write-Host " $t" }
function WARN($t)           { C " WARN" Yellow;   Write-Host " $t" }
function FAIL($t)           { C " FAIL" Red;      Write-Host " $t" }
function INFO($t)           { C "  -> " DarkGray; Write-Host $t }

# ─── config ────────────────────────────────────────────────────────
$PROJECT   = $PSScriptRoot
$VENV      = Join-Path $PROJECT "venv"
$PYTHON    = Join-Path $VENV "Scripts\python.exe"
$PIP       = Join-Path $VENV "Scripts\pip.exe"
$API_PORT  = 8000
$FE_PORT   = 5173
$REDIS_PORT= 6379
$API_URL   = "http://localhost:${API_PORT}/api/v1/health"

# ─── prerequisites ─────────────────────────────────────────────────
function Test-Docker {
    try { docker info 2>&1 | Out-Null; return $true }
    catch { return $false }
}

function Test-DockerRunning {
    try { docker ps 2>&1 | Out-Null; return $true }
    catch { return $false }
}

function Ensure-Venv {
    if (Test-Path $PYTHON) { return $true }
    H "Creating Python venv ..."
    python -m venv $VENV
    if (-not $?) { FAIL "Could not create venv"; return $false }
    INFO "Installing requirements ..."
    & $PIP install --upgrade pip --quiet 2>&1 | Out-Null
    & $PIP install -r (Join-Path $PROJECT "requirements.txt") --quiet 2>&1 | Out-Null
    OK "venv ready"
    return $true
}

function Wait-For($url, $label, $timeout=60) {
    $t = 0
    while ($t -lt $timeout) {
        try {
            $r = Invoke-WebRequest -Uri $url -TimeoutSec 2 -UseBasicParsing
            if ($r.StatusCode -eq 200) { OK "$label is up"; return $true }
        } catch {}
        Start-Sleep -Milliseconds 500
        $t += 0.5
    }
    FAIL "$label did not respond within ${timeout}s"
    return $false
}

# ─── DOCKER STACK ──────────────────────────────────────────────────
function Start-DockerStack {
    H "Starting Docker stack ..."
    
    if (-not (Test-DockerRunning)) {
        WARN "Docker Desktop not running. Starting ..."
        Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
        INFO "Waiting for Docker daemon ..."
        $t = 0
        while ($t -lt 120) {
            if (Test-DockerRunning) { break }
            Start-Sleep -Seconds 2; $t += 2
        }
        if (-not (Test-DockerRunning)) { FAIL "Docker Desktop did not start"; return }
        OK "Docker Desktop ready"
    }

    Set-Location $PROJECT
    INFO "Building & starting containers ..."
    docker compose up -d --build 2>&1 | ForEach-Object {
        if ($_ -match "error|fail") { WARN $_ } else { INFO $_ }
    }

    H "Waiting for services ..."
    Wait-For $API_URL "API" 90 | Out-Null

    H "Stack status"
    docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}" 2>&1 | ForEach-Object { INFO $_ }

    H "Ready"
    Write-Host ""
    C "  Frontend : " Yellow; Write-Host "http://localhost:${FE_PORT}"
    C "  API docs : " Yellow; Write-Host "http://localhost:${API_PORT}/docs"
    C "  Health   : " Yellow; Write-Host $API_URL
    Write-Host ""
    C "  Logs     : " DarkGray; Write-Host "docker compose logs -f"
    C "  Worker   : " DarkGray; Write-Host "docker compose logs -f worker"
    C "  Stop     : " DarkGray; Write-Host ".\start.ps1 stop"
    Write-Host ""
}

# ─── NATIVE STACK ──────────────────────────────────────────────────
function Start-NativeStack {
    H "Starting native stack ..."
    
    if (-not (Ensure-Venv)) { return }

    # Check Redis
    $redisUp = $false
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:${REDIS_PORT}" -TimeoutSec 1 -UseBasicParsing -ErrorAction Stop
    } catch {
        try {
            $tcp = New-Object System.Net.Sockets.TcpClient
            $tcp.Connect("127.0.0.1", $REDIS_PORT)
            $redisUp = $true; $tcp.Close()
        } catch { $redisUp = $false }
    }
    if (-not $redisUp) {
        WARN "Redis not detected on port ${REDIS_PORT}"
        if (Test-DockerRunning) {
            INFO "Starting Redis container ..."
            docker run -d --name fx-redis -p "${REDIS_PORT}:${REDIS_PORT}" redis:7-alpine 2>&1 | Out-Null
            Start-Sleep -Seconds 3
            OK "Redis started"
        } else {
            FAIL "Start Docker Desktop first, or run Redis manually"
            return
        }
    } else { OK "Redis detected on port ${REDIS_PORT}" }

    # Start API
    H "Starting API server (port ${API_PORT}) ..."
    $apiProc = Start-Process -FilePath $PYTHON -ArgumentList @(
        "-m", "uvicorn", "api.main:app",
        "--host", "127.0.0.1", "--port", $API_PORT, "--reload"
    ) -PassThru -WindowStyle Hidden
    INFO "PID: $($apiProc.Id)"
    Wait-For $API_URL "API" 30 | Out-Null

    # Start Celery worker
    H "Starting Celery worker ..."
    $workerProc = Start-Process -FilePath $PYTHON -ArgumentList @(
        "-m", "celery", "-A", "api.tasks.celery_app",
        "worker", "--loglevel=info", "--concurrency=1", "-E", "-P", "solo"
    ) -PassThru -WindowStyle Hidden
    INFO "PID: $($workerProc.Id)"
    OK "Worker started"

    # Start Frontend
    H "Starting Frontend dev server (port ${FE_PORT}) ..."
    $feDir = Join-Path $PROJECT "frontend"
    $npm = Get-Command npm -ErrorAction SilentlyContinue
    if ($npm) {
        $feProc = Start-Process -FilePath "npm" -ArgumentList @("run", "dev") `
            -PassThru -WindowStyle Hidden -WorkingDirectory $feDir
        INFO "PID: $($feProc.Id)"
        Start-Sleep -Seconds 3
        OK "Frontend started"
    } else {
        WARN "npm not found — start frontend manually: cd frontend ; npm run dev"
    }

    # Save PIDs for cleanup
    $pids = @($apiProc.Id, $workerProc.Id)
    if ($feProc) { $pids += $feProc.Id }
    $pids | Set-Content (Join-Path $PROJECT ".stack_pids")

    H "Ready"
    Write-Host ""
    C "  Frontend : " Yellow; Write-Host "http://localhost:${FE_PORT}"
    C "  API docs : " Yellow; Write-Host "http://localhost:${API_PORT}/docs"
    C "  Health   : " Yellow; Write-Host $API_URL
    Write-Host ""
    C "  Stop     : " DarkGray; Write-Host ".\start.ps1 stop"
    C "  Logs     : " DarkGray; Write-Host ".\start.ps1 logs"
    Write-Host ""
}

# ─── STOP ──────────────────────────────────────────────────────────
function Stop-All {
    H "Stopping services ..."
    
    # Docker
    if (Test-DockerRunning) {
        Set-Location $PROJECT
        docker compose down 2>&1 | ForEach-Object { INFO $_ }
    }

    # Native PIDs
    $pidFile = Join-Path $PROJECT ".stack_pids"
    if (Test-Path $pidFile) {
        $pids = Get-Content $pidFile
        foreach ($p in $pids) {
            try {
                $proc = Get-Process -Id $p -ErrorAction Stop
                Stop-Process -Id $p -Force
                INFO "Killed PID $p ($($proc.ProcessName))"
            } catch {}
        }
        Remove-Item $pidFile -Force
    }

    # Fallback: kill by port
    foreach ($port in @($API_PORT, $FE_PORT)) {
        $conn = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
        foreach ($c in $conn) {
            try { Stop-Process -Id $c.OwningProcess -Force -ErrorAction Stop } catch {}
        }
    }

    # Kill celery if still running
    Get-Process -Name "celery" -ErrorAction SilentlyContinue | Stop-Process -Force
    # Kill uvicorn workers
    Get-Process -Name "uvicorn" -ErrorAction SilentlyContinue | Stop-Process -Force

    OK "All services stopped"
}

# ─── STATUS ────────────────────────────────────────────────────────
function Show-Status {
    H "Service health"

    # Docker containers
    if (Test-DockerRunning) {
        Set-Location $PROJECT
        Write-Host ""
        docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}" 2>&1 | ForEach-Object { INFO $_ }
    }

    Write-Host ""
    # API
    try {
        $r = Invoke-WebRequest -Uri $API_URL -TimeoutSec 3 -UseBasicParsing
        $body = $r.Content | ConvertFrom-Json
        C "  API    : " Green; Write-Host "up (v$($body.version), redis=$($body.redis), db_rows=$($body.db_rows))"
    } catch {
        FAIL "API not responding on port ${API_PORT}"
    }

    # Redis
    try {
        $tcp = New-Object System.Net.Sockets.TcpClient
        $tcp.Connect("127.0.0.1", $REDIS_PORT)
        OK "Redis on port ${REDIS_PORT}"; $tcp.Close()
    } catch {
        FAIL "Redis not reachable on port ${REDIS_PORT}"
    }

    # Frontend
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:${FE_PORT}" -TimeoutSec 3 -UseBasicParsing
        OK "Frontend on port ${FE_PORT}"
    } catch {
        FAIL "Frontend not responding on port ${FE_PORT}"
    }

    # Celery worker check
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:${API_PORT}/api/v1/backtest?limit=1" -TimeoutSec 3 -UseBasicParsing
        OK "Backtest endpoint reachable"
    } catch {
        WARN "Backtest endpoint not reachable"
    }

    Write-Host ""
    INFO "Run '.\start.ps1 logs' to see live output"
}

# ─── LOGS ──────────────────────────────────────────────────────────
function Show-Logs {
    H "Live logs (Ctrl+C to exit)"
    Write-Host ""
    
    if (Test-DockerRunning) {
        Set-Location $PROJECT
        docker compose logs -f --tail 50
    } else {
        WARN "Docker not running. Showing native process output is not available."
        INFO "Check terminal windows for uvicorn / celery / npm output."
    }
}

# ─── TEST ──────────────────────────────────────────────────────────
function Run-QuickTest {
    H "Quick backtest test ..."
    
    try {
        $r = Invoke-WebRequest -Uri $API_URL -TimeoutSec 3 -UseBasicParsing
    } catch {
        FAIL "API is not running. Start it first: .\start.ps1 docker"
        return
    }

    INFO "Submitting backtest (EURUSD, logistic, 1 month) ..."
    $body = @{
        pair = "EURUSD"
        models = @("logistic")
        months = 1
        trading_costs = $true
    } | ConvertTo-Json

    $resp = Invoke-RestMethod -Uri "http://localhost:${API_PORT}/api/v1/backtest" `
        -Method POST -Body $body -ContentType "application/json"
    
    $jobId = $resp.job_id
    C "  Job ID : " Yellow; Write-Host $jobId
    Write-Host ""

    INFO "Polling status (30s timeout) ..."
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Milliseconds 500
        $status = Invoke-RestMethod -Uri "http://localhost:${API_PORT}/api/v1/backtest/${jobId}" `
            -TimeoutSec 3 -ErrorAction SilentlyContinue
        $s = $status.status
        $p = $status.progress
        C "  [$($i)s] " DarkGray; Write-Host "status=$s"
        
        if ($s -eq "completed") {
            OK "Backtest completed!"
            $results = Invoke-RestMethod -Uri "http://localhost:${API_PORT}/api/v1/backtest/${jobId}/results" -TimeoutSec 3
            Write-Host ""
            foreach ($m in $results.metrics) {
                C "  $($m.model) : " Cyan
                Write-Host "sharpe=$($m.sharpe)  return=$($m.total_return)  trades=$($m.total_trades)  win_rate=$($m.win_rate)"
            }
            return
        }
        if ($s -eq "failed") {
            FAIL "Backtest failed: $($status.error)"
            return
        }
    }
    WARN "Timed out waiting for result. Worker may not be running."
    INFO "Check: .\start.ps1 logs"
}

# ─── INTERACTIVE MENU ──────────────────────────────────────────────
function Show-Menu {
    Clear-Host
    Write-Host ""
    C "  ╔══════════════════════════════════════════════╗`n" Cyan
    C "  ║     FX ML Backtester — Launcher              ║`n" Cyan
    C "  ╚══════════════════════════════════════════════╝`n" Cyan
    Write-Host ""

    # Quick status
    try {
        $r = Invoke-WebRequest -Uri $API_URL -TimeoutSec 1 -UseBasicParsing
        C "  ● API up" Green
    } catch {
        C "  ○ API down" Red
    }
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:${FE_PORT}" -TimeoutSec 1 -UseBasicParsing
        C "  ● Frontend up" Green
    } catch {
        C "  ○ Frontend down" Red
    }
    Write-Host "`n`n"

    Write-Host "  1. Start Docker stack    (recommended)" -ForegroundColor White
    Write-Host "  2. Start Native stack    (venv + local processes)"
    Write-Host "  3. Stop all services"
    Write-Host "  4. Show status"
    Write-Host "  5. Show live logs"
    Write-Host "  6. Run quick backtest test"
    Write-Host "  7. Open Frontend         (http://localhost:${FE_PORT})"
    Write-Host "  8. Open API docs         (http://localhost:${API_PORT}/docs)"
    Write-Host "  Q. Quit"
    Write-Host ""
    $choice = Read-Host "  Choose"

    switch ($choice) {
        "1" { Start-DockerStack; Read-Host "`n  Press Enter to return"; Show-Menu }
        "2" { Start-NativeStack; Read-Host "`n  Press Enter to return"; Show-Menu }
        "3" { Stop-All; Read-Host "`n  Press Enter to return"; Show-Menu }
        "4" { Show-Status; Read-Host "`n  Press Enter to return"; Show-Menu }
        "5" { Show-Logs; Show-Menu }
        "6" { Run-QuickTest; Read-Host "`n  Press Enter to return"; Show-Menu }
        "7" { Start-Process "http://localhost:${FE_PORT}"; Show-Menu }
        "8" { Start-Process "http://localhost:${API_PORT}/docs"; Show-Menu }
        "q" { return }
        default { Show-Menu }
    }
}

# ─── ENTRY POINT ───────────────────────────────────────────────────
Set-Location $PROJECT

switch ($Action) {
    "docker"  { Start-DockerStack }
    "native"  { Start-NativeStack }
    "stop"    { Stop-All }
    "status"  { Show-Status }
    "logs"    { Show-Logs }
    "test"    { Run-QuickTest }
    default   { Show-Menu }
}
