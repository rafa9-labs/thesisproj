"""Thread-safe per-job log ring buffer for Full Cycle progress visibility."""
import json
import os
import threading
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional


class LogEntry:
    __slots__ = ("index", "timestamp", "level", "message", "phase", "phase_number",
                 "phase_progress", "category", "metrics")

    def __init__(self, index: int, timestamp: str, level: str, message: str,
                 phase: str = "", phase_number: int = 0, phase_progress: str = "",
                 category: str = "", metrics: Optional[Dict] = None):
        self.index = index
        self.timestamp = timestamp
        self.level = level
        self.message = message
        self.phase = phase
        self.phase_number = phase_number
        self.phase_progress = phase_progress
        self.category = category
        self.metrics = metrics

    def to_dict(self) -> dict:
        d = {"index": self.index, "timestamp": self.timestamp,
             "level": self.level, "message": self.message}
        if self.phase:
            d["phase"] = self.phase
        if self.phase_number:
            d["phase_number"] = self.phase_number
        if self.phase_progress:
            d["phase_progress"] = self.phase_progress
        if self.category:
            d["category"] = self.category
        if self.metrics:
            d["metrics"] = self.metrics
        return d


PHASE_LABELS = {
    1: "Phase 1: Feature Sweep",
    2: "Phase 2: HPO Tuning",
    3: "Phase 3: Committee Assembly",
    4: "Phase 4: Validation",
    5: "Phase 5: Factory Optimization",
}


class LogBuffer:
    """Per-job ring buffer. Thread-safe. Max 20000 lines per job."""

    _lock: threading.RLock
    _buffers: Dict[str, deque]
    _counters: Dict[str, int]
    _log_files: Dict[str, Optional[str]]
    _truncated: Dict[str, bool]
    _MAX = 20000

    def __init__(self):
        self._lock = threading.RLock()
        self._buffers = {}
        self._counters = {}
        self._log_files = {}
        self._truncated = {}

    def append(self, job_id: str, level: str, message: str,
               phase: str = "", phase_number: int = 0, phase_progress: str = "",
               category: str = "", metrics: Optional[Dict] = None) -> int:
        ts = datetime.utcnow().strftime("%H:%M:%S")
        with self._lock:
            if job_id not in self._buffers:
                self._buffers[job_id] = deque(maxlen=self._MAX)
                self._counters[job_id] = 0
                self._truncated[job_id] = False
            idx = self._counters[job_id]
            self._counters[job_id] += 1
            entry = LogEntry(idx, ts, level, message,
                             phase=phase, phase_number=phase_number,
                             phase_progress=phase_progress,
                             category=category, metrics=metrics)
            was_full = len(self._buffers[job_id]) == self._MAX
            self._buffers[job_id].append(entry)
            if was_full and not self._truncated[job_id]:
                self._truncated[job_id] = True
                trunc_entry = LogEntry(idx + 1, ts, "warn",
                                       f"[TRUNCATED] Log buffer exceeded {self._MAX} lines -- oldest entries dropped")
                self._buffers[job_id].append(trunc_entry)
                self._counters[job_id] += 1

            log_path = self._log_files.get(job_id)
            if log_path:
                try:
                    entry_dict = entry.to_dict()
                    line = json.dumps(entry_dict, default=str) + "\n"
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(line)
                except Exception:
                    pass

            return idx

    def get_logs(self, job_id: str, since: int = 0) -> List[dict]:
        with self._lock:
            buf = self._buffers.get(job_id)
            if not buf:
                return []
            return [e.to_dict() for e in buf if e.index >= since]

    def clear(self, job_id: str):
        with self._lock:
            self._buffers.pop(job_id, None)
            self._counters.pop(job_id, None)
            self._log_files.pop(job_id, None)
            self._truncated.pop(job_id, None)

    def enable_file_logging(self, job_id: str, log_path: str):
        with self._lock:
            self._log_files[job_id] = log_path


_LOG_BUFFER = LogBuffer()


def _fmt_msg(phase_number: int, message: str) -> str:
    """Prefix message with phase tag if phase_number provided."""
    if phase_number:
        label = PHASE_LABELS.get(phase_number, f"Phase {phase_number}")
        return f"[{label}] {message}"
    return message


def job_log(job_id: str, level: str, message: str,
            phase: str = "", phase_number: int = 0, phase_progress: str = "",
            category: str = "", metrics: Optional[Dict] = None) -> int:
    """Write to both stdout and the in-memory log buffer. Returns entry index."""
    ts = datetime.utcnow().strftime("%H:%M:%S")
    display_msg = _fmt_msg(phase_number, message)
    level_tag = level.upper()
    if level == "metric":
        level_tag = "INFO"
    print(f"[{ts}] [{level_tag}] {display_msg}", flush=True)
    return _LOG_BUFFER.append(job_id, level, message,
                              phase=phase, phase_number=phase_number,
                              phase_progress=phase_progress,
                              category=category, metrics=metrics)


# -- Legacy aliases (backward-compatible) --

def log_info(job_id: str, message: str,
             phase: str = "", phase_number: int = 0, phase_progress: str = "",
             category: str = "", metrics: Optional[Dict] = None) -> int:
    return job_log(job_id, "info", message,
                   phase=phase, phase_number=phase_number,
                   phase_progress=phase_progress, category=category,
                   metrics=metrics)


def log_warn(job_id: str, message: str,
             phase: str = "", phase_number: int = 0, phase_progress: str = "",
             category: str = "", metrics: Optional[Dict] = None) -> int:
    return job_log(job_id, "warn", message,
                   phase=phase, phase_number=phase_number,
                   phase_progress=phase_progress, category=category,
                   metrics=metrics)


def log_error(job_id: str, message: str,
              phase: str = "", phase_number: int = 0, phase_progress: str = "",
              category: str = "", metrics: Optional[Dict] = None) -> int:
    return job_log(job_id, "error", message,
                   phase=phase, phase_number=phase_number,
                   phase_progress=phase_progress, category=category,
                   metrics=metrics)


# -- Structured helpers --

def log_phase_start(job_id: str, phase_number: int, phase_name: str,
                    phase: str = "", phase_progress: str = "",
                    **kw_metrics) -> int:
    """Announce start of a pipeline phase. Renders: === Phase N: Name === """
    label = PHASE_LABELS.get(phase_number, f"Phase {phase_number}")
    return job_log(job_id, "info", f"Phase {phase_number} started: {phase_name}",
                   phase=phase, phase_number=phase_number,
                   phase_progress=phase_progress,
                   category="phase_start", metrics=kw_metrics or None)


def log_phase_complete(job_id: str, phase_number: int, summary: str = "",
                       phase: str = "", **kw_metrics) -> int:
    """Announce completion of a pipeline phase. Renders: Phase N complete: ..."""
    label = PHASE_LABELS.get(phase_number, f"Phase {phase_number}")
    msg = f"Phase {phase_number} complete"
    if summary:
        msg += f": {summary}"
    return job_log(job_id, "info", msg,
                   phase=phase, phase_number=phase_number,
                   category="phase_complete", metrics=kw_metrics or None)


def log_progress(job_id: str, phase_number: int, message: str,
                 current: int = 0, total: int = 0,
                 phase: str = "", **kw_metrics) -> int:
    """Report sub-phase progress. Renders progress fraction if current/total given."""
    progress_str = ""
    if current and total:
        progress_str = f"{current}/{total}"
    return job_log(job_id, "info", message,
                   phase=phase, phase_number=phase_number,
                   phase_progress=progress_str,
                   category="progress", metrics=kw_metrics or None)


def log_metric(job_id: str, metric_name: str, value,
               phase: str = "", phase_number: int = 0) -> int:
    """Log a single numeric metric. Renders: metric_name = value"""
    metrics = {metric_name: value}
    return job_log(job_id, "metric", f"  {metric_name} = {value:.4f}" if isinstance(value, float) else f"  {metric_name} = {value}",
                   phase=phase, phase_number=phase_number,
                   category="metric", metrics=metrics)


# -- Retrieval / maintenance --

def get_job_logs(job_id: str, since: int = 0) -> List[dict]:
    return _LOG_BUFFER.get_logs(job_id, since)


def clear_job_logs(job_id: str):
    _LOG_BUFFER.clear(job_id)


def enable_file_logging(job_id: str, log_dir: str):
    log_path = os.path.join(log_dir, "pipeline.log")
    _LOG_BUFFER.enable_file_logging(job_id, log_path)


def health():
    """Return buffer stats."""
    with _LOG_BUFFER._lock:
        return {k: len(v) for k, v in _LOG_BUFFER._buffers.items()}
