"""
Active resource monitor: daemon thread that samples CPU/GPU utilization
and enforces the 67% ceiling via lock-free throttle signals.

Architecture:
  - Monitor thread: 500ms sample interval, EMA(alpha=0.3) over 10 samples
  - Throttle engine: escalation ladder L1(sleep) -> L2(batch) -> L3(n_jobs) -> L4(serial)
  - Hysteresis: release only when EMA < 50% for 3 consecutive samples
  - Idle bypass: cpu < 30% for 3 samples -> instant release

Usage:
  budget = get_resource_budget()
  with ResourceMonitor(budget) as signal:
      for epoch in range(n):
          if signal.level >= 1:
              time.sleep(signal.delay)
          if signal.level >= 3:
              n_jobs = max(1, n_jobs // 2)
          train_one_epoch()
"""
import collections
import os
import threading
import time
from typing import Optional

from pipeline.resource_budget import ResourceBudget

_current_signal: Optional["ThrottleSignal"] = None


def get_throttle_signal() -> Optional["ThrottleSignal"]:
    return _current_signal


class ThrottleSignal:
    __slots__ = ("level", "delay", "batch_reduce", "n_jobs_half", "serial")

    def __init__(self):
        self.level: int = 0
        self.delay: float = 0.0
        self.batch_reduce: bool = False
        self.n_jobs_half: bool = False
        self.serial: bool = False

    def reset(self):
        self.level = 0
        self.delay = 0.0
        self.batch_reduce = False
        self.n_jobs_half = False
        self.serial = False


class ResourceMonitor:
    def __init__(self, budget: Optional[ResourceBudget] = None,
                 sample_interval: float = 0.5,
                 window_size: int = 10,
                 ema_alpha: float = 0.3,
                 ceiling: float = 67.0,
                 release_threshold: float = 50.0,
                 idle_threshold: float = 30.0):

        self.budget = budget
        self.sample_interval = sample_interval
        self.window_size = int(window_size)
        self.ema_alpha = float(ema_alpha)
        self.ceiling = float(ceiling)
        self.release_threshold = float(release_threshold)
        self.idle_threshold = float(idle_threshold)

        self.signal = ThrottleSignal()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._samples = collections.deque(maxlen=window_size)
        self._ema: float = 0.0
        self._consecutive_below_release = 0
        self._consecutive_idle = 0
        self._nvidia_handle = None
        self._psutil_available = False

    def __enter__(self):
        global _current_signal
        self.start()
        _current_signal = self.signal
        return self.signal

    def __exit__(self, *args):
        global _current_signal
        self.stop()
        _current_signal = None

    def start(self):
        if self._thread is not None:
            return
        try:
            import psutil
            self._psutil_available = True
        except Exception:
            pass

        if self.budget and self.budget.gpu_available:
            try:
                import pynvml
                pynvml.nvmlInit()
                self._nvidia_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                pass

        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._nvidia_handle is not None:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvidia_handle = None
        self.signal.reset()

    def _sample_cpu(self) -> float:
        if not self._psutil_available:
            return 0.0
        try:
            import psutil
            return float(psutil.cpu_percent(interval=None))
        except Exception:
            return 0.0

    def _sample_gpu(self) -> float:
        if self._nvidia_handle is None:
            return 0.0
        try:
            import pynvml
            util = pynvml.nvmlDeviceGetUtilizationRates(self._nvidia_handle)
            return float(getattr(util, "gpu", 0))
        except Exception:
            return 0.0

    def _run(self):
        while not self._stop.is_set():
            cpu = self._sample_cpu()
            gpu = self._sample_gpu()
            peak = max(cpu, gpu) if self._nvidia_handle is not None else cpu

            self._samples.append(peak)

            if self._samples:
                self._ema = self._samples[0]
                for v in list(self._samples)[1:]:
                    self._ema = self.ema_alpha * v + (1 - self.ema_alpha) * self._ema

            self._evaluate_throttle(peak)

            self._stop.wait(self.sample_interval)

    def _evaluate_throttle(self, current_sample: float):
        if current_sample < self.idle_threshold:
            self._consecutive_idle += 1
        else:
            self._consecutive_idle = 0

        if self._consecutive_idle >= 3:
            self.signal.reset()
            self._consecutive_below_release = 0
            return

        if self._ema < self.release_threshold:
            self._consecutive_below_release += 1
        else:
            self._consecutive_below_release = 0

        if self._consecutive_below_release >= 3 and self.signal.level > 0:
            self.signal.level = max(0, self.signal.level - 1)
            self._apply_release_step()
            self._consecutive_below_release = 0
            return

        if self._ema > 75.0 and self.signal.level < 4:
            self.signal.level = 4
            self._apply_throttle_level(4)
        elif self._ema > self.ceiling and self.signal.level < 4:
            new_level = min(4, self.signal.level + 1)
            if new_level > self.signal.level:
                self.signal.level = new_level
                self._apply_throttle_level(new_level)

    def _apply_throttle_level(self, level: int):
        self.signal.delay = 0.0
        self.signal.batch_reduce = False
        self.signal.n_jobs_half = False
        self.signal.serial = False

        if level >= 1:
            self.signal.delay = 0.05
        if level >= 2:
            self.signal.batch_reduce = True
            self.signal.delay = 0.10
        if level >= 3:
            self.signal.n_jobs_half = True
            self.signal.delay = 0.15
        if level >= 4:
            self.signal.serial = True
            self.signal.delay = 0.25

    def _apply_release_step(self):
        prev = self.signal.level
        self.signal.delay = 0.0
        self.signal.batch_reduce = False
        self.signal.n_jobs_half = False
        self.signal.serial = False
        if prev >= 2:
            self.signal.delay = 0.05
        if prev >= 3:
            self.signal.batch_reduce = True
            self.signal.delay = 0.10
