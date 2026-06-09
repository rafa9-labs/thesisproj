import os
import sys
import time
import math


class HPOProgress:
    """Structured progress display for Optuna HPO + WFO backtests.

    Collapses ~550 lines of per-trial noise into ~10 structured lines
    with a live-updating progress bar (``\\r`` carriage return).

    *Per-trial* config dumps, debug gates, and verbose fold tables are
    suppressed unless ``KODAQUANT_VERBOSE=1`` is set.
    """

    def __init__(self):
        self._start = 0.0
        self._last_trial = -1
        self._header_drawn = False
        self._n_total = 0
        self.verbose = bool(int(os.getenv("KODAQUANT_VERBOSE", "0")))
        self._is_tty = sys.stdout.isatty()
        self._enabled = bool(int(os.getenv("KODAQUANT_PROGRESS", "1")))

    def draw_header(self, model, pair, timeframe, date_range, cv_info=""):
        if not self._enabled:
            return
        print("─" * 60)
        print(f" KodaQuant  │  {pair} {timeframe}  │  {model}  │  {date_range[0]}→{date_range[1]}")
        if cv_info:
            print(f" CV: {cv_info}")
        print("─" * 60)
        self._start = time.time()
        self._header_drawn = True

    def set_n_trials(self, n):
        self._n_total = n
        if self._start <= 0:
            self._start = time.time()

    def update_trial(self, trial_num, best_val, fold_srs=None, cv_result=None):
        if not self._enabled:
            return
        elapsed = time.time() - self._start
        n = self._n_total
        bar = self._bar(trial_num, n)
        avg_str = f"{(elapsed / max(1, trial_num)):.1f}s/t" if trial_num > 0 else "?"
        if self._is_tty:
            sys.stdout.write(
                f"\r HPO {bar}  {trial_num}/{n}  │  Best SR: {best_val:.2f}  │  "
                f"⌀ {avg_str}  │  ETA {self._eta(trial_num, n, elapsed)}"
            )
        if fold_srs is not None and trial_num != self._last_trial:
            self._last_trial = trial_num
            grid = "  ".join(self._fmt_fold(sr) for sr in fold_srs)
            sys.stdout.write(f"\n  Trial #{trial_num} folds: {grid}")
            if cv_result:
                brier = cv_result.get("brier", float("nan"))
                brier_str = f"{brier:.3f}" if math.isfinite(brier) else "N/A"
                sys.stdout.write(
                    f"\n  CV result: SR={cv_result.get('sr', float('nan')):.2f}"
                    f" ±{cv_result.get('sr_std', float('nan')):.2f}"
                    f"  │  Cal={brier_str}"
                    f"  │  Cov={cv_result.get('coverage', 0):.0f}%"
                )
        sys.stdout.flush()

    def draw_wfo_progress(self, fold, total, elapsed_s, best_sr=None):
        if not self._enabled:
            return
        if not self._is_tty:
            return
        bar = self._bar(fold, total)
        best = f"  │  Best SR: {best_sr:.2f}" if best_sr is not None else ""
        avg_str = f"{elapsed_s / max(1, fold):.1f}s/fold" if fold > 0 else "?"
        sys.stdout.write(f"\r WFO {bar}  {fold}/{total} folds  │  ⌀ {avg_str}{best}")
        sys.stdout.flush()

    def draw_final(self, elapsed, trades, sr, sharpe, dd):
        if not self._enabled:
            return
        trades_str = f"{int(trades)}" if (trades is not None and trades == trades) else "N/A"
        print(f"\n ✓ Complete  │  Total: {elapsed:.0f}s  │  Trades: {trades_str}  │  SR={sr:.2f}")
        print(f"   Sharpe={sharpe:.2f}  │  DD={dd:.1f}%")

    def _fmt_fold(self, sr):
        if sr is not None and math.isfinite(sr):
            return f"✓ {sr:.2f}"
        return "✗ PRUNE"

    def _bar(self, n, total, width=16):
        if total <= 0:
            return "░" * width
        filled = int(width * n / total)
        return "▓" * filled + "░" * (width - filled)

    @staticmethod
    def _eta(n, total, elapsed):
        if n <= 0 or total <= 0:
            return "?"
        if n <= 1:
            return "?m"
        remaining = (elapsed / n) * (total - n)
        if remaining < 60:
            return f"{remaining:.0f}s"
        if remaining < 3600:
            return f"{remaining/60:.1f}m"
        return f"{remaining/3600:.1f}h"
