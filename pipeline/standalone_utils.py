"""
Standalone utility functions (not class methods).

Extracted from MLBacktesterNoWFO.py lines 295-504.
"""

from pipeline._imports import *  # noqa: F401,F403

_DATA_CACHE: dict = {}


def clear_data_cache():
    """Clear the module-level CSV data cache to free memory."""
    n = len(_DATA_CACHE)
    _DATA_CACHE.clear()
    return n

try:
    import pyarrow  # noqa: F401
    _CSV_ENGINE = "pyarrow"
except Exception:
    _CSV_ENGINE = "c"

def _norm_class_counts(d: object) -> dict:
    """
    Normalize class-count dict keys to plain ints.

    value_counts() often yields keys like np.int64(-1) or -1.0. If logs later do
    raw.get(-1), they miss and show fake zeros.

    Telemetry-only: does not affect trading logic or metrics.
    """
    out = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        try:
            out[int(k)] = int(v)
        except Exception:
            continue
    return out


def print_block_summary(block_id, calib_info, gate_info, reliability,
                        class_dists, block_stats, fold_label: str = "Mini-Block Fold") -> None:
    """
    Per-fold compact summary. All fields are already precomputed by the caller.
    """
    line = "-" * 70
    log_print(f"\n\n\n{line}\n{fold_label} #{block_id}\n{line}", level="COMPACT")

    try:
        _bars_total = int(calib_info.get("bars_total", calib_info.get("bars", 0)) or 0)
    except Exception:
        _bars_total = int(calib_info.get("bars", 0) or 0)
    try:
        _bars_elig = int(calib_info.get("bars_eligible", calib_info.get("bars", _bars_total)) or 0)
    except Exception:
        _bars_elig = int(calib_info.get("bars", _bars_total) or 0)

    log_print(
        f"Coverage target {calib_info['target']:.2f} | "
        f"conf_thr {calib_info['conf_thr']:.3f} | "
        f"bars total {_bars_total} | eligible {_bars_elig}",
        level="COMPACT",
    )
    log_print(
        "Dynamic alphabetagamma -> "
        f"base={gate_info['base']:.3f} "
        f"alpha={gate_info['alpha']:.3f} "
        f"beta={gate_info['beta']:.3f} "
        f"gamma={gate_info['gamma']:.3f} | "
        f"median_thr={gate_info['median_thr']:.3f}",
        level="COMPACT",
    )
    
    try:
        _rows_total = int(block_stats.get("rows_total", block_stats.get("rows", 0)) or 0)
    except Exception:
        _rows_total = int(block_stats.get("rows", 0) or 0)
    try:
        _rows_elig = int(block_stats.get("rows_eligible", block_stats.get("rows", _rows_total)) or 0)
    except Exception:
        _rows_elig = int(block_stats.get("rows", _rows_total) or 0)

    # ---- Sharpe string computed once (avoid nested f-strings) ----
    _sr_val = block_stats.get("sr", float("nan"))
    try:
        _sr_val = float(_sr_val)
    except Exception:
        _sr_val = float("nan")
    sr_str = "--" if (_sr_val != _sr_val) else f"{_sr_val:+.3f}"

    log_print(
        f"Denoms -> "
        f"val_rows={_rows_total} | "
        f"post_feature_bars_total={int(calib_info.get('bars_total', _rows_total) or _rows_total)} | "
        f"eligible={int(calib_info.get('bars_eligible', _rows_elig) or _rows_elig)} | "
        f"eval_bars={int(block_stats.get('rows', _rows_elig) or _rows_elig)}   "
        f"trades={block_stats['trades']}   "
        f"active_rate={block_stats['ar']:.3f}   "
        f"Sharpe={sr_str}",
        level="COMPACT",
    )

    log_print(
        f"Coverage nudge -> band +/-{gate_info['band']:.2f} "
        f"step {gate_info['step']:.3f}",
        level="COMPACT",
    )
    log_print(
        "Reliability -> "
        f"PSRalpha={reliability['psr_alpha']:.2f} "
        f"cutoff={reliability['cutoff']:.2f} "
        f"min_trades={reliability['min_trades']} "
        f"indep={reliability['min_indep']}",
        level="COMPACT",
    )

    log_print(line, level="COMPACT")
    raw = _norm_class_counts(class_dists.get("raw", {}))
    final = _norm_class_counts(class_dists.get("final", {}))
    
    log_print(
        "Class dist (raw)     "
        f"-1:{raw.get(-1, 0)}   "
        f"0:{raw.get(0, 0)}   "
        f"+1:{raw.get(1, 0)}",
        level="COMPACT",
    )
    log_print(
        "After filter          "
        f"0:{final.get(0, 0)}   "
        f"-1:{final.get(-1, 0)}  "
        f"+1:{final.get(1, 0)}",
        level="COMPACT",
    )

    log_print(line, level="COMPACT")
    sr = block_stats.get("sr", "--")
    if isinstance(sr, (float, int)) and np.isfinite(sr):
        sr_str = f"{float(sr):.3f}"
    else:
        sr_str = "--"

    log_print(
        f"Denoms -> "
        f"val_rows={_rows_total} | "
        f"post_feature_bars_total={int(calib_info.get('bars_total', _rows_total) or _rows_total)} | "
        f"eligible={int(calib_info.get('bars_eligible', _rows_elig) or _rows_elig)} | "
        f"eval_bars={int(block_stats.get('rows', _rows_elig) or _rows_elig)}   "
        f"trades={block_stats['trades']}   "
        f"active_rate={block_stats['ar']:.3f}   "
        f"Sharpe={sr_str}",
        level="COMPACT",
    )
    
    # Optional: show trade-intent precision for this fold (post confidence gating)
    try:
        p_int = block_stats.get("precision_intent", None)
        n_int = block_stats.get("intent_bars", None)
        p_int = float(p_int) if p_int is not None else float("nan")
        n_int = int(n_int) if n_int is not None else 0
        if (n_int > 0) and (p_int == p_int):  # not NaN
            log_print(f"Intent precision p={p_int:.3f} (n={n_int})", level="COMPACT")
    except Exception:
        pass
    log_print(line, level="COMPACT")
    
def print_pruned_block_summary(
    block_id: int,
    reason: str,
    rows: int | None = None,
    trades: int | None = None,
    active_rate: float | None = None,
    sharpe: float | None = None,
    fold_label: str = "Mini-Block Fold",
) -> None:
    """
    Compact summary for Mini-Block folds that were pruned or marked invalid.
    Mirrors the style of `print_block_summary` but focuses on the prune/invalid reason.
    """
    line = "-" * 70

    # Fallbacks for stats
    rows_str = "--" if rows is None else str(int(rows))
    trades_str = "--" if trades is None else str(int(trades))

    if isinstance(active_rate, (float, int)) and np.isfinite(active_rate):
        ar_str = f"{float(active_rate):.3f}"
    else:
        ar_str = "--"

    if isinstance(sharpe, (float, int)) and np.isfinite(sharpe):
        sr_str = f"{float(sharpe):.3f}"
    else:
        sr_str = "--"

    # Keep reason on a single, not-too-long line
    reason_str = str(reason).replace("\n", " ").strip()
    if len(reason_str) > 200:
        reason_str = reason_str[:197] + "..."

    log_print(
        f"\n\n\n{line}\n{fold_label} #{block_id} [PRUNED / INVALID]\n{line}",
        level="COMPACT",
    )
    log_print(f"Reason: {reason_str}", level="COMPACT")
    log_print(
        f"rows={rows_str}   trades={trades_str}   "
        f"active_rate={ar_str}   Sharpe={sr_str}",
        level="COMPACT",
    )
    log_print(line, level="COMPACT")



def _load_csv_cached(path, parse_dates=None, index_col=None):
    key = (path, tuple(parse_dates or []), index_col)
    if key not in _DATA_CACHE:
        df = pd.read_csv(path, parse_dates=parse_dates, engine=_CSV_ENGINE)
        if index_col:
            df.set_index(index_col, inplace=True)
        _DATA_CACHE[key] = df
    return _DATA_CACHE[key].copy()

# Silence pandas/ta deprecation noise from PSAR internals in ta.trend
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="ta.trend",
)

