"""Auto-extracted mixin -- see composed.py for the full MLBacktester."""
from config import PIPELINE_CONSTANTS as _PC
from pipeline._imports import *  # noqa: F401,F403


class DeepMixin:
    """
    keras fit, calibration, subprocess

    Auto-extracted from MLBacktesterNoWFO.py lines 3366-4225.
    """
    def _fit_keras_with_cv_controls(
        self,
        model,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        base_epochs=20,
        base_batch=128,
        verbose=0,
        validation_split_if_needed=0.10,
        extra_callbacks=None,
    ):
        """
        Centralizes keras.fit so deep models run fast when inside Optuna CV.
        - Uses model.early_stop_callback if it exists.
        - If no explicit (X_val, y_val) but EarlyStopping exists, uses a **time-ordered tail holdout**
        (never Keras validation_split with shuffling).
        - During CV (_in_optuna_cv=True), caps epochs/batch from features_config.
        """

        # Defaults
        epochs = int(base_epochs)
        batch  = int(base_batch)
        callbacks = []

        # Respect EarlyStopping created by builders
        early_cb = getattr(model, "early_stop_callback", None)
        if early_cb is not None:
            callbacks.append(early_cb)

        # Any extra callbacks (e.g., time-limit)
        if extra_callbacks:
            callbacks.extend(extra_callbacks if isinstance(extra_callbacks, (list, tuple)) else [extra_callbacks])

        # CV caps (multi-fidelity proxy during Optuna CV)
        if getattr(self, "_in_optuna_cv", False):
            cfg = getattr(self, "features_config", {}) or {}
            # We optionally tag models with a short name ("cnn", "lstm", "transformer")
            # so that different deep families can have different CV budgets.
            #
            # Research basis:
            # - Prechelt (1998, Neural Networks 11(4)) shows that early stopping
            #   + limited epochs can reduce deep training time ~4x with minimal
            #   generalisation loss.
            # - Wu et al. (2020, AISTATS) and Won et al. (2025, ICT Express) argue
            #   for multi-fidelity HPO: use cheaper fidelities (fewer epochs/samples)
            #   during tuning and reserve full budgets for final refit.
            model_tag = getattr(model, "_mlb_model_tag", None)

            # Requested (from model config / Optuna) vs CV caps (compute control)
            req_epochs = int(epochs)
            req_batch  = int(batch)

            # Global CV caps (fallback)
            epochs_cap_default = int(cfg.get("deep_cv_max_epochs", req_epochs))
            batch_cap_default  = int(cfg.get("deep_cv_batch_size", req_batch))

            if model_tag:
                # Optional per-model overrides, e.g. cnn_cv_max_epochs, lstm_cv_max_epochs, ...
                epochs_cap = int(cfg.get(f"{model_tag}_cv_max_epochs", epochs_cap_default))
                batch_cap  = int(cfg.get(f"{model_tag}_cv_batch_size",  batch_cap_default))
            else:
                epochs_cap = epochs_cap_default
                batch_cap  = batch_cap_default

            # Apply caps (so tuning matters up to the cap)
            epochs = min(req_epochs, epochs_cap)
            batch  = min(req_batch,  batch_cap)
            
            print(
                f"[DEEP-CV] model={model_tag or 'generic'} | "
                f"epochs={epochs} (req={req_epochs}, cap={epochs_cap}), "
                f"batch_size={batch} (req={req_batch}, cap={batch_cap}) "
                f"(patience={getattr(early_cb, 'patience', 'NA')})"
            )

        # --- Time-ordered validation (tail split) ---
        use_tail_val = (X_val is None and y_val is None and early_cb is not None
                        and validation_split_if_needed and validation_split_if_needed > 0.0)

        if use_tail_val:
            n = int(getattr(X_train, "shape", [0])[0])
            n_val = max(1, int(round(n * float(validation_split_if_needed))))
            n_val = min(max(1, n_val), n - 1) if n > 1 else 1
            split = n - n_val

            X_tr, y_tr = X_train[:split], y_train[:split]
            X_v,  y_v  = X_train[split:], y_train[split:]
        else:
            X_tr, y_tr = X_train, y_train
            X_v,  y_v  = X_val,   y_val

        # --- Optional class weights (helps LSTM/Transformer with skewed labels) ---
        class_weight = None
        try:
            cfg = getattr(self, "features_config", {}) or {}
            use_cw = bool(cfg.get("deep_use_class_weight", False))
            # You can also enable per-model: lstm_use_class_weight / transformer_use_class_weight
            if not use_cw:
                use_cw = bool(cfg.get("lstm_use_class_weight", False) or cfg.get("transformer_use_class_weight", False))
            if use_cw:
                y_for_cw = np.ravel(y_tr if use_tail_val else y_train)
                classes = np.unique(y_for_cw)
                if classes.size >= 2:
                    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_for_cw)
                    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
                    print(f"[CLASS-WEIGHT] {class_weight}")
        except Exception as _e:
            print(f"[WARN] class_weight computation skipped: {_e}")

        fit_kwargs = dict(
            x=X_tr, y=y_tr,
            epochs=epochs,
            batch_size=batch,
            verbose=verbose,
            shuffle=False,  # <-- time-series safe
        )


        if X_v is not None and y_v is not None:
            fit_kwargs.update({"validation_data": (X_v, y_v)})

        if class_weight is not None:
            fit_kwargs.update({"class_weight": class_weight})

        history = model.fit(callbacks=callbacks, **fit_kwargs)
        return history
    
    def _debug_dump_first_bars(self, index, raw_classes=None, max_conf=None, final_preds=None, n=10, label=""):
        """
        Print a tiny table with the first n window-end decisions aligned to eval index.
        Only runs if self._dbg_first_bars is True (set in real_trading_simulation).
        """
        try:
            if (not getattr(self, "_dbg_first_bars", False)) or getattr(self, "_in_cv", False):
                return
            import numpy as _np
            idx = pd.Index(index)
            if len(idx) == 0:
                print("[INFO] [debug-dump] empty index.")
                return
            n = int(min(n, len(idx)))

            def _as_arr(a, length, fill=_np.nan):
                if a is None:
                    return _np.full(length, fill, dtype=float)
                arr = _np.asarray(a)
                return arr[:length] if len(arr) >= length else _np.pad(arr, (0, length - len(arr)), constant_values=fill)

            # make equal length slices
            m = n
            rc  = _as_arr(raw_classes, m, _np.nan)
            mc  = _as_arr(max_conf,   m, _np.nan)
            fp  = _as_arr(final_preds, m, _np.nan)

            print(f"\n[SEARCH] First {m} window-end decisions{(' -- '+str(label)) if label else ''}")
            print("timestamp                 | raw_cls | max_conf | final_pred")
            for i in range(m):
                ts = str(idx[i])
                rc_i = "NA" if not _np.isfinite(rc[i]) else f"{int(rc[i])}"
                mc_i = "NA" if not _np.isfinite(mc[i]) else f"{float(mc[i]):.3f}"
                fp_i = "NA" if not _np.isfinite(fp[i]) else f"{int(fp[i])}"
                print(f"{ts:>23} | {rc_i:>7} | {mc_i:>8} | {fp_i:>10}")
        except Exception as e:
            print(f"[debug-dump] skipped: {e}")
                               
                               
    def run_pbo_mcs_analysis(self):
        """
        Post-hoc analysis of WFO/WFS results using a CSCV-style PBO estimate
        and a simple Model Confidence Set (MCS) approximation.

        This function is **read-only** with respect to the main pipeline:
        it only consumes self._wfo_monthly_records and does not modify
        models, thresholds, or trading results.

        Returns
        -------
        dict or None
            Dictionary with keys:
            - 'matrix': DataFrame (index=test_end, columns=strategy_id)
            - 'mean_return': per-strategy mean monthly return
            - 'std_return': per-strategy std of monthly return
            - 'sharpe_like': per-strategy mean/std
            - 'pbo': estimated Probability of Backtest Overfitting (0-1 or NaN)
            - 'mcs_strategies': list of strategy_ids in a simple MCS proxy
        """
        import numpy as _np
        import pandas as _pd

        recs = getattr(self, "_wfo_monthly_records", None)
        if not recs:
            log_print("[PBO/MCS] No monthly records available; skipping analysis.", level="COMPACT")
            return None


        df = _pd.DataFrame(recs)
        if df.empty:
            print("[PBO/MCS] Monthly records DataFrame empty; skipping analysis.")
            return None
        
        # Ensure datetime ordering by month end
        df["test_end"] = _pd.to_datetime(df["test_end"])
        df = df.dropna(subset=["test_end", "strategy_id", "strategy_return"])
        df = df.sort_values(["test_end", "strategy_id"])

        if df.empty:
            log_print(
                "[PBO/MCS] No valid (test_end, strategy_id, strategy_return) rows; skipping.",
                level="COMPACT",
            )
            return None

        # Build strategy x month matrix of monthly returns
        mat = df.pivot_table(
            index="test_end",
            columns="strategy_id",
            values="strategy_return",
            aggfunc="first",
        ).sort_index()

        # Drop strategies with too few months
        min_months = 6
        valid_cols = [c for c in mat.columns if mat[c].notna().sum() >= min_months]
        mat = mat[valid_cols]

        if mat.shape[1] < 2:
            log_print(
                "[PBO/MCS] Need >=2 strategies with sufficient history for PBO/MCS; skipping.",
                level="COMPACT",
            )
            return None


        # Basic per-strategy summary
        mean_ret = mat.mean(axis=0)
        std_ret = mat.std(axis=0, ddof=1)
        sharpe_like = mean_ret / std_ret.replace(0.0, _np.nan)

        # --- CSCV-style PBO estimate (Bailey et al.) ---
        R = mat.to_numpy(dtype=float)
        T, S = R.shape

        n_splits = min(200, max(20, T * 10))  # scale with #months, but cap for runtime
        omegas = []

        rng = _np.random.default_rng(seed=42)  # deterministic for reproducibility
        for _ in range(n_splits):
            # Random train/test split over months (roughly half-half)
            if T < 4:
                break
            train_idx = _np.sort(rng.choice(T, size=max(2, T // 2), replace=False))
            test_mask = _np.ones(T, dtype=bool)
            test_mask[train_idx] = False
            if test_mask.sum() < 2:
                continue

            train_mask = _np.zeros(T, dtype=bool)
            train_mask[train_idx] = True

            R_train = R[train_mask]
            R_test = R[test_mask]

            # In-sample mean per strategy; pick best
            is_mean = _np.nanmean(R_train, axis=0)
            if _np.all(~_np.isfinite(is_mean)):
                continue
            best_idx = int(_np.nanargmax(is_mean))

            # Out-of-sample performance for all strategies
            oos_mean = _np.nanmean(R_test, axis=0)
            if not _np.isfinite(oos_mean[best_idx]):
                continue

            # Empirical OOS quantile of the chosen strategy
            # rank 1=worst, S=best  -> quantile in (0,1)
            ranks = _np.argsort(_np.argsort(oos_mean))  # 0-based rank
            u = (ranks[best_idx] + 1) / float(S + 1e-9)
            if u <= 0.0 or u >= 1.0 or not _np.isfinite(u):
                continue

            # Overfitting statistic omega = logit(u)
            omega = _np.log(u / (1.0 - u))
            omegas.append(omega)

        if omegas:
            omegas = _np.asarray(omegas, dtype=float)
            pbo = float(_np.mean(omegas <= 0.0))
        else:
            pbo = float("nan")

        # --- Simple MCS proxy (NOT full Hansen-Lunde-Nason MCS) ---
        # Keep strategies whose mean return is within 1 std-error of the best.
        T_eff = float(mat.shape[0])
        best_mean = float(mean_ret.max())
        best_se = float(std_ret[mean_ret.idxmax()] / _np.sqrt(max(T_eff, 1.0)))
        # Allow a small band around the best
        band = best_mean - best_se
        mcs_strategies = [sid for sid, mu in mean_ret.items() if mu >= band]

        summary = {
            "matrix": mat,
            "mean_return": mean_ret,
            "std_return": std_ret,
            "sharpe_like": sharpe_like,
            "pbo": pbo,
            "mcs_strategies": mcs_strategies,
        }

        try:
            log_print("\n[PBO/MCS] Per-strategy summary (mean, std, sharpe-like):", level="COMPACT")
            log_print(
                _pd.DataFrame({
                    "mean_return": mean_ret,
                    "std_return": std_ret,
                    "sharpe_like": sharpe_like,
                }).to_string(),
                level="COMPACT",
            )
            log_print(
                f"[PBO/MCS] Estimated PBO = {pbo:.3f} based on {len(omegas) if omegas else 0} splits.",
                level="COMPACT",
            )
            log_print(
                f"[PBO/MCS] MCS proxy strategies: {mcs_strategies}",
                level="COMPACT",
            )
        except Exception:
            pass


        return summary

    def _is_deep_model_type(self, model_type: str) -> bool:
        """Return True if this model family uses TF/Keras in this engine."""
        mt = str(model_type or "").lower().strip()
        if mt in {"cnn", "lstm", "transformer", "gru", "gru_lstm"}:
            return True
        if mt in {"ensemble_cnn_lstm_xgboost", "ensemble_adaptive_regime"}:
            return True
        return False

    def _maybe_configure_tf_runtime_once(self, model_type: str) -> None:
        """Configure TF runtime knobs only when a deep model is actually used (once per instance)."""
        if not self._is_deep_model_type(model_type):
            return
        if getattr(self, "_tf_runtime_configured", False):
            return

        try:
            import os
            _threads = int(os.getenv("BLAS_THREADS_PER_TRIAL", os.getenv("MLB_THREADS", str(max(1, (os.cpu_count() or 8) - 2)))))
            os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
            os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(_threads))
            os.environ.setdefault("TF_NUM_INTEROP_THREADS", str(max(2, min(4, _threads // 4))))
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ.setdefault(k, str(_threads))
        except Exception:
            pass

        try:
            import tensorflow as tf
            try:
                for _gpu in tf.config.list_physical_devices("GPU"):
                    try:
                        tf.config.experimental.set_memory_growth(_gpu, True)
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                tf.config.set_soft_device_placement(True)
            except Exception:
                pass
            try:
                gpus = tf.config.list_physical_devices("GPU")
                if gpus:
                    print(f"[deep_mixin] GPU detected: {len(gpus)} device(s) -- {gpus[0].name}")
                else:
                    print("[deep_mixin] No GPU detected -- using CPU")
            except Exception:
                pass
        except Exception:
            pass

        self._tf_runtime_configured = True

    def _get_cost_arrays_aligned(self, cost_df, index):
        """Align cost columns once and return (returns_series, spread_np, slippage_np)."""
        import numpy as _np
        import pandas as _pd

        if cost_df is None or index is None:
            return _pd.Series([], dtype=float), _np.asarray([], dtype=_np.float32), _np.asarray([], dtype=_np.float32)

        aligned = cost_df.reindex(index)

        # returns: keep as Series for realized_vol() (pandas rolling)
        try:
            rets = _pd.to_numeric(aligned.get("returns", 0.0), errors="coerce").astype(float)
        except Exception:
            rets = _pd.Series(0.0, index=index, dtype=float)

        def _col_to_f32(name: str):
            try:
                s = _pd.to_numeric(aligned.get(name, 0.0), errors="coerce")
                return s.to_numpy(dtype=_np.float32, copy=False)
            except Exception:
                return _np.zeros(len(index), dtype=_np.float32)

        sprd = _col_to_f32("spread")
        slip = _col_to_f32("slippage_bps")
        return rets, sprd, slip


    def _ensure_cost_columns(self, df, config):
        """
        Attach real per-bar cost columns (no synthetic means).
        - 'spread' is copied from self.data (if present).
        - 'slippage_bps' is volatility-aware: base->high on high-vol bars.
        """
        import numpy as _np
        import pandas as pd
        from utilsNoWFO import realized_vol

        # 0) Guard
        try:
            use_costs = bool(config.get("eval_use_trading_costs", getattr(self, "trading_costs", True)))
        except Exception:
            use_costs = bool(getattr(self, "trading_costs", True))
        if (not use_costs) or df is None or len(df) == 0:
            return df

        df = df.copy()

        # 1) Spread: copy the real per-bar series from self.data if missing
        if "spread" not in df.columns:
            try:
                if hasattr(self, "data") and isinstance(self.data, pd.DataFrame) and "spread" in self.data.columns:
                    df["spread"] = self.data["spread"].reindex(df.index)
            except Exception:
                pass  # leave missing; metrics fn tolerates it
        else:
            try:
                _s = pd.to_numeric(df["spread"], errors="coerce").astype(float)
                if hasattr(self, "data") and isinstance(self.data, pd.DataFrame) and "spread" in self.data.columns:
                    _s = _s.combine_first(self.data["spread"].reindex(df.index))
                df["spread"] = _s.fillna(0.0)
            except Exception:
                df["spread"] = 0.0
                
                
        # ANCHOR: # 2) Vol-aware slippage_bps per bar (0.20 base -> 0.30 on high vol)
        # Price series is needed to normalize spread (price units) into fractional drag.
        # compute_full_evaluation_metrics() uses 'price' or 'mid_close' if available.
        if ("price" not in df.columns) and ("mid_close" not in df.columns):
            try:
                if hasattr(self, "data") and isinstance(self.data, pd.DataFrame):
                    if "price" in self.data.columns:
                        df["price"] = pd.to_numeric(self.data["price"].reindex(df.index), errors="coerce").astype(float)
                    elif "mid_close" in self.data.columns:
                        df["mid_close"] = pd.to_numeric(self.data["mid_close"].reindex(df.index), errors="coerce").astype(float)
            except Exception:
                pass

        # --- Safe config access (supports config=None + df.attrs fallback) ---
        cfg = config if isinstance(config, dict) else {}
        try:
            cfg_from_attrs = dict(df.attrs.get("features_config", {}) or {})
        except Exception:
            cfg_from_attrs = {}
            
            
        # Also allow fallback to self.features_config (train-anchored values persisted there)
        try:
            cfg_from_self = dict(getattr(self, "features_config", {}) or {})
        except Exception:
            cfg_from_self = {}

        def _get_cfg(k, default=None):
            # explicit config wins, then df.attrs, then self.features_config
            return cfg.get(k, cfg_from_attrs.get(k, cfg_from_self.get(k, default)))

        # 2) Vol-aware slippage_bps per bar (base -> high on high vol; MED fallback if thr missing)
        if "slippage_bps" not in df.columns:
            base = float(_get_cfg("eval_slip_bps_lo", _get_cfg("cv_slippage_bps_base", 0.20)))
            high = float(_get_cfg("eval_slip_bps_hi", _get_cfg("cv_slippage_bps_high", 0.30)))
            # Optional middle regime (used as safe fallback if high-vol threshold is missing)
            med  = float(_get_cfg("eval_slip_bps_med", _get_cfg("cv_slippage_bps_med", (base + high) / 2.0)))
            vol_w = int(_get_cfg("vol_window_bars", _PC["vol_window_bars"]))
            qhi   = float(_get_cfg("high_vol_q", _PC["high_vol_q"]))

            # Optional override: caller may provide a precomputed (train-anchored) threshold.
            # If not provided, DO NOT derive a threshold from the eval df (leakage).
            thr_override = _get_cfg("high_vol_thr", None)
            try:
                thr_override = float(thr_override) if thr_override is not None else None
            except Exception:
                thr_override = None

            # Last-chance fallback: pull a train-anchored threshold cached on the instance.
            # This prevents the LeakageGuard path when callers forget to pass high_vol_thr.
            if thr_override is None:
                try:
                    _thr_last = getattr(self, "_last_high_vol_thr_train", None)
                    _thr_last = float(_thr_last) if _thr_last is not None else None
                    if _thr_last is not None and _np.isfinite(_thr_last):
                        thr_override = _thr_last
                except Exception:
                    pass
            if thr_override is None:
                try:
                    _thr_fc = (getattr(self, "features_config", {}) or {}).get("high_vol_thr", None)
                    _thr_fc = float(_thr_fc) if _thr_fc is not None else None
                    if _thr_fc is not None and _np.isfinite(_thr_fc):
                        thr_override = _thr_fc
                except Exception:
                    pass

            try:
                if "returns" in df.columns:
                    rv = realized_vol(df["returns"].astype(float), window=vol_w)

                    if thr_override is not None and _np.isfinite(thr_override):
                        thr = thr_override
                        mask = (rv >= thr)
                        if getattr(self, "_is_debug", lambda: False)():
                            print(f"[Costs] Using provided high_vol_thr={thr:.6g} (q={qhi:.2f}, vol_w={vol_w})")

                        # Normal regime-aware assignment
                        df["slippage_bps"] = _np.where(mask, high, base).astype(float)

                    else:
                        # Leakage guard: no train-anchored threshold was passed.
                        # IMPORTANT: fallback to MED slippage (do NOT punish all bars with the max regime).
                        print("[Costs][LeakageGuard] high_vol_thr missing; applying MED slippage for all eval bars.")
                        df["slippage_bps"] = float(med)

                else:
                    # No returns column to compute RV: default to MED (safer than assuming HI, less strict than LO)
                    df["slippage_bps"] = float(med)

            except Exception:
                # Hard fallback
                df["slippage_bps"] = float(med)

        return df


    @contextmanager
    def _persist_results_guard(self, persist_results: bool = True):
        # Always run optional cleanup on context exit (even when persisting results).
        # This is used to aggressively release TensorFlow state between Optuna CV folds.
        if bool(persist_results):
            try:
                yield
            finally:
                try:
                    self._maybe_tf_cleanup()
                except Exception:
                    pass
            return
        _snap = {}
        try:
            d = getattr(self, '__dict__', {}) or {}
            for k, v in list(d.items()):
                if (
                    k in ('results', 'results_full', '_cv_last_eval_df', '_cv_fold_eval_frames')
                    or k.startswith('_last_')
                    or k.startswith('_cv_last_')
                ):
                    _snap[k] = v
            yield
        finally:
            try:
                d = getattr(self, '__dict__', {}) or {}
                cur_keys = set(d.keys())
                snap_keys = set(_snap.keys())
                # Remove any newly-created ephemeral keys
                for k in (cur_keys - snap_keys):
                    if (
                        k in ('results', 'results_full', '_cv_last_eval_df', '_cv_fold_eval_frames')
                        or k.startswith('_last_')
                        or k.startswith('_cv_last_')
                    ):
                        try:
                            delattr(self, k)
                        except Exception:
                            pass
                # Restore snapshots
                for k, v in _snap.items():
                    try:
                        setattr(self, k, v)
                    except Exception:
                        pass
            except Exception:
                pass
            # Optional TF cleanup (CV deep models) even when we snapshot/restore results.
            try:
                self._maybe_tf_cleanup()
            except Exception:
                pass

    def _maybe_tf_cleanup(self):
        """Best-effort memory cleanup hook (primarily for Optuna CV)."""
        try:
            do = bool(getattr(self, "_tf_cleanup_do", False))
            if not do:
                return

            # Drop model reference if requested
            if bool(getattr(self, "_tf_cleanup_del_model", False)):
                try:
                    if hasattr(self, "model"):
                        self.model = None
                except Exception:
                    pass

            # Clear TF/Keras graph state (helps prevent accumulation across folds)
            try:
                tf.keras.backend.clear_session()
            except Exception:
                pass

            # Release Python-side allocations
            try:
                import gc as _gc
                _gc.collect()
            except Exception:
                pass

            # Small yield to let allocators settle (no functional effect)
            try:
                time.sleep(0.05)
            except Exception:
                pass
        finally:
            # Never let flags leak into the next call
            try:
                self._tf_cleanup_do = False
                self._tf_cleanup_del_model = False
            except Exception:
                pass



    def _fit_deep_calibration_and_coverage(
        self,
        *,
        X_cal,
        y_cal,
        pred_fn,
        model_type: str,
        in_cv: bool,
    ) -> None:
        """
        Unified deep calibration (optional temperature) + coverage threshold fit.

        Contract:
        - X_cal: array-like, first dim is sample/window count
        - y_cal: array-like or None, aligned with X_cal (optional)
        - pred_fn: callable(X)->proba (n, C)
        - model_type: 'cnn'/'lstm'/'transformer'
        - in_cv: True when running Optuna CV folds
        """
        try:
            cfg = getattr(self, "features_config", {}) or {}
            
            if not callable(pred_fn):
                return

            frac = float(cfg.get("deep_calibration_frac", 0.10))
            nmin = int(cfg.get("deep_calibration_min_samples", 500))

            nwin = int(getattr(X_cal, "shape", [0])[0]) if X_cal is not None else 0
            if nwin <= 1:
                return

            # robust clamp
            frac = max(0.01, min(frac, 0.99))
            ncal = max(nmin, int(round(nwin * frac)))
            ncal = min(ncal, nwin - 1) if nwin > 1 else 0
            if ncal < 50:
                return

            X_tail = X_cal[-ncal:]
            y_tail = None
            try:
                if y_cal is not None and len(y_cal) >= ncal:
                    y_tail = np.asarray(y_cal[-ncal:], dtype=int)
            except Exception:
                y_tail = None

            # predict proba on tail
            p_tail = sanitize_proba(pred_fn(X_tail))

            # --- optional: Brier/NLL for selection (only if labels exist)
            try:
                if y_tail is not None:
                    brier, nll = compute_brier_and_nll(p_tail, y_tail.astype(int))
                    self._last_calib_brier = float(brier)
                    self._last_calib_nll   = float(nll)
                    self._last_calib_n     = int(len(y_tail))
                    if bool(cfg.get("print_cv_debug", False)):
                        _ctx = "cv" if in_cv else "eval"
                        print(
                            f"[Calib/deep] model={model_type} ctx={_ctx} "
                            f"brier={float(brier):.6f} nll={float(nll):.6f} n={int(len(y_tail))}"
                        )
            except Exception as _e2:
                if bool(cfg.get("print_cv_debug", False)):
                    print(f"[WARN] [Calib/deep] metrics skipped: {_e2}")

            # --- temperature (keep prior behavior: do NOT do this in CV unless you explicitly enable it)
            use_temp = bool(cfg.get("deep_calibrate", False)) and (
                str(cfg.get("deep_calibration_method", "temperature")).lower() == "temperature"
            )
            allow_temp_in_cv = bool(cfg.get("deep_calibrate_in_cv", False))
            if use_temp and (not in_cv or allow_temp_in_cv):
                if y_tail is not None:
                    try:
                        self._deep_temp_T = float(fit_temperature_from_proba(p_tail, y_tail))
                        p_tail = apply_temperature_to_proba(p_tail, float(self._deep_temp_T))
                        if self._is_debug():
                            _ctx = "cv" if in_cv else "eval"
                            print(
                                f"[Calib] model={model_type} ctx={_ctx} Temp T={float(self._deep_temp_T):.3f} "
                                f"on {int(len(y_tail))} cal rows."
                            )
                    except Exception as _e:
                        if self._is_debug():
                            print(f"[WARN] [Calib] temperature fit skipped: {_e}")

            # --- coverage threshold (requested if gating_mode=coverage OR target_active_rate>0)
            _mode = str(cfg.get("gating_mode", cfg.get("gate_mode", "threshold"))).lower()
            _tar = cfg.get("target_active_rate", None)
            try:
                _tar = float(_tar) if _tar is not None else None
            except Exception:
                _tar = None
            _use_cov_fit = (_mode == "coverage") or (_tar is not None and _tar > 0)

            if _use_cov_fit:
                if in_cv and (not bool(cfg.get("coverage_calibrate_in_cv", True))):
                    return
                tgt = float(_tar) if (_tar is not None and _tar > 0) else float(cfg.get("target_coverage", 0.10))
                thr = float(fit_coverage_threshold_on_calibration(p_tail, tgt))

                # store consistently
                self._deep_coverage_thr = float(thr)
                self._coverage_conf_thr = float(thr)
                try:
                    setattr(self, "_last_cov_cal_rows", int(ncal))
                except Exception:
                    pass
                if in_cv:
                    setattr(self, "_cv_cov_thr_last", float(thr))

                # ctx marker contract
                _ctx = "cv" if in_cv else "eval"
                if not in_cv:
                    try:
                        if bool(getattr(self, "_in_real_sim", False)):
                            mx = int(cfg.get("month_ix", getattr(self, "_rt_month_ix", 0) or 0))
                            _ctx = f"real_m{mx}"
                    except Exception:
                        pass

                print(
                    f"[Calib][Coverage] model={model_type} conf_thr={float(thr):.6f} "
                    f"target_active_rate={float(tgt):.6f} cal_rows={int(ncal)} ctx={_ctx}"
                )

        except Exception as _e:
            print(f"[WARN] [Calib/deep] model={model_type} skipped: {_e}")
            
    # ------------------------------------------------------------------
    # Lazy singleton ProcessPoolExecutor for deep model isolation.
    # Reusing the same worker process avoids repeated TF import (~15-30s
    # per call).  Created on first use, shut down via atexit.
    # ---
    # Per-job tracking: after pool creation, the worker PIDs are registered
    # with api.process_cleanup so cancellation can terminate them.
    # ------------------------------------------------------------------
    _deep_pool: "concurrent.futures.ProcessPoolExecutor | None" = None
    _deep_pool_job_id: "str | None" = None   # tracks which job owns the pool

    @classmethod
    def _get_deep_pool(cls, job_id: str = "") -> "concurrent.futures.ProcessPoolExecutor":
        """Return (and lazily create) the shared deep-model worker pool."""
        if cls._deep_pool is None or cls._deep_pool._shutdown_mutex.locked():
            import concurrent.futures
            cls._deep_pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=1,
                mp_context=__import__("multiprocessing").get_context("spawn"),
            )
            cls._deep_pool_job_id = job_id if job_id else None
            # Register worker PIDs with process registry
            if job_id:
                try:
                    for w in cls._deep_pool._processes.values():
                        pid = getattr(w, "pid", None)
                        if pid is not None:
                            from api.process_cleanup import register_job_process as _reg_proc
                            _reg_proc(job_id, pid)
                except Exception:
                    pass
        return cls._deep_pool

    @classmethod
    def _shutdown_deep_pool(cls):
        """Gracefully shut down the worker pool (called at exit or on cancel)."""
        if cls._deep_pool is not None:
            try:
                cls._deep_pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            cls._deep_pool = None
            cls._deep_pool_job_id = None

    def _deep_fit_predict_subprocess(
        self,
        *,
        model_type: str,
        mode: str,  # "seq" or "3d"
        X_train_2d: np.ndarray,
        y_train_1d: np.ndarray,
        X_test_2d: np.ndarray,
        win: int,
        train_stride: int,
        max_train_windows: int,
        batch_size: int,
        epochs: int,
        params: dict,
    ):
        """
        Run deep fit+predict in an isolated worker process via
        ProcessPoolExecutor (spawn-safe on Windows).

        Benefits over subprocess.run():
          - Worker process is reused -> no repeated TF import
          - Proper exception propagation via Future.result()
          - Same memory isolation (OS-level process boundary)

        Returns (proba_test: np.ndarray, coverage_thr: float).
        """
        in_cv = bool(getattr(self, "_in_optuna_cv", False))
        cfg = dict(getattr(self, "features_config", {}) or {})
        allow_in_cv = bool(cfg.get("deep_use_subprocess_in_cv", False)) or \
            str(os.getenv("MLB_DEEP_SUBPROCESS_CV", "0")).lower() in ("1", "true", "yes")
        if in_cv and (not allow_in_cv):
            return None, None

        # --- Prepare temp dir + .npy files (same as before) ---
        tmpdir = tempfile.mkdtemp(prefix="mlb_deep_subproc_")
        Xtr_p = os.path.join(tmpdir, "X_train.npy")
        ytr_p = os.path.join(tmpdir, "y_train.npy")
        Xte_p = os.path.join(tmpdir, "X_test.npy")
        proba_out = os.path.join(tmpdir, "proba_test.npy")
        job_json = os.path.join(tmpdir, "job.json")
        out_json = os.path.join(tmpdir, "out.json")

        np.save(Xtr_p, np.asarray(X_train_2d, dtype=np.float32))
        np.save(ytr_p, np.asarray(y_train_1d, dtype=np.int32))
        np.save(Xte_p, np.asarray(X_test_2d, dtype=np.float32))

        job = {
            "model_type": str(model_type),
            "mode": str(mode),
            "win": int(win or 0),
            "train_stride": int(train_stride or 1),
            "max_train_windows": int(max_train_windows or 10000),
            "batch_size": int(batch_size or 128),
            "epochs": int(epochs or 20),
            "params": dict(params or {}),
            "features_config": dict(getattr(self, "features_config", {}) or {}),
            "seed": int(getattr(self, "_current_seed", 11111)),
            "X_train_path": Xtr_p,
            "y_train_path": ytr_p,
            "X_test_path": Xte_p,
            "proba_test_out": proba_out,
            "out_json": out_json,
        }
        with open(job_json, "w", encoding="utf-8") as f:
            json.dump(job, f)

        # --- Submit to ProcessPoolExecutor ---
        try:
            from pipeline.workers import deep_fit_predict_worker
            job_id = getattr(self, "_job_id", "") or ""
            pool = self._get_deep_pool(job_id)
            future = pool.submit(deep_fit_predict_worker, job_json)
            result = future.result(timeout=600)  # 10-min timeout
        except Exception as e:
            print(f"[WARN] [DEEP_WORKER] failed: {e}")
            return None, None

        if not result.get("success", False):
            print(f"[WARN] [DEEP_WORKER] error: {result.get('error', 'unknown')}")
            return None, None

        try:
            thr = float(result.get("coverage_thr", np.nan))
            proba = np.load(proba_out)
            return proba, thr
        except Exception as e:
            print(f"[WARN] [DEEP_WORKER] load outputs failed: {e}")
            return None, None


import atexit
atexit.register(DeepMixin._shutdown_deep_pool)

