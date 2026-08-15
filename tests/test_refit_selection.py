"""Tests for the corrected refit-selection protocol (P4).

The deployment refit (uncapped) must only replace the Optuna-validated
snapshot when it actually beats it on the same fold; ties prefer the
original (validated) metrics.
"""
import numpy as np

from pipeline.tuning import refit as refit_mod


def _metrics_tuple(cstrategy=1.0):
    """A 16-metric tuple matching METRIC_NAMES order."""
    t = [0.0] * 16
    t[0] = cstrategy  # cstrategy
    t[2] = cstrategy  # creturns (placeholder)
    return tuple(t)


class _FakeBacktester:
    pass


class TestSelectBetterResult:
    def test_refit_better_is_chosen(self, monkeypatch):
        calls = {}

        def fake_original(backtester, best_params, t_s, t_e, ts, te, allow_dqn=False):
            calls["original"] = 1
            return _metrics_tuple(1.0), "cnn"

        def fake_refit(backtester, best_params, t_s, t_e, ts, te, overrides=None, **_):
            calls["refit"] = 1
            return _metrics_tuple(1.5)

        monkeypatch.setattr(refit_mod, "_evaluate_original_no_refit", fake_original)
        chosen = refit_mod._select_better_result(
            _FakeBacktester(), {}, "2025-01-01", "2025-02-01",
            "2025-02-01", "2025-03-01", refit_func=fake_refit,
        )
        assert calls.get("original") == 1 and calls.get("refit") == 1
        assert chosen[0] == 1.5

    def test_refit_worse_original_chosen(self, monkeypatch):
        def fake_original(backtester, best_params, t_s, t_e, ts, te, allow_dqn=False):
            return _metrics_tuple(1.0), "cnn"

        def fake_refit(backtester, best_params, t_s, t_e, ts, te, overrides=None, **_):
            return _metrics_tuple(0.9)

        monkeypatch.setattr(refit_mod, "_evaluate_original_no_refit", fake_original)
        chosen = refit_mod._select_better_result(
            _FakeBacktester(), {}, "2025-01-01", "2025-02-01",
            "2025-02-01", "2025-03-01", refit_func=fake_refit,
        )
        assert chosen[0] == 1.0

    def test_tie_prefers_original(self, monkeypatch):
        def fake_original(backtester, best_params, t_s, t_e, ts, te, allow_dqn=False):
            return _metrics_tuple(1.0), "cnn"

        def fake_refit(backtester, best_params, t_s, t_e, ts, te, overrides=None, **_):
            return _metrics_tuple(1.005)  # within 1% tolerance

        monkeypatch.setattr(refit_mod, "_evaluate_original_no_refit", fake_original)
        chosen = refit_mod._select_better_result(
            _FakeBacktester(), {}, "2025-01-01", "2025-02-01",
            "2025-02-01", "2025-03-01", refit_func=fake_refit,
        )
        assert chosen[0] == 1.0

    def test_invalid_metrics_shape_falls_back_to_original(self, monkeypatch):
        def fake_original(backtester, best_params, t_s, t_e, ts, te, allow_dqn=False):
            return _metrics_tuple(1.0), "cnn"

        def fake_refit(backtester, best_params, t_s, t_e, ts, te, overrides=None, **_):
            return (1.5,)  # wrong arity

        monkeypatch.setattr(refit_mod, "_evaluate_original_no_refit", fake_original)
        chosen = refit_mod._select_better_result(
            _FakeBacktester(), {}, "2025-01-01", "2025-02-01",
            "2025-02-01", "2025-03-01", refit_func=fake_refit,
        )
        assert chosen[0] == 1.0


class TestFinalRefitIfDeepRouting:
    def test_deep_model_routes_through_selection(self, monkeypatch):
        captured = {}

        def fake_select(backtester, best_params, train_start, train_end,
                        test_start, test_end, refit_func, select_metric,
                        tolerance, overrides):
            captured["refit_func"] = refit_func.__name__
            return _metrics_tuple(1.0)

        monkeypatch.setattr(refit_mod, "_select_better_result", fake_select)
        refit_mod.final_refit_if_deep(
            _FakeBacktester(), {"model_type": "cnn"},
            "2025-01-01", "2025-02-01", "2025-02-01", "2025-03-01",
        )
        assert captured["refit_func"] == "refit_cnn_with_overrides"

    def test_ensemble_routes_through_selection(self, monkeypatch):
        captured = {}

        def fake_select(backtester, best_params, train_start, train_end,
                        test_start, test_end, refit_func, select_metric,
                        tolerance, overrides):
            captured["refit_func"] = refit_func.__name__
            return _metrics_tuple(1.0)

        monkeypatch.setattr(refit_mod, "_select_better_result", fake_select)
        refit_mod.final_refit_if_deep(
            _FakeBacktester(), {"model_type": "ensemble_cnn_lstm_xgboost"},
            "2025-01-01", "2025-02-01", "2025-02-01", "2025-03-01",
        )
        assert captured["refit_func"] == "refit_ensemble_cnn_lstm_xgb_with_overrides"

    def test_classical_model_runs_original_only(self, monkeypatch):
        captured = {}

        def fake_original(backtester, best_params, t_s, t_e, ts, te, allow_dqn=False):
            captured["called"] = True
            return _metrics_tuple(2.0), "xgboost"

        monkeypatch.setattr(refit_mod, "_evaluate_original_no_refit", fake_original)
        out = refit_mod.final_refit_if_deep(
            _FakeBacktester(), {"model_type": "xgboost"},
            "2025-01-01", "2025-02-01", "2025-02-01", "2025-03-01",
        )
        assert captured["called"] is True
        assert out[0] == 2.0
