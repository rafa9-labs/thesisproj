"""RL environment wrappers backported from utilsNoWFO.py.

Phase 4.5 -- cost-aware and reward-shaping wrappers for gym environments.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class CostAwareWrapper:
    """
    Wraps a TradingEnv-like environment so the step reward becomes:

        reward_net = reward_gross
                     - cost_scale * (spread + slippage)
                     - turnover_penalty  (on flips)

    Assumes `action` encodes position state; if action changes vs. previous,
    we charge costs aligned to bar t (arrays are aligned to env steps).

    - `cost_scale` > 1.0 makes transaction costs bite harder (e.g. to
      reflect unmodelled costs or deliberately discourage churn).
    - `turnover_penalty` adds an extra fixed penalty on every flip,
      independent of the spread / slippage arrays.
    """
    def __init__(
        self,
        env,
        *,
        spread=None,
        slippage_bps=None,
        mid_price=None,
        cost_scale: float = 1.0,
        turnover_penalty: float = 0.0,
    ):
        self.env = env
        self.spread = np.asarray(spread, dtype=np.float32) if spread is not None else None
        self.slip   = np.asarray(slippage_bps, dtype=np.float32) if slippage_bps is not None else None
        self.cost_scale = float(cost_scale)
        self.price  = np.asarray(mid_price, dtype=np.float32) if mid_price is not None else None

        self.turnover_penalty = float(turnover_penalty)
        self.t = 0
        self._fallback_t = 0

        # Optional: try to discover a price series so spread (price units) can be converted
        # into fractional return drag consistent with env reward units.
        self._price = None
        try:
            if hasattr(env, "data") and isinstance(env.data, pd.DataFrame):
                if "price" in env.data.columns:
                    self._price = pd.to_numeric(env.data["price"], errors="coerce").to_numpy(dtype=np.float32, copy=False)
                elif "mid_close" in env.data.columns:
                    self._price = pd.to_numeric(env.data["mid_close"], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        except Exception:
            self._price = None

    def reset(self, *args, **kwargs):
        self.t = 0
        self.prev_action = 0
        self._fallback_t = 0
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if self.spread is not None and self.slip is not None:
            try:
                if action != self.prev_action:
                    # IMPORTANT:
                    # TradingEnv computes reward using the *next* bar (idx+1) then increments idx.
                    # After env.step(), env.idx points to the bar that generated this reward.
                    env_idx = getattr(self.env, "idx", None)
                    if env_idx is None:
                        env_idx = self._fallback_t
                    env_idx = int(env_idx)

                    c_sp = float(self.spread[env_idx]) if 0 <= env_idx < int(self.spread.shape[0]) else 0.0
                    c_sl = float(self.slip[env_idx]) * 1e-4 if 0 <= env_idx < int(self.slip.shape[0]) else 0.0

                    # Convert spread to fractional return drag if we have a price series.
                    if self._price is not None and 0 <= env_idx < int(self._price.shape[0]):
                        px = float(self._price[env_idx])
                        if np.isfinite(px) and px > 0:
                            c_sp = c_sp / px

                    total_cost = self.cost_scale * (c_sp + c_sl) + self.turnover_penalty
                    reward = float(reward) - float(total_cost)

                    # Helpful audit hook (ignored by the rest of the pipeline)
                    try:
                        if isinstance(info, dict):
                            info["tx_cost"] = float(total_cost)
                    except Exception:
                        pass
            except Exception:
                pass
        self.prev_action = action
        self._fallback_t += 1
        return state, reward, done, info

    # proxy everything else to the wrapped env
    def __getattr__(self, name):
        return getattr(self.env, name)


class RewardProcessWrapper:
    """
    Optional wrapper for the environment's reward to stabilise learning.

    Supports three transforms (can combine):
    - clip_mode = "tanh":  r' = tanh(k * r)
    - clip_mode = "range": r' = clip(r, lo, hi)
    - norm = True: running mean-variance normalization (like PPO/A2C baselines).
    """
    def __init__(
        self,
        env,
        clip_mode=None,
        tanh_k=3.0,
        clip_range=(-1.0, 1.0),
        norm=True,
        norm_beta=0.99,
    ):
        self.env = env
        self.clip_mode = clip_mode
        self.tanh_k = float(tanh_k)
        self.clip_range = tuple(clip_range)
        self.norm = bool(norm)
        self.norm_beta = float(norm_beta)
        self._running_mean = 0.0
        self._running_var = 1.0

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        r = float(reward)

        # Optional clipping
        if self.clip_mode == "tanh":
            import numpy as _np
            r = float(_np.tanh(self.tanh_k * r))
        elif self.clip_mode == "range":
            lo, hi = self.clip_range
            r = float(max(lo, min(hi, r)))

        # Optional running normalization
        if self.norm:
            m = self._running_mean
            v = self._running_var
            beta = self.norm_beta
            m_new = (1 - beta) * r + beta * m
            v_new = (1 - beta) * ((r - m) ** 2) + beta * v
            self._running_mean, self._running_var = m_new, v_new
            if v_new > 0:
                r = (r - m_new) / (v_new ** 0.5)

        return obs, r, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)

