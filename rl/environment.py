import numpy as np
import pandas as pd

class TradingEnv:
    def __init__(self, data, features, slippage=0.00005, window=10):
        # Make indexing cheap + stable.
        self.data = data.reset_index(drop=True)

        # Guardrail: the agent must never observe the raw reward target.
        # If you want return information, use *lagged* return features (e.g., returns_lag1).
        self.features = list(features)
        if "returns" in self.features:
            raise ValueError("TradingEnv: 'returns' must not be included in features (leakage risk).")

        if "returns" not in self.data.columns:
            raise ValueError("TradingEnv: data must include a 'returns' column for reward computation.")

        missing = [c for c in self.features if c not in self.data.columns]
        if missing:
            raise ValueError(f"TradingEnv: missing feature columns: {missing}")

        self.slippage = float(slippage)
        self.window = int(window)
        if self.window < 1:
            raise ValueError("TradingEnv: window must be >= 1")

        self.n_steps = int(len(self.data))
        if self.n_steps < 2:
            raise ValueError("TradingEnv: need at least 2 rows to compute next-bar reward.")

        self.action_space = [-1, 0, 1]

        # Precompute arrays (speed). Rewards are in return units.
        self._feat = self.data[self.features].to_numpy(dtype=np.float32, copy=False)
        self._rets = self.data["returns"].to_numpy(dtype=np.float32, copy=False)
        self._rets = np.nan_to_num(self._rets, nan=0.0, posinf=0.0, neginf=0.0)

        # IMPORTANT: keep reward shaping/costs OUT of the base env.
        # Use wrappers (CostAwareWrapper / RewardProcessWrapper) so we can
        # reason about economics + avoid double-charging.
        self.reset()

    def reset(self):
        # Start where the state window is fully populated.
        # We cap at n_steps-2 because the reward uses next-bar returns.
        self.idx = max(0, min(self.n_steps - 2, self.window - 1))  
        self.position = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        start = max(0, self.idx - self.window + 1)
        state = self._feat[start:self.idx + 1]
        if state.shape[0] < self.window:
            pad = np.zeros((self.window - state.shape[0], self._feat.shape[1]), dtype=np.float32)
            state = np.vstack([pad, state])
        return state

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode finished. Call reset() first.")
        reward = self._compute_reward(action)
        self.idx += 1
        if self.idx >= self.n_steps - 1:
            self.done = True
        next_state = self._get_state()
        return next_state, reward, self.done, {}

    def _compute_reward(self, action):
        # Causal timing: action at time t earns PnL on bar t+1.
        # Prevents the agent from exploiting same-bar returns present in the state.
        next_i = self.idx + 1
        if next_i >= self.n_steps:
            return 0.0

        r_next = float(self._rets[next_i])
        reward = float(action) * r_next

        # Internal slippage/bonuses are intentionally not applied here.
        # Costs and reward stabilization belong in wrappers.
        self.position = action
        return reward