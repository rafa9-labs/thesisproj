import numpy as np
import random
import os
import json

# LEAGCY

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")  # no pre-allocation

def _dqn_is_debug() -> bool:
    return os.environ.get("LOG_MODE", "COMPACT").upper() == "DEBUG"

import tensorflow as tf
from tensorflow import keras

# Keras aliases so the rest of the code stays the same
Sequential  = keras.Sequential
Dense       = keras.layers.Dense
LSTM        = keras.layers.LSTM
Dropout     = keras.layers.Dropout
Lambda      = keras.layers.Lambda
Input       = keras.Input

Adam         = keras.optimizers.Adam
regularizers = keras.regularizers


# ✅ Optional: Enable dynamic memory allocation
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if os.environ.get("LOG_MODE", "COMPACT").upper() == "DEBUG":
            if _dqn_is_debug():
               print("✅ Enabled dynamic GPU memory allocation.")
    except RuntimeError as e:
        print("❌ Could not enable memory growth:", e)


def filter_dqn_config(config):
    """Filter only the kwargs accepted by DQNAgent and rename keys if needed."""
    valid_keys = [
        "state_size", "action_size", "gamma", "epsilon", "epsilon_min", "epsilon_decay",
        "learning_rate", "batch_size", "buffer_size", "target_update_freq",
        "replay_freq", "episodes", "window", "warmup_steps", "max_steps_per_episode",
        "log_every", "epsilon_decay_steps", "use_prioritized_replay", "per_alpha", "per_beta_start", "per_beta_steps",
        "env_reward_clip", "env_reward_tanh_k", "env_reward_clip_range",
        "env_reward_norm", "env_reward_norm_beta" # <— keep linear ε on load()
    ]
    mapping = {"memory_size": "buffer_size"}  # Rename in config
    filtered = {}
    for k, v in config.items():
        if k in mapping:
            filtered[mapping[k]] = v
        elif k in valid_keys:
            filtered[k] = v
    return filtered


# ----------------------------- Replay Buffer -----------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        )
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return (
            np.stack(states).astype(np.float32),
            actions.astype(np.int64),
            rewards.astype(np.float32),
            np.stack(next_states).astype(np.float32),
            dones.astype(np.bool_),
        )

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, alpha=0.6, eps=1e-6):
        super().__init__(capacity)

        self.alpha = float(alpha)
        self.eps = float(eps)
        self.priorities = np.zeros(int(capacity), dtype=np.float32)
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        super().push(state, action, reward, next_state, done)
        idx = (self.position - 1) % self.capacity
        self.priorities[idx] = self.max_priority

    def sample(self, batch_size, beta=0.4):
        size = len(self.buffer)
        assert size >= batch_size, "Not enough samples for PER."
        pr = self.priorities[:size] + self.eps
        probs = pr ** self.alpha
        probs = probs / probs.sum()
        idxs = np.random.choice(size, batch_size, p=probs, replace=False)
        (states, actions, rewards, next_states, dones) = map(np.array, zip(*[self.buffer[i] for i in idxs]))
        weights = (size * probs[idxs]) ** (-float(beta))
        weights = weights / weights.max()
        return (
            np.stack(states).astype(np.float32),
            actions.astype(np.int64),
            rewards.astype(np.float32),
            np.stack(next_states).astype(np.float32),
            dones.astype(np.bool_),
            idxs.astype(np.int64),
            weights.astype(np.float32),
        )

    def update_priorities(self, idxs, td_errors):
        import numpy as np
        p = np.abs(np.asarray(td_errors, dtype=np.float32)) + self.eps
        self.priorities[idxs] = p
        self.max_priority = max(self.max_priority, float(p.max()))


# -------------------------- Dueling merge layer ---------------------------
class DuelingMerge(keras.layers.Layer):
    """
    Custom layer for dueling DQN:
    Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
    """
    def call(self, inputs):
        V, A = inputs
        mean_A = tf.reduce_mean(A, axis=1, keepdims=True)
        return V + (A - mean_A)

    def get_config(self):
        base = super().get_config()
        return base


# ------------------------------- DQN Agent -------------------------------
class DQNAgent:
    def __init__(self, state_size, action_size=3, window=10, gamma=0.99, epsilon=1.0,
                epsilon_min=0.1, epsilon_decay=0.995, learning_rate=0.001,
                batch_size=32, buffer_size=10000, target_update_freq=10,
                replay_freq=1, episodes=2, seed=None, **kwargs):
        # Optional: reproducibility. Prefer explicit `seed=` argument; fall back to kwargs.
        # (We intentionally do NOT force full TF determinism; that can be slow/brittle.)
        seed = seed if seed is not None else kwargs.get("seed", None)
        self.seed = seed
        if self.seed is not None:
            try:
                s = int(self.seed)
                import random as _random
                _random.seed(s)
                np.random.seed(s)
                tf.random.set_seed(s)
            except Exception:
                pass


        # Core dims
        self.window = int(window)
        self.state_size = int(state_size)
        self.action_size = int(action_size)

        # RL params
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.batch_size = int(batch_size)
        self.target_update_freq = int(target_update_freq)
        self.replay_freq = int(replay_freq)
        self.episodes = int(episodes)

        # Persist full config for save()/load()
        self.config = {
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "learning_rate": float(learning_rate),
            "batch_size": self.batch_size,
            "buffer_size": int(buffer_size),
            "target_update_freq": self.target_update_freq,
            "replay_freq": self.replay_freq,
            "episodes": self.episodes,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "window": self.window,
            "warmup_steps": int(kwargs.get("warmup_steps", max(5 * self.batch_size, 512))),
            "max_steps_per_episode": kwargs.get("max_steps_per_episode", None),
            "log_every": int(kwargs.get("log_every", 250)),
        }
        
        if self.seed is not None:
            self.config["seed"] = int(self.seed)
        
        # Persist advanced config keys so save()/load() round-trips them
        for k in [
            "epsilon_decay_steps",
            "use_prioritized_replay", "per_alpha", "per_beta_start", "per_beta_steps",
            "env_reward_clip", "env_reward_tanh_k", "env_reward_clip_range",
            "env_reward_norm", "env_reward_norm_beta",
        ]:
            if k in kwargs:
                self.config[k] = kwargs[k]

        # Runtime hooks (allow overrides via config)
        self.warmup_steps = int(self.config["warmup_steps"])
        self.max_steps_per_episode = self.config["max_steps_per_episode"]
        self.replay_freq = int(self.config.get("replay_freq", self.replay_freq))
        self.target_update_freq = int(self.config.get("target_update_freq", self.target_update_freq))
        self.log_every = int(self.config.get("log_every", 250))

        # Will be created in fit()/load()
        self.model = None
        self.target_model = None
        self.buffer = None
        self.train_step = 0

        # Store LR for _build_model
        self._learning_rate = float(learning_rate)

        # --- NEW: define optimizer + loss on the agent (used by custom train step) ---
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self._learning_rate)
        # Huber is robust; swap for MSE if you prefer.
        # Use a legal reduction mode for this TF/Keras version.
        self.loss_fn = tf.keras.losses.Huber(reduction='sum_over_batch_size')  # scalar loss


        # Canonical math dtype for DQN (keeps us safe even if global policy is fp16 elsewhere)
        self._dtype = tf.float32
        
        # --- PER flags (config-driven) ---
        self.use_per = bool(kwargs.get("use_prioritized_replay", True))
        self.per_alpha = float(kwargs.get("per_alpha", 0.6))
        self.per_beta_start = float(kwargs.get("per_beta_start", 0.4))
        self.per_beta_steps = int(kwargs.get("per_beta_steps", 50000))
        self.per_beta = self.per_beta_start
        
        # Linear epsilon schedule (optional override via config kwargs)
        self.epsilon_decay_steps = int(kwargs.get("epsilon_decay_steps", 20000))
        self._eps_step = 0
        self._eps_start = float(self.epsilon)

    def _update_epsilon(self):
        """Update epsilon once per environment step (correct linear schedule)."""
        # If epsilon_decay_steps > 0: linear schedule from eps_start -> eps_min.
        if self.epsilon_decay_steps and self.epsilon_decay_steps > 0:
            self._eps_step += 1
            frac = min(1.0, self._eps_step / float(self.epsilon_decay_steps))
            eps = self._eps_start + frac * (self.epsilon_min - self._eps_start)
            self.epsilon = float(max(self.epsilon_min, eps))
        else:
            # Fallback: multiplicative decay
            self.epsilon = float(max(self.epsilon_min, self.epsilon * self.epsilon_decay))


    
    def _build_model(self, learning_rate):
        # Functional so we can build Value/Advantage streams (dueling)
        inp = Input(shape=(self.window, self.state_size))
        x = LSTM(128, activation='tanh')(inp)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = Dropout(0.2)(x)
        
        V = Dense(1)(x)                         # state-value
        A = Dense(self.action_size)(x)          # per-action advantages
        # Dueling Q-values via custom layer (no Lambda with tf in the serialized graph)
        Q = DuelingMerge()([V, A])

        model = tf.keras.Model(inp, Q)
        optimizer = Adam(learning_rate=learning_rate, clipnorm=5.0)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    # ----------------------------- Core methods ----------------------------
    def act(self, state):
        """ε-greedy action from a single state of shape (window, features)."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        # Direct call avoids the extra overhead of model.predict()
        state_b = np.asarray(state, dtype=np.float32)[np.newaxis, :, :]
        q_values = self.model(state_b, training=False)
        # q_values is a Tensor; convert once and argmax on CPU
        q_np = q_values.numpy()
        return int(np.argmax(q_np[0]))


    def fit(self, env, episodes=None):
        """Train the agent with heartbeats and step caps for speed."""
        if episodes is None:
            episodes = int(self.config.get("episodes", self.episodes))

        if self.use_per:
            self.buffer = PrioritizedReplayBuffer(self.config.get("buffer_size", 10000), alpha=self.per_alpha)
        else:
            self.buffer = ReplayBuffer(self.config.get("buffer_size", 10000))

        self.model = self._build_model(self._learning_rate)
        self.target_model = self._build_model(self._learning_rate)
        self._build_train_step()  # compiles a dtype-safe tf.function using self.loss_fn/self.optimizer
        self.update_target_network(tau=0.005)
        self.train_step = 0

        # Reset ε schedule at the start of training
        self._eps_step = 0
        self._eps_start = float(self.epsilon)

        # Set a sensible default cap once we know env length.
        # We cap each episode to a fraction of the available bars so that
        # training uses enough transitions but does not run excessively long.
        if self.max_steps_per_episode is None:
            horizon_frac = float(self.config.get("episode_horizon_frac", 0.4))
            min_cap = int(self.config.get("episode_min_steps", 4000))
            max_cap = int(self.config.get("episode_max_steps", 20000))
            self.max_steps_per_episode = int(
                min(max_cap, max(min_cap, horizon_frac * env.n_steps))
            )

        if _dqn_is_debug():
            print(f"🚦 [DQN] Starting training for {episodes} episodes...")
 
        print(f"    window={self.window} | state_size={self.state_size} | "
              f"batch={self.batch_size} | replay_freq={self.replay_freq} | "
              f"target_update_freq={self.target_update_freq} | warmup={self.warmup_steps} | "
              f"max_steps_per_episode={self.max_steps_per_episode}")

        for ep in range(episodes):
            state = env.reset()
            total_reward, done, step = 0.0, False, 0
            last_loss = None
            action_counts = np.zeros(self.action_size, dtype=int)

            while not done:
                action = self.act(state)
                action_counts[action] += 1

                next_state, reward, done, _ = env.step(self._map_action(action))
                self.remember(state, action, reward, next_state, done)

                # Train every replay_freq steps after warmup
                if len(self.buffer) >= self.warmup_steps and (self.train_step % self.replay_freq == 0):
                    if self.use_per:
                        # Anneal PER beta toward 1.0
                        frac = min(1.0, self.train_step / float(max(1, self.per_beta_steps)))
                        self._per_beta = self.per_beta_start + frac * (1.0 - self.per_beta_start)
                        (S, A, R, S2, D, idxs, W) = self.buffer.sample(self.batch_size, beta=self._per_beta)
                        loss, td = self._train_step_fn(S, A, R, S2, D, W)   # weighted Huber
                        self.buffer.update_priorities(idxs, td.numpy())
                        last_loss = float(loss)
                    else:
                        (S, A, R, S2, D) = self.buffer.sample(self.batch_size)
                        loss, _ = self._train_step_fn(S, A, R, S2, D, None)
                        last_loss = float(loss)
                        
                # Update ε once per env step (correct schedule)
                self._update_epsilon()


                state = next_state
                total_reward += float(reward)
                self.train_step += 1
                step += 1

                # Hard cap episode length (fast debug)
                if self.max_steps_per_episode and step >= self.max_steps_per_episode:
                    done = True

                if self.train_step % self.target_update_freq == 0:
                    self.update_target_network(tau=0.005)

                # 🔊 heartbeat
                if step % self.log_every == 0:
                    lim = self.max_steps_per_episode or env.n_steps
                    if _dqn_is_debug():
                        print(f"[DQN] ep {ep+1}/{episodes} | step {step}/{lim} | "
                            f"ε={self.epsilon:.3f} | r={total_reward:.4f}"
                            f"loss={('%.6f' % last_loss) if last_loss is not None else '—'} | "
                            f"actions={action_counts.tolist()}")

            if _dqn_is_debug():
                print(f"✅ [DQN] Ep {ep+1}/{episodes} reward={total_reward:.4f} ε={self.epsilon:.3f}")


    def update_target_network(self, tau=None):
        if tau is None:
            self.target_model.set_weights(self.model.get_weights())
            return
        w  = self.model.get_weights()
        wt = self.target_model.get_weights()
        self.target_model.set_weights([tau*wi + (1.0 - tau)*wti for wi, wti in zip(w, wt)])


    def remember(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def replay(self):
        if len(self.buffer) < max(self.batch_size, self.warmup_steps):
            return None
        idx = np.random.randint(len(self.buffer), size=self.batch_size)
        batch = [self.buffer.buffer[i] for i in idx]
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        loss, _ = self._train_step_fn(
            states.astype(np.float32),
            actions.astype(np.int32),
            rewards.astype(np.float32),
            next_states.astype(np.float32),
            dones.astype(np.float32),
            None,  # weights
        )
        # keep the same ε schedule here too (for callers that still use replay())
        self._update_epsilon()
        return float(loss.numpy())



    # ----------------------------- Inference ------------------------------
    def predict(self, states):
        """Return best actions for given states. Accepts (batch, window, features) or (window, features)."""
        states = np.asarray(states)
        if states.ndim == 2:
            states = states[np.newaxis, :, :]
        assert states.shape[1] == self.window, f"Expected window {self.window}, got {states.shape[1]}"
        q_vals = self.model(states.astype(np.float32), training=False)
        q_np = q_vals.numpy()
        return np.argmax(q_np, axis=1)

    def predict_proba(self, states):
        states = np.asarray(states)
        if states.ndim == 2:
            states = states[np.newaxis, :, :]
        q_vals = self.model(states.astype(np.float32), training=False)
        probs = tf.nn.softmax(q_vals, axis=1)
        return probs.numpy()

    def predict_q(self, states):
        """Raw Q-values for meta fusion. Returns [n_samples, n_actions]."""
        states = np.asarray(states)
        if states.ndim == 2:
            states = states[np.newaxis, :, :]
        assert states.shape[1] == self.window, f"Expected window={self.window}, got shape={states.shape}"
        q_vals = self.model(states.astype(np.float32), training=False)
        return q_vals.numpy()

    # ------------------------------- Utils --------------------------------
    def _map_action(self, a):
        # map index -> {-1,0,1}
        return [-1, 0, 1][int(a)]

    def save(self, model_path, config_path):
        """Save the Keras model and the config dict as JSON."""
        self.model.save(model_path)
        with open(config_path, "w") as f:
            json.dump(self.config, f)
            
        if _dqn_is_debug():
            print(f"✅ DQNAgent saved to: {model_path} and {config_path}")

    def _build_train_step(self):
        # Per-sample Huber for PER; scalar Huber if not using PER
        huber_none = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)

        @tf.function
        def train_step_fn(states, actions, rewards, next_states, dones, weights=None):
            # Everything in one canonical dtype
            gamma = tf.cast(self.gamma, self._dtype)

            states      = tf.cast(states, self._dtype)
            next_states = tf.cast(next_states, self._dtype)
            rewards     = tf.cast(rewards, self._dtype)
            dones       = tf.cast(dones, self._dtype)
            actions     = tf.cast(actions, tf.int32)

            # Q(s, a) with the online net
            with tf.GradientTape() as tape:
                q_values = self.model(states, training=True)
                idx = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
                q_sa = tf.gather_nd(q_values, idx)

                # Double-DQN: a* = argmax_a Q_online(s', a)
                q_next_online = self.model(next_states, training=False)
                next_actions = tf.argmax(q_next_online, axis=1, output_type=tf.int32)

                # Target net value at (s', a*)
                q_next_target = self.target_model(next_states, training=False)
                idx2 = tf.stack([tf.range(tf.shape(next_actions)[0]), next_actions], axis=1)
                q_target_next = tf.gather_nd(q_next_target, idx2)

                # Bellman target with terminal masking
                mask = tf.cast(1.0 - dones, self._dtype)
                target = rewards + gamma * mask * q_target_next

                # Huber loss; weighted if PER is on
                td_errors = target - q_sa
                if weights is not None:
                    w = tf.cast(weights, self._dtype)
                    loss_vec = huber_none(target, q_sa)
                    loss = tf.reduce_mean(w * loss_vec)
                else:
                    loss = tf.reduce_mean(tf.keras.losses.huber(target, q_sa))

            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            return loss, tf.abs(td_errors)

        # 🔧 Bind the compiled function to the instance so callers can use it
        self._train_step_fn = train_step_fn
        return train_step_fn


    @staticmethod
    def load(model_path, config_path):
        """Reload a DQNAgent from disk (model and config), inferring missing dims."""
        import json
        from keras.models import load_model
        
        # 1) Load the Keras model first so we can infer input/output shapes if needed.
        #    safe_mode=False + custom_objects so Keras can restore our custom layer.
        model = load_model(
            model_path,
            safe_mode=False,
            custom_objects={"DuelingMerge": DuelingMerge},
        )

        # 2) Best-effort read of JSON config (may be from older versions)
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}

        # 3) Infer window/state_size/action_size from the model if missing
        try:
            _, inferred_window, inferred_state = model.input_shape  # (None, win, state)
        except Exception:
            inferred_window, inferred_state = None, None

        if "state_size" not in cfg and inferred_state is not None:
            cfg["state_size"] = int(inferred_state)
        if "window" not in cfg and inferred_window is not None:
            cfg["window"] = int(inferred_window)

        # If action_size unknown, use model's output width
        try:
            inferred_actions = int(model.output_shape[-1])
        except Exception:
            inferred_actions = 3
        cfg.setdefault("action_size", inferred_actions)

        # 4) Filter/normalize and construct agent
        cfg = filter_dqn_config(cfg)
        agent = DQNAgent(**cfg)
        
        # 5) Attach weights to both online and target nets, and prep optimizer/loss/train step
        agent.model = model
        # Load target model with the same relaxed safe mode and custom layer
        agent.target_model = load_model(
            model_path,
            safe_mode=False,
            custom_objects={"DuelingMerge": DuelingMerge},
        )
        
        agent.optimizer = tf.keras.optimizers.Adam(learning_rate=agent._learning_rate)
        # Match the constructor: legal reduction mode only
        agent.loss_fn = tf.keras.losses.Huber(reduction='sum_over_batch_size')
        agent._build_train_step()
        # Buffer for possible continued training (ensure it's not None)
        try:
            use_per = bool(cfg.get("use_prioritized_replay", True))
            if use_per:
                agent.buffer = PrioritizedReplayBuffer(cfg.get("buffer_size", 10000), alpha=float(cfg.get("per_alpha", 0.6)))
            else:
                agent.buffer = ReplayBuffer(cfg.get("buffer_size", 10000))
        except Exception:
            agent.buffer = ReplayBuffer(cfg.get("buffer_size", 10000))

        return agent

    def batch_predict(self, states):
        """Batch best actions from states of shape (batch, window, state_size)."""
        states = np.asarray(states)
        if states.ndim == 2:
            states = states[np.newaxis, :, :]
        assert states.shape[1] == self.window, f"Expected window={self.window}, got shape={states.shape}"
        q_values = self.model.predict(states, verbose=0)
        return np.argmax(q_values, axis=1)

