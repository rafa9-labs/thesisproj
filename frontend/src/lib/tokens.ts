export const layout = {
  sidebarCollapsed: 64,
  sidebarExpanded: 220,
  headerHeight: 48,
  breadcrumbHeight: 32,
  statusBarHeight: 24,
  terminalHeight: 200,
  minWidth: 1280,
  minHeight: 800,
} as const;

export const typography = {
  metricValue: { fontFamily: "JetBrains Mono", fontSize: "24px", fontWeight: 600 },
  metricLabel: { fontFamily: "Inter", fontSize: "11px", fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.12em" },
  navItem: { fontFamily: "Inter", fontSize: "11px", fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.12em" },
  sectionHeader: { fontFamily: "Inter", fontSize: "16px", fontWeight: 600 },
  body: { fontFamily: "Inter", fontSize: "13px", fontWeight: 400 },
  bodySmall: { fontFamily: "Inter", fontSize: "12px", fontWeight: 400 },
  engineParam: { fontFamily: "JetBrains Mono", fontSize: "12px", fontWeight: 400 },
  price: { fontFamily: "JetBrains Mono", fontSize: "14px", fontWeight: 400 },
  button: { fontFamily: "Inter", fontSize: "12px", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em" },
  terminalLog: { fontFamily: "JetBrains Mono", fontSize: "12px", fontWeight: 400 },
} as const;

export const modelCategories = {
  classical: {
    label: "Classical",
    color: "#22D3EE",
    models: ["logistic", "svm", "random_forest", "decision_tree", "xgboost", "lightgbm", "catboost"],
  },
  deep: {
    label: "Deep Learning",
    color: "#A78BFA",
    models: ["cnn", "lstm", "transformer", "gru", "gru_lstm"],
  },
  rl: {
    label: "Reinforcement Learning",
    color: "#F59E0B",
    models: ["dqn"],
  },
  ensemble: {
    label: "Ensemble",
    color: "#EC4899",
    models: ["ensemble_adaptive_regime", "ensemble_cnn_lstm_xgboost", "meta_ensemble", "stacking_ensemble", "regime_classifier"],
  },
} as const;

export const modelDescriptions: Record<string, { name: string; short: string; apprentice: string }> = {
  logistic: { name: "Logistic Regression", short: "The Baseline", apprentice: "Fast linear model. Use to establish baseline performance before deploying heavy neural networks." },
  svm: { name: "Support Vector Machine", short: "The Kernel", apprentice: "Kernel-based classifier (RBF). Good for non-linear decision boundaries in feature space." },
  random_forest: { name: "Random Forest", short: "The Forest", apprentice: "Ensemble of decision trees with bagging. Robust, handles mixed features well." },
  decision_tree: { name: "Decision Tree", short: "The Tree", apprentice: "Single decision tree. Highly interpretable but prone to overfitting. Use for feature analysis." },
  xgboost: { name: "XGBoost", short: "The Workhorse", apprentice: "Gradient-boosted trees. Exceptionally good at finding complex non-linear patterns in tabular data." },
  lightgbm: { name: "LightGBM", short: "The Speedster", apprentice: "Histogram-based gradient boosting (Microsoft). Faster training than XGBoost with competitive accuracy." },
  catboost: { name: "CatBoost", short: "The Cipher", apprentice: "Ordered boosting with native categorical handling (Yandex). Excels with minimal tuning." },
  cnn: { name: "CNN", short: "Pattern Scanner", apprentice: "1D convolutional network that learns local price patterns across sliding windows." },
  lstm: { name: "LSTM", short: "Time Traveler", apprentice: "Deep learning network designed for sequential time-series. Remembers price action across hundreds of bars." },
  transformer: { name: "Transformer", short: "Attention Engine", apprentice: "Self-attention architecture that weighs the importance of every historical bar simultaneously." },
  gru: { name: "GRU", short: "The Efficiency Expert", apprentice: "Gated Recurrent Unit. Simpler and faster than LSTM while matching or exceeding performance on FX data." },
  gru_lstm: { name: "GRU-LSTM Hybrid", short: "The Hybrid", apprentice: "Stacks GRU before LSTM layers. Research shows this outperforms standalone models on forex prediction." },
  dqn: { name: "Dueling DQN", short: "Autonomous Agent", apprentice: "Reinforcement learning agent. Instead of predicting price, it learns a trading policy through trial and error." },
  ensemble_adaptive_regime: { name: "Adaptive Regime", short: "The Shapeshifter", apprentice: "Dynamically shifts between models depending on market regime (trending vs ranging)." },
  ensemble_cnn_lstm_xgboost: { name: "CNN+LSTM+XGB", short: "The Triad", apprentice: "Three-way stacking ensemble combining CNN, LSTM, and XGBoost for robust predictions." },
  meta_ensemble: { name: "Signal Committee", short: "The Committee", apprentice: "Wraps multiple models and combines their predictions via voting. Run logistic + xgboost + lstm as one." },
  stacking_ensemble: { name: "Stacking Ensemble", short: "The Strategist", apprentice: "Trains a meta-learner (Logistic Regression) on out-of-fold predictions from multiple base models." },
  regime_classifier: { name: "Regime Classifier", short: "The Diplomat", apprentice: "Random Forest classifier labeling market regime (7 states). Used by exploration agents and committee routers for specialist model selection." },
};
