export const colors = {
  app: "#050608",
  surface: "#0A0D12",
  elevated: "#11151C",

  glass: "rgba(255,255,255,0.03)",
  glassBorder: "rgba(255,255,255,0.06)",
  glassHover: "rgba(255,255,255,0.05)",

  border: "#1A1F2A",
  borderSubtle: "#131820",
  borderActive: "rgba(0,229,255,0.25)",

  textPrimary: "#E8ECF1",
  textSecondary: "#7A8494",
  textMuted: "#4A5568",
  textInverse: "#050608",

  primary: "#00E5FF",
  primaryGlow: "rgba(0,229,255,0.15)",

  brand: "#00E5FF",
  brandGlow: "rgba(0,229,255,0.15)",

  accent: "#00E5FF",
  accentSuccess: "#22C55E",
  accentDanger: "#EF4444",
  accentWarning: "#F59E0B",
  accentInfo: "#00E5FF",
  accentClassical: "#22D3EE",
  accentDeep: "#A78BFA",
  accentRl: "#F59E0B",
  accentEnsemble: "#EC4899",

  chartLine: "#00E5FF",
  chartBuyhold: "#555555",
  chartDrawdown: "rgba(239,68,68,0.35)",

  eventHigh: "#EF4444",
  eventMedium: "#F59E0B",
  eventLow: "#00E5FF",
} as const;

export const spacing = {
  1: "4px",
  2: "8px",
  3: "12px",
  4: "16px",
  5: "20px",
  6: "24px",
  8: "32px",
  10: "40px",
} as const;

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
    color: colors.accentClassical,
    models: ["logistic", "svm", "random_forest", "decision_tree", "xgboost"],
  },
  deep: {
    label: "Deep Learning",
    color: colors.accentDeep,
    models: ["cnn", "lstm", "transformer"],
  },
  rl: {
    label: "Reinforcement Learning",
    color: colors.accentRl,
    models: ["dqn"],
  },
  ensemble: {
    label: "Ensemble",
    color: colors.accentEnsemble,
    models: ["ensemble_adaptive_regime", "meta_ensemble"],
  },
} as const;

export const modelDescriptions: Record<string, { name: string; short: string; apprentice: string }> = {
  logistic: { name: "Logistic Regression", short: "The Baseline", apprentice: "Fast linear model. Use to establish baseline performance before deploying heavy neural networks." },
  svm: { name: "Support Vector Machine", short: "The Kernel", apprentice: "Kernel-based classifier (RBF). Good for non-linear decision boundaries in feature space." },
  random_forest: { name: "Random Forest", short: "The Forest", apprentice: "Ensemble of decision trees with bagging. Robust, handles mixed features well." },
  decision_tree: { name: "Decision Tree", short: "The Tree", apprentice: "Single decision tree. Highly interpretable but prone to overfitting. Use for feature analysis." },
  xgboost: { name: "XGBoost", short: "The Workhorse", apprentice: "Gradient-boosted trees. Exceptionally good at finding complex non-linear patterns in tabular data." },
  cnn: { name: "CNN", short: "Pattern Scanner", apprentice: "1D convolutional network that learns local price patterns across sliding windows." },
  lstm: { name: "LSTM", short: "Time Traveler", apprentice: "Deep learning network designed for sequential time-series. Remembers price action across hundreds of bars." },
  transformer: { name: "Transformer", short: "Attention Engine", apprentice: "Self-attention architecture that weighs the importance of every historical bar simultaneously." },
  dqn: { name: "Dueling DQN", short: "Autonomous Agent", apprentice: "Reinforcement learning agent. Instead of predicting price, it learns a trading policy through trial and error." },
  ensemble_adaptive_regime: { name: "Adaptive Regime", short: "The Shapeshifter", apprentice: "Dynamically shifts between models depending on market regime (trending vs ranging)." },
  meta_ensemble: { name: "Signal Committee", short: "The Committee", apprentice: "Wraps multiple models and combines their predictions via voting. Run logistic + xgboost + lstm as one." },
};
