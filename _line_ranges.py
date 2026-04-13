"""Extract line ranges of key functions from utilsNoWFO.py."""
import ast
tree = ast.parse(open('utilsNoWFO.py', 'r', encoding='utf-8').read())
targets = {
    'build_features_from_params', 'triple_barrier_labels', 'attach_macro_features',
    'select_topk_by_mutual_info', 'prefilter_features_train', 'drop_near_constant_features',
    'drop_high_corr_features', 'ConformalClassifier', 'CostAwareWrapper',
    'RewardProcessWrapper', 'RollingStandardizer', 'calibrate_prefit_and_predict_proba',
    'fit_coverage_threshold_on_calibration', 'apply_temperature_to_proba',
    'fit_temperature_from_proba', 'probabilistic_sharpe_ratio', 'compute_dsr_scores',
    'first_tradable_test_bar', 'compute_required_test_warmup_bars', 'predict_decisions',
    'realized_vol', 'compute_rolling_hit_rate', 'compute_rolling_sharpe_series',
    'compute_drawdown_curve', 'compute_brier_and_nll', 'sanitize_proba',
    '_cliffs_delta', 'enforce_day1_eval_anchor', 'close_trade',
    'build_trade_log_from_df', 'find_hit_rate_switch_idx', 'fracdiff',
    'add_cyclic_hour_features', 'step', 'reset', 'add', 'add_feat',
    'transform', 'fit_transform',
}
for node in ast.walk(tree):
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        if node.name in targets:
            print(f"{node.name:50s} L{node.lineno:5d}-{node.end_lineno:5d}")