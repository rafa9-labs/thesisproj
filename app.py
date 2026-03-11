"""
Forex Trading Engine - Minimalist Streamlit UI

CRITICAL Features:
- Custom CSS removes Streamlit clutter for native-app look
- Real XGBoost predictions (no synthetic data)
- Aggressive caching prevents recalculation on UI interactions
- Float32 equity curves for fast rendering
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure page (MUST be first Streamlit command)
st.set_page_config(
    page_title="Forex Trading Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS: Remove Streamlit clutter for hyper-minimalist design
st.markdown("""
<style>
    /* Hide Streamlit branding and menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Remove top padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Clean metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 600;
        font-family: 'Courier New', monospace;
    }
    
    [data-testid="stMetricDelta"] {
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0E1117;
        padding-top: 1rem;
    }
    
    /* Sidebar text styling - make all text white */
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    
    /* Sidebar headers */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
        font-size: 1.5rem;
        margin-bottom: 1rem;
        color: #FFFFFF !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
        font-size: 1.1rem;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        color: #00D9FF !important;
    }
    
    /* Override ALL sidebar text to be black for input elements */
    [data-testid="stSidebar"] * input,
    [data-testid="stSidebar"] * textarea,
    [data-testid="stSidebar"] * select,
    [data-testid="stSidebar"] [data-baseweb="select"] *,
    [data-testid="stSidebar"] [data-baseweb="date-input"] *,
    [data-testid="stSidebar"] .stSelectbox *,
    [data-testid="stSidebar"] .stDateInput * {
        color: #000000 !important;
    }
    
    /* Ensure input backgrounds are white */
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] select,
    [data-testid="stSidebar"] [data-baseweb="select"],
    [data-testid="stSidebar"] [data-baseweb="date-input"] {
        background-color: #FFFFFF !important;
    }
    
    /* Sidebar paragraph and label text - keep descriptions white */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: #FFFFFF !important;
    }
    
    /* Sidebar input text - make black for visibility */
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] .stSelectbox > div > div > div,
    [data-testid="stSidebar"] .stDateInput > div > div > div {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }
    
    /* More specific selectbox styling */
    [data-testid="stSidebar"] [data-baseweb="select"] {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }
    
    [data-testid="stSidebar"] [data-baseweb="select"] span {
        color: #000000 !important;
    }
    
    [data-testid="stSidebar"] [data-baseweb="select"] div {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }
    
    /* Sidebar selectbox selections - make black text */
    [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] span {
        color: #000000 !important;
    }
    
    [data-testid="stSidebar"] select {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }
    
    /* Sidebar selectbox dropdown options */
    [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stSelectbox"] option {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }
    
    /* Date input styling */
    [data-testid="stSidebar"] [data-testid="stDateInput"] input {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stDateInput"] div[data-baseweb="date-input"] input {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }
    
    /* Main area text inputs - black text */
    [data-testid="stTextArea"] textarea {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }
    
    [data-testid="stTextArea"] div[data-baseweb="textarea"] textarea {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }
    
    /* Text input styling */
    [data-testid="stTextInput"] input {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }
    
    [data-testid="stTextInput"] div[data-baseweb="input"] input {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 4px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button[kind="primary"] {
        background-color: #00D9FF;
        color: #0E1117;
        border: none;
    }
    
    .stButton>button[kind="primary"]:hover {
        background-color: #00B8D9;
        box-shadow: 0 4px 8px rgba(0, 217, 255, 0.3);
    }
    
    /* Checkbox styling */
    [data-testid="stCheckbox"] {
        padding: 0.2rem 0;
    }
    
    /* Expander styling */
    [data-testid="stExpander"] {
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 4px;
        margin-top: 1rem;
    }
    
    /* DataFrame styling */
    [data-testid="stDataFrame"] {
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
    }
    
    /* Title styling */
    h1 {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        color: #FFFFFF;
    }
    
    h2 {
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #E0E0E0;
    }
    
    /* Info/warning boxes */
    [data-testid="stAlert"] {
        border-radius: 4px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

from src.ui import AppState, render_data_tab, render_model_tab, render_dashboard
from src.core.config import AppConfig


def main():
    """Main application entry point with modern tabbed interface"""
    
    try:
        # Initialize state (cached)
        config = AppState.get_config()
        
        # App title and subtitle
        st.title("📊 Forex Trading Engine")
        st.markdown("*Modern ML-powered forex backtesting platform*")
        st.markdown("---")
        
        # Initialize session state for settings persistence
        if 'data_settings' not in st.session_state:
            st.session_state.data_settings = {}
        if 'model_settings' not in st.session_state:
            st.session_state.model_settings = {}
        if 'show_results' not in st.session_state:
            st.session_state.show_results = False
        if 'backtest_results' not in st.session_state:
            st.session_state.backtest_results = None
        
        # Create three-tab interface
        tab1, tab2, tab3 = st.tabs([
            "🏠 Dashboard",
            "⚙️ Config & Setup",
            "🚀 Current Run"
        ])
        
        # Tab 2: Config & Setup (Both Data and Model in one tab)
        with tab2:
            st.header("⚙️ Configuration & Setup")
            st.markdown("Configure your backtest parameters before running")
            
            # Section 1: Data & Features
            with st.expander("📊 Data & Features", expanded=True):
                data_settings = render_data_tab(config)
                st.session_state.data_settings = data_settings
            
            # Section 2: Model & Tuning
            with st.expander("🧠 Model & Tuning", expanded=True):
                model_settings = render_model_tab(
                    config,
                    st.session_state.model_settings.get('model_type', 'XGBoost')
                )
                st.session_state.model_settings = model_settings
            
            st.success("✅ Configuration saved! Switch to the 🚀 Current Run tab to execute.")
        
        # Tab 3: Current Run (Execution with live telemetry)
        with tab3:
            st.header("🚀 Current Run")
            st.markdown("Execute backtest and monitor real-time HPO progress")
            
            # Massive Run Button
            st.markdown("---")
            run_backtest = st.button(
                "🚀 Start Backtest",
                type="primary",
                use_container_width=True,
                help="Train model and execute backtest with current settings"
            )
            
            if run_backtest:
                st.balloons()
                
                # Merge all settings
                settings = {**st.session_state.data_settings, **st.session_state.model_settings}
                
                # Apply WFO configuration with NEW sizing fields
                from src.core.config import WFOConfig
                new_wfo_config = WFOConfig(
                    train_window_type=settings.get('train_window_type', 'expanding'),
                    training_duration=settings.get('training_duration', 2000),
                    test_period_duration=settings.get('test_period_duration', 100),
                    n_mini_folds=settings.get('n_mini_folds', None),
                    hpo_validation_split=settings.get('hpo_validation_split', 0.20),
                    # Legacy fields for backward compatibility
                    train_window_size=settings.get('training_duration', 2000),
                    test_window_size=settings.get('test_period_duration', 100)
                )
                config = config.model_copy(update={'wfo': new_wfo_config})
                
                # Create placeholders for live HPO updates
                st.markdown("### 📡 Live HPO Telemetry")
                
                hpo_progress_container = st.container()
                with hpo_progress_container:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        trial_counter = st.empty()
                    with col2:
                        best_score_metric = st.empty()
                    with col3:
                        current_trial_metric = st.empty()
                    
                    trial_history_table = st.empty()
                    overall_progress = st.progress(0)
                
                # Define Streamlit Optuna callback
                trial_data = []
                
                def streamlit_optuna_callback(study, trial):
                    """Real-time callback for Optuna trials."""
                    n_trials = settings.get('n_trials', 50)
                    
                    # Update trial counter
                    trial_counter.metric(
                        "Trial",
                        f"{trial.number + 1} / {n_trials}"
                    )
                    
                    # Update best score
                    try:
                        best_value = study.best_value
                        # Handle None trial values (failed trials)
                        if trial.value is not None and trial.number > 0:
                            delta = f"{trial.value - best_value:.4f}"
                        else:
                            delta = None
                        
                        best_score_metric.metric(
                            "Best Sharpe",
                            f"{best_value:.4f}",
                            delta=delta
                        )
                    except ValueError:
                        # No trials completed yet, show placeholder
                        best_score_metric.metric(
                            "Best Sharpe",
                            "N/A"
                        )
                    
                    # Update current trial
                    current_trial_metric.metric(
                        "Current Trial Sharpe",
                        f"{trial.value:.4f}" if trial.value is not None else "Failed"
                    )
                    
                    # Append trial data (avoid duplicates)
                    trial_number = trial.number + 1
                    # Remove existing entry for this trial number if present
                    trial_data[:] = [t for t in trial_data if t['Trial'] != trial_number]
                    
                    # Handle trial data safely
                    trial_sharpe = f"{trial.value:.4f}" if trial.value is not None else "Failed"
                    
                    try:
                        best_sharpe = f"{study.best_value:.4f}"
                    except ValueError:
                        best_sharpe = "N/A"
                    
                    trial_data.append({
                        'Trial': trial_number,
                        'Sharpe': trial_sharpe,
                        'Best Sharpe': best_sharpe,
                        'Params': str({k: f"{v:.4f}" if isinstance(v, float) else v 
                                      for k, v in list(trial.params.items())[:3]})
                    })
                    
                    # Update trial history table
                    import pandas as pd
                    trial_df = pd.DataFrame(trial_data)
                    trial_history_table.dataframe(
                        trial_df.tail(10),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Update progress bar
                    progress = (trial.number + 1) / n_trials
                    overall_progress.progress(progress)
                
                # Execute backtest with callback
                st.markdown("---")
                st.markdown("### 📊 Backtest Execution")
                
                # CRITICAL: Capture success flag for strict blocking
                workflow_success = run_backtest_workflow_with_callback(settings, config, streamlit_optuna_callback)
                
                # STRICT BLOCKING: Only render if successful
                if workflow_success:
                    st.session_state.show_results = True
                    
                    st.markdown("---")
                    st.markdown("### 📊 Final Results")
                    
                    results = st.session_state.backtest_results
                    render_dashboard(
                        results['metrics'],
                        results['equity_curve'],
                        results['results_df'],
                        results.get('fold_logs', [])
                    )
                    
                    st.success(f"✅ Backtest complete! Analyzed {len(results['results_df'])} bars with {results['metrics'].get('trades', 0)} trades.")
                else:
                    st.error("❌ Backtest failed. Please check the error message above and try again.")
        
        # Tab 1: Dashboard (Results or Welcome)
        with tab1:
            if st.session_state.show_results and st.session_state.backtest_results is not None:
                # Display cached results
                results = st.session_state.backtest_results
                render_dashboard(
                    results['metrics'],
                    results['equity_curve'],
                    results['results_df'],
                    results.get('fold_logs', None)
                )
                st.success(f"✅ Backtest complete! Analyzed {len(results['results_df'])} bars with {results['metrics'].get('trades', 0)} trades.")
            else:
                # Show welcome screen
                render_welcome_screen()
    
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        st.error(f"❌ Application Error: {str(e)}")
        st.info("Please check the logs for details or try adjusting your settings.")


def run_backtest_workflow_with_callback(settings: dict, config: AppConfig, trial_callback=None):
    """
    Execute complete backtest workflow with optional HPO callback.
    
    NEW: Accepts trial_callback for real-time Optuna telemetry.
    NEW: Returns success flag for strict UI blocking.
    
    Args:
        settings: User settings from UI
        config: Application config with WFO settings
        trial_callback: Optional callback for Optuna trials
        
    Returns:
        bool: True if workflow completed successfully, False otherwise
    """
    try:
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Load data (cached)
        status_text.text("📥 Loading historical data...")
        progress_bar.progress(20)
        
        df_data = AppState.load_historical_data(
            instrument=settings['instrument'],
            start_date=settings['start_date'],
            end_date=settings['end_date'],
            granularity=settings['granularity']
        )
        
        if len(df_data) < 100:
            st.error("❌ Insufficient data. Please select a longer date range.")
            progress_bar.empty()
            status_text.empty()
            return
        
        logger.info(f"Loaded {len(df_data)} bars")
        
        # Step 2: Build features (cached)
        status_text.text("🔧 Building features...")
        progress_bar.progress(40)
        
        df_features, features = AppState.build_features(
            df_data,
            use_rsi=settings['use_rsi'],
            use_macd=settings['use_macd'],
            use_ema=settings['use_ema'],
            use_bbands=settings['use_bbands'],
            use_atr=settings['use_atr'],
            use_adx=settings['use_adx']
        )
        
        if len(features) == 0:
            st.warning("⚠️ No features selected. Using default features.")
        
        logger.info(f"Built {len(features)} features")
        
        # Step 3: Train model and generate predictions (cached)
        # NEW: Show HPO progress if enabled
        if settings.get('enable_hpo', False):
            status_text.text(f"🔍 Running HPO ({settings['hpo_mode']}, {settings['n_trials']} trials)...")
            st.info(
                f"⚙️ **Hyperparameter Optimization in Progress**\n\n"
                f"Mode: `{settings['hpo_mode']}`\n\n"
                f"Trials: `{settings['n_trials']}`\n\n"
                f"Parameters: `{len(settings.get('active_params', []))} selected` "
                f"({'all' if not settings.get('active_params', []) else ', '.join(settings['active_params'][:3]) + '...' if len(settings.get('active_params', [])) > 3 else ', '.join(settings['active_params'])})\n\n"
                f"⏳ This may take several minutes. Please wait..."
            )
        else:
            status_text.text("🤖 Training XGBoost model...")
        
        progress_bar.progress(60)
        
        # CRITICAL: NO CACHE - fresh execution every time
        predictions, probabilities, stitched_indices = AppState.train_and_predict(
            df_features,
            features,
            train_split=settings['train_split'],
            enable_hpo=settings.get('enable_hpo', False),
            hpo_mode=settings.get('hpo_mode', 'single_time'),
            n_trials=settings.get('n_trials', 50),
            active_params=tuple(settings.get('active_params', [])),
            trial_callback=trial_callback
        )
        
        logger.info(f"Generated {len(predictions)} predictions")
        
        # Prepare test data - need original data with required columns, not just features
        split_idx = int(len(df_features) * settings['train_split'])
        
        # Ensure df_data has returns column (it's created in build_features but we need it here)
        if 'returns' not in df_data.columns:
            # Convert price columns to numeric if they're strings
            price_cols = ['close', 'mid_close', 'open', 'mid_open', 'high', 'mid_high', 'low', 'mid_low']
            for col in price_cols:
                if col in df_data.columns:
                    df_data[col] = pd.to_numeric(df_data[col], errors='coerce')
            
            # Create returns column from close price
            if 'close' in df_data.columns:
                df_data['returns'] = df_data['close'].pct_change()
            elif 'mid_close' in df_data.columns:
                df_data['returns'] = df_data['mid_close'].pct_change()
        
        # Get the indices from df_features test period and use them to select corresponding rows from df_data
        test_indices = df_features.iloc[split_idx:].index
        df_test = df_data.loc[test_indices].copy()
        
        # Verify alignment and required columns
        assert len(df_test) == len(df_features.iloc[split_idx:]), f"Data alignment failed: {len(df_test)} vs {len(df_features.iloc[split_idx:])}"
        assert 'returns' in df_test.columns, "Returns column missing from test data"
        
        # DEBUG: Verify prediction arrays before backtest
        with st.expander("🔍 Prediction Array Debug", expanded=False):
            st.write("**Prediction Array Diagnostics:**")
            st.write(f"- Predictions shape: `{predictions.shape}`")
            st.write(f"- Probabilities shape: `{probabilities.shape}`")
            st.write(f"- Unique predictions: `{np.unique(predictions)}`")
            st.write(f"- Prediction distribution: `{np.bincount(predictions + 1)}`")
            st.write(f"- Expected test samples: `{len(df_test)}`")
            st.write(f"- Actual prediction samples: `{len(predictions)}`")
            if len(predictions) == len(df_test):
                st.success("✅ Prediction array length matches test data")
            else:
                st.error(f"❌ Length mismatch: {len(predictions)} predictions vs {len(df_test)} test samples")
        
        # Step 4: Run backtest (cached)
        status_text.text("📊 Running backtest...")
        progress_bar.progress(80)
        
        results_df, metrics, equity_curve = AppState.run_backtest(
            predictions=predictions,
            probabilities=probabilities,
            _df_test=df_test
        )
        
        logger.info(f"Backtest complete: Sharpe={metrics.get('sharpe', 0):.2f}")
        
        # Step 5: Cache results in session state for Dashboard tab
        status_text.text("✅ Caching results...")
        progress_bar.progress(100)
        
        # Store results in session state (fold_logs will be added by state.py)
        st.session_state.backtest_results = {
            'metrics': metrics,
            'equity_curve': equity_curve,
            'results_df': results_df,
            'fold_logs': []  # Will be populated by train_and_predict
        }
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # SUCCESS: Return True (dashboard will render in Current Run tab)
        return True
        
    except Exception as e:
        logger.error(f"Backtest workflow error: {e}", exc_info=True)
        st.error(f"❌ Backtest Error: {str(e)}")
        st.exception(e)  # Show full traceback in UI
        
        # FAILURE: Return False
        return False


def run_backtest_workflow(settings: dict, config: AppConfig = None):
    """
    Execute complete backtest workflow with real XGBoost predictions.
    
    CRITICAL: Uses real model training and predictions, not synthetic data.
    
    Args:
        settings: User settings from sidebar
        config: Application config (optional, will load default if not provided)
    """
    try:
        # Use provided config or load default
        if config is None:
            config = AppState.get_config()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Load data (cached)
        status_text.text("📥 Loading historical data...")
        progress_bar.progress(20)
        
        df_data = AppState.load_historical_data(
            instrument=settings['instrument'],
            start_date=settings['start_date'],
            end_date=settings['end_date'],
            granularity=settings['granularity']
        )
        
        if len(df_data) < 100:
            st.error("❌ Insufficient data. Please select a longer date range.")
            progress_bar.empty()
            status_text.empty()
            return
        
        logger.info(f"Loaded {len(df_data)} bars")
        
        # Step 2: Build features (cached)
        status_text.text("🔧 Building features...")
        progress_bar.progress(40)
        
        df_features, features = AppState.build_features(
            df_data,
            use_rsi=settings['use_rsi'],
            use_macd=settings['use_macd'],
            use_ema=settings['use_ema'],
            use_bbands=settings['use_bbands'],
            use_atr=settings['use_atr'],
            use_adx=settings['use_adx']
        )
        
        if len(features) == 0:
            st.warning("⚠️ No features selected. Using default features.")
        
        logger.info(f"Built {len(features)} features")
        
        # Step 3: Train model and generate predictions (cached)
        # NEW: Show HPO progress if enabled
        if settings.get('enable_hpo', False):
            status_text.text(f"🔍 Running HPO ({settings['hpo_mode']}, {settings['n_trials']} trials)...")
            st.info(
                f"⚙️ **Hyperparameter Optimization in Progress**\n\n"
                f"Mode: `{settings['hpo_mode']}`\n\n"
                f"Trials: `{settings['n_trials']}`\n\n"
                f"Parameters: `{len(settings.get('active_params', []))} selected` "
                f"({'all' if not settings.get('active_params', []) else ', '.join(settings['active_params'][:3]) + '...' if len(settings.get('active_params', [])) > 3 else ', '.join(settings['active_params'])})\n\n"
                f"⏳ This may take several minutes. Please wait..."
            )
        else:
            status_text.text("🤖 Training XGBoost model...")
        
        progress_bar.progress(60)
        
        # CRITICAL: Pass HPO params for cache busting
        predictions, probabilities = AppState.train_and_predict(
            df_features,
            features,
            train_split=settings['train_split'],
            enable_hpo=settings.get('enable_hpo', False),
            hpo_mode=settings.get('hpo_mode', 'single_time'),
            n_trials=settings.get('n_trials', 50),
            active_params=tuple(settings.get('active_params', []))  # Convert to tuple for hashability
        )
        
        logger.info(f"Generated {len(predictions)} predictions")
        
        # Prepare test data
        split_idx = int(len(df_features) * settings['train_split'])
        df_test = df_features.iloc[split_idx:].copy()
        
        # Step 4: Run backtest (cached)
        status_text.text("📊 Running backtest...")
        progress_bar.progress(80)
        
        results_df, metrics, equity_curve = AppState.run_backtest(
            predictions=predictions,
            probabilities=probabilities,
            _df_test=df_test
        )
        
        logger.info(f"Backtest complete: Sharpe={metrics.get('sharpe', 0):.2f}")
        
        # Step 5: Cache results in session state for Dashboard tab
        status_text.text("✅ Caching results...")
        progress_bar.progress(100)
        
        # Store results in session state (fold_logs will be added by state.py)
        st.session_state.backtest_results = {
            'metrics': metrics,
            'equity_curve': equity_curve,
            'results_df': results_df,
            'fold_logs': []  # Will be populated by train_and_predict
        }
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Display results immediately
        render_dashboard(metrics, equity_curve, results_df, [])
        
        # Success message
        st.success(f"✅ Backtest complete! Analyzed {len(results_df)} bars with {metrics.get('trades', 0)} trades.")
        
    except Exception as e:
        logger.error(f"Backtest workflow error: {e}", exc_info=True)
        st.error(f"❌ Backtest Error: {str(e)}")
        st.info("Please try adjusting your settings or check the data availability.")


def render_welcome_screen():
    """Render welcome screen when no backtest is running"""
    
    st.title("📊 Forex Trading Engine")
    
    st.markdown("""
    ### Welcome to the Institutional-Grade Trading Platform
    
    Built with a complete modular architecture across 5 phases:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Backend (Phases 1-4)**
        - ✅ **Phase 1**: Configuration & Data Loading
        - ✅ **Phase 2**: Feature Engineering Pipeline
        - ✅ **Phase 3**: Model Integration Layer
        - ✅ **Phase 4**: Execution & Backtesting Engine
        """)
    
    with col2:
        st.markdown("""
        **Key Features**
        - 🚀 Vectorized backtesting (10-100x faster)
        - 🎯 Execution delay enforcement (no look-ahead bias)
        - 💾 Float32 equity curves (memory-efficient)
        - 📊 16 standard metrics (HAC-adjusted Sharpe)
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### Getting Started
    
    1. **Configure Settings** 👈 in the sidebar
       - Select currency pair and date range
       - Choose technical indicators (RSI, MACD, etc.)
       - Adjust train/test split
    
    2. **Run Backtest** 🚀
       - Click the primary action button
       - XGBoost model trains on 80% of data
       - Predictions generated on remaining 20%
       - Full backtest with cost modeling
    
    3. **Analyze Results** 📈
       - Holy Trinity metrics (Sharpe, Drawdown, Win Rate)
       - Interactive equity curve (Plotly)
       - Detailed performance metrics
       - Trade statistics and logs
    """)
    
    st.info("👈 **Configure settings in the sidebar to begin**")
    
    # Technical details expander
    with st.expander("🔧 Technical Architecture"):
        st.markdown("""
        **State Management**
        - `@st.cache_resource`: Singleton objects (config, factories, pipelines)
        - `@st.cache_data`: Data loading and computation (1-hour TTL)
        - **No recalculation** on button clicks or UI interactions
        
        **Performance Optimization**
        - Vectorized NumPy operations (fast-path)
        - Float32 equity curves (50% memory savings)
        - Aggressive caching at every layer
        - Lazy loading of heavy components
        
        **Model Training**
        - Real XGBoost predictions (no synthetic data)
        - 80/20 train/test split (configurable)
        - Execution delay enforcement (signal at t executes at t+1)
        - Cost modeling (spread, slippage, commission)
        
        **Metrics**
        - 16 standard trading metrics
        - HAC-adjusted Sharpe ratio (Newey-West)
        - Reliability guards (min trades threshold)
        - Classification metrics (precision, F1, accuracy)
        """)
    
    # Sample workflow
    with st.expander("📋 Sample Workflow"):
        st.code("""
# Complete pipeline execution:

1. Load Data (cached)
   → EUR_USD H1 bars from 2023-01-01 to 2023-12-31

2. Build Features (cached)
   → RSI, MACD, EMA, Bollinger Bands, ATR
   → Multi-timeframe alignment
   → Regime classification

3. Train Model (cached)
   → XGBoost on first 80% of data
   → 100 estimators, max_depth=5

4. Generate Predictions (cached)
   → Predict on remaining 20%
   → Output: {-1, 0, 1} signals + probabilities

5. Run Backtest (cached)
   → Vectorized execution (fast-path)
   → Cost modeling (spread, slippage)
   → Execution delay enforcement

6. Display Results
   → Holy Trinity: Sharpe, Drawdown, Win Rate
   → Interactive equity curve (float32)
   → Detailed metrics and trade log
        """, language="python")


if __name__ == "__main__":
    main()
