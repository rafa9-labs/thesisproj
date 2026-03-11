# Phase 3: Model Integration Layer ✅

**Status**: Complete  
**Date**: March 7, 2026

---

## 🎯 Objectives Achieved

✅ **BaseStrategy Interface**: Abstract base class with standardized API  
✅ **Model Wrappers**: XGBoost, CNN, LSTM with strict type hints  
✅ **Ensemble Voting**: Parallel training with weighted averaging  
✅ **Calibration Engine**: Temperature scaling, isotonic, conformal prediction  
✅ **ModelTrainer**: WFO orchestration with leakage prevention  
✅ **Memory Cleanup**: `_cleanup()` method for GPU/RAM management  
✅ **Type Strictness**: `np.ndarray` and `pd.DataFrame` type hints throughout  
✅ **Divide & Conquer**: New modules only, originals untouched

---

## 📁 Project Structure

```
src/models/
├── __init__.py              # Package exports
├── base.py                  # BaseStrategy abstract class - 250 lines
├── calibration.py           # Temperature scaling & conformal - 380 lines
├── xgboost_wrapper.py       # XGBoost model wrapper - 240 lines
├── cnn_wrapper.py           # CNN model wrapper - 270 lines
├── lstm_wrapper.py          # LSTM model wrapper - 270 lines
├── ensemble_wrapper.py      # Ensemble voting wrapper - 340 lines
└── trainer.py               # WFO orchestration - 480 lines

example_phase3_usage.py      # Usage examples
PHASE3_README.md            # This file
```

**Total**: ~2,230 lines of modular, type-safe model integration code

---

## 🔧 Core Components

### 1. BaseStrategy (`src/models/base.py`)

**Abstract base class for all models.**

```python
from src.models import BaseStrategy

class MyStrategy(BaseStrategy):
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'MyStrategy':
        # Train model
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Predict labels
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Predict probabilities (float32)
        return probabilities
    
    def save(self, path: str) -> None:
        # Save model to disk
        pass
    
    def load(self, path: str) -> 'MyStrategy':
        # Load model from disk
        return self
    
    def _cleanup(self) -> None:
        # Clean up GPU/RAM memory
        super()._cleanup()
```

**Key Features:**
- Standardized interface across all models
- `_cleanup()` method for memory management
- `is_fitted()` status checking
- `get_params()` / `set_params()` for configuration

---

### 2. Model Wrappers

#### **XGBoostStrategy** (`src/models/xgboost_wrapper.py`)

```python
from src.models import XGBoostStrategy

config = {
    'n_estimators': 400,
    'max_depth': 6,
    'learning_rate': 0.1,
    'use_gpu': False,
    'xgb_early_stopping_rounds': 50
}

strategy = XGBoostStrategy(config)
strategy.fit(X_train, y_train, eval_set=(X_val, y_val))

proba = strategy.predict_proba(X_test)  # Returns float32
```

**Features:**
- GPU/CPU automatic fallback
- Early stopping support
- Parameter filtering (`xgb_` prefix or unprefixed)
- Feature importance extraction

---

#### **CNNStrategy** (`src/models/cnn_wrapper.py`)

```python
from src.models import CNNStrategy

config = {
    'cnn_filters1': 32,
    'cnn_filters2': 64,
    'cnn_kernel_size': 3,
    'cnn_dense_units': 64,
    'cnn_dropout_rate': 0.3,
    'cnn_learning_rate': 0.001,
    'cnn_epochs': 50,
    'cnn_batch_size': 32,
    'cnn_use_early_stopping': True,
    'cnn_patience': 10
}

strategy = CNNStrategy(config)
strategy.fit(X_train, y_train, validation_split=0.1)

# CRITICAL: Cleanup TensorFlow session
strategy._cleanup()
```

**Features:**
- Automatic windowing for 2D input
- StandardScaler fitted on training data only
- Early stopping callback
- TensorFlow session cleanup via `_cleanup()`

---

#### **LSTMStrategy** (`src/models/lstm_wrapper.py`)

```python
from src.models import LSTMStrategy

config = {
    'lstm_units': 64,
    'lstm_dense_units': 64,
    'lstm_dropout_rate': 0.3,
    'lstm_learning_rate': 0.001,
    'lstm_num_layers': 1,
    'lstm_bidirectional': False,
    'lstm_epochs': 50,
    'lstm_batch_size': 32
}

strategy = LSTMStrategy(config)
strategy.fit(X_train, y_train, validation_split=0.1)

strategy._cleanup()  # CRITICAL
```

**Features:**
- Bidirectional LSTM support
- Automatic windowing
- StandardScaler integration
- TensorFlow cleanup

---

#### **EnsembleStrategy** (`src/models/ensemble_wrapper.py`)

```python
from src.models import EnsembleStrategy, CNNStrategy, LSTMStrategy, XGBoostStrategy

# Create base models
cnn = CNNStrategy(cnn_config)
lstm = LSTMStrategy(lstm_config)
xgb = XGBoostStrategy(xgb_config)

# Create ensemble
ensemble_config = {
    'voting_method': 'soft',  # or 'hard'
    'learn_weights': True,
    'max_workers': 3
}

ensemble = EnsembleStrategy(ensemble_config, models=[cnn, lstm, xgb])

# Train in parallel
ensemble.fit(X_train, y_train, parallel=True)

# Predict with weighted voting
proba = ensemble.predict_proba(X_test)

# CRITICAL: Cleans up ALL sub-models
ensemble._cleanup()
```

**Features:**
- Parallel training with ThreadPoolExecutor
- Soft voting (weighted probability averaging)
- Hard voting (majority vote)
- Automatic weight learning via meta-learner
- Cleanup for all sub-models

---

### 3. Calibration (`src/models/calibration.py`)

#### **Temperature Scaling**

```python
from src.models import TemperatureScaling

calibrator = TemperatureScaling()
calibrator.fit(proba_cal, y_cal)

proba_calibrated = calibrator.transform(proba_test)

print(f"Temperature: {calibrator.temperature:.3f}")
```

#### **Isotonic Calibration**

```python
from src.models import IsotonicCalibrator

calibrator = IsotonicCalibrator(method='isotonic')  # or 'sigmoid'
calibrator.fit(proba_cal, y_cal)

proba_calibrated = calibrator.transform(proba_test)
```

#### **Conformal Prediction**

```python
from src.models import ConformalPredictor

conformal = ConformalPredictor(alpha=0.1)  # 90% coverage
conformal.fit(proba_cal, y_cal)

# Get prediction sets with coverage guarantee
prediction_sets = conformal.predict_sets(proba_test)
```

---

### 4. ModelTrainer (`src/models/trainer.py`)

**Main orchestration class with WFO and leakage prevention.**

```python
from src.core.config import load_default_config
from src.data.factory import DataFactory
from src.features.pipeline import FeaturePipeline
from src.models import ModelTrainer, XGBoostStrategy

config = load_default_config()
factory = DataFactory(config)
pipeline = FeaturePipeline(config, factory)
trainer = ModelTrainer(config, pipeline, factory)

strategy = XGBoostStrategy(xgb_config)

# Walk-Forward Optimization
trained_strategy, metrics = trainer.train_with_wfo(
    strategy=strategy,
    train_start='2023-01-01',
    train_end='2023-06-30',
    test_start='2023-07-01',
    test_end='2023-12-31',
    calibrate=True
)

print(f"Accuracy: {metrics['accuracy']:.4f}")
```

**CRITICAL Leakage Prevention:**
```python
# ✓ CORRECT: Scaler fitted ONLY on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # NO fit on test!

# ✓ CORRECT: Calibrator fitted ONLY on validation split
calibrator.fit(proba_cal, y_cal)  # Cal set from training period
proba_test_calibrated = calibrator.transform(proba_test)

# ✗ WRONG: Would cause leakage
# scaler.fit(np.vstack([X_train, X_test]))  # NEVER DO THIS
```

---

## 🚀 Quick Start

### Installation

Dependencies from Phase 1 & 2 already installed:
```bash
pip install -r requirements.txt
```

Additional dependencies:
```bash
pip install optuna scipy
```

### Run Examples

```bash
python example_phase3_usage.py
```

This demonstrates:
1. ✅ XGBoost training with calibration
2. ✅ CNN training with early stopping
3. ✅ Ensemble voting (CNN + LSTM + XGB)
4. ✅ Walk-Forward Optimization
5. ✅ Time-series cross-validation
6. ✅ Hyperparameter optimization
7. ✅ Calibration methods comparison
8. ✅ Memory cleanup importance

---

## 🎨 Design Principles Applied

### ✅ **Leakage Prevention**

**CRITICAL: All data-dependent transformations fitted ONLY on training data.**

```python
# In trainer.py train_with_wfo():

# 1) Load data for periods
df_train = load_data(train_start, train_end)
df_test = load_data(test_start, test_end)

# 2) Build features SEPARATELY
df_train_feat, _ = pipeline.build_features(df_train)
df_test_feat, _ = pipeline.build_features(df_test)

# 3) LEAKAGE PREVENTION: Fit scaler ONLY on train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit on train
X_test_scaled = scaler.transform(X_test)        # transform only

# 4) Split train for calibration
X_train_fit = X_train_scaled[:-n_cal]
X_cal = X_train_scaled[-n_cal:]  # From TRAINING period

# 5) Calibrator fitted ONLY on cal set (from training)
calibrator.fit(proba_cal, y_cal)
```

---

### ✅ **Memory Guard**

**CRITICAL: `_cleanup()` called after EVERY trial/fold.**

```python
# In trainer.py optimize_hyperparameters():

def objective(trial):
    strategy = StrategyClass(params)
    
    try:
        strategy.fit(X_train, y_train)
        score = evaluate(strategy, X_val, y_val)
    finally:
        # CRITICAL: Cleanup after EVERY trial
        strategy._cleanup()
    
    return score
```

**Cleanup hierarchy:**
```python
# BaseStrategy._cleanup()
gc.collect()

# DeepLearningStrategy._cleanup()
tf.keras.backend.clear_session()  # TensorFlow
super()._cleanup()                 # Parent cleanup

# EnsembleStrategy._cleanup()
for model in self.models:
    model._cleanup()  # Cleanup all sub-models
super()._cleanup()
```

---

### ✅ **Type Strictness**

**All functions use strict `np.ndarray` and `pd.DataFrame` type hints.**

```python
def fit(
    self,
    X: np.ndarray,  # Strict type hint
    y: np.ndarray,  # Strict type hint
    **kwargs
) -> 'BaseStrategy':
    """Train model"""
    pass

def predict_proba(self, X: np.ndarray) -> np.ndarray:
    """Returns float32 probabilities"""
    proba = self.model.predict_proba(X)
    return proba.astype(np.float32)  # Explicit float32
```

**Benefits:**
- IDE autocomplete and type checking
- Prevents DataFrame/array confusion
- Ensures FeaturePipeline output matches model input
- float32 for memory efficiency

---

## 📊 Architecture Achievement

**Before**: Monolithic training logic embedded in MLBacktester  
**After**: Modular, testable, reusable model wrappers

```
MLBacktesterNoWFO.py (19,451 lines)
├── get_model() [10231-10600] (370 lines)
├── test_strategy() [4983-5700] (700+ lines)
├── Calibration logic scattered throughout
    └── Extracted to:
        ├── base.py (250 lines) - BaseStrategy
        ├── xgboost_wrapper.py (240 lines)
        ├── cnn_wrapper.py (270 lines)
        ├── lstm_wrapper.py (270 lines)
        ├── ensemble_wrapper.py (340 lines)
        ├── calibration.py (380 lines)
        └── trainer.py (480 lines)
```

---

## 🔒 Critical Technical Details

### **1. Leakage Prevention in WFO**

```python
# CORRECT WFO fold structure:
for fold in wfo_folds:
    # Load data for THIS fold only
    df_train = load_data(fold.train_start, fold.train_end)
    df_test = load_data(fold.test_start, fold.test_end)
    
    # Build features SEPARATELY
    df_train_feat, _ = pipeline.build_features(df_train)
    df_test_feat, _ = pipeline.build_features(df_test)
    
    # Fit scaler ONLY on train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train and evaluate
    strategy.fit(X_train_scaled, y_train)
    metrics = evaluate(strategy, X_test_scaled, y_test)
    
    # CRITICAL: Cleanup
    strategy._cleanup()
```

---

### **2. Memory Cleanup Hierarchy**

```python
# Classical ML (XGBoost, etc.)
class XGBoostStrategy(ClassicalMLStrategy):
    def _cleanup(self):
        gc.collect()  # Python garbage collection

# Deep Learning (CNN, LSTM)
class CNNStrategy(DeepLearningStrategy):
    def _cleanup(self):
        tf.keras.backend.clear_session()  # TensorFlow
        super()._cleanup()                 # gc.collect()

# Ensemble
class EnsembleStrategy(BaseStrategy):
    def _cleanup(self):
        for model in self.models:
            model._cleanup()  # Cleanup all sub-models
        super()._cleanup()
```

---

### **3. Type Safety**

```python
# All model methods return float32 probabilities
def predict_proba(self, X: np.ndarray) -> np.ndarray:
    proba = self.model.predict_proba(X)
    return proba.astype(np.float32)  # Explicit cast

# FeaturePipeline output matches model input
df_features, features = pipeline.build_features(df)
X = df_features[features].values.astype(np.float32)

# Type hints prevent errors
strategy.fit(X, y)  # X must be np.ndarray
```

---

## 📝 Usage Patterns

### **Pattern 1: Single Model Training**

```python
from src.models import XGBoostStrategy

config = {'n_estimators': 100, 'max_depth': 4}
strategy = XGBoostStrategy(config)
strategy.fit(X_train, y_train)
proba = strategy.predict_proba(X_test)
strategy._cleanup()
```

---

### **Pattern 2: WFO with Calibration**

```python
from src.models import ModelTrainer, XGBoostStrategy

trainer = ModelTrainer(config, pipeline, factory)
strategy = XGBoostStrategy(xgb_config)

trained_strategy, metrics = trainer.train_with_wfo(
    strategy=strategy,
    train_start='2023-01-01',
    train_end='2023-06-30',
    test_start='2023-07-01',
    test_end='2023-12-31',
    calibrate=True  # Temperature scaling
)
```

---

### **Pattern 3: Ensemble with Parallel Training**

```python
from src.models import EnsembleStrategy, CNNStrategy, LSTMStrategy

cnn = CNNStrategy(cnn_config)
lstm = LSTMStrategy(lstm_config)

ensemble = EnsembleStrategy(
    config={'voting_method': 'soft', 'max_workers': 2},
    models=[cnn, lstm]
)

ensemble.fit(X_train, y_train, parallel=True)
proba = ensemble.predict_proba(X_test)
ensemble._cleanup()  # Cleans up ALL models
```

---

### **Pattern 4: Hyperparameter Optimization**

```python
from src.models import ModelTrainer, XGBoostStrategy

search_space = {
    'n_estimators': {'type': 'int', 'low': 50, 'high': 200},
    'max_depth': {'type': 'int', 'low': 3, 'high': 8},
    'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True}
}

trainer = ModelTrainer(config, pipeline)
best_strategy, best_params = trainer.optimize_hyperparameters(
    strategy_class=XGBoostStrategy,
    search_space=search_space,
    X=X,
    y=y,
    n_trials=100
)
```

---

## 🔄 Backward Compatibility

Phase 3 is **standalone and non-breaking**:
- Original `MLBacktesterNoWFO.py` untouched
- New modular code in `src/models/`
- Can gradually migrate MLBacktester to use new wrappers
- Both systems can coexist during transition

---

## ✅ Requirements Met

| Requirement | Status | Implementation |
|------------|--------|----------------|
| BaseStrategy interface | ✅ | Abstract class with fit/predict/save/load |
| XGBoost wrapper | ✅ | GPU/CPU support, early stopping |
| CNN wrapper | ✅ | Windowing, scaling, TF cleanup |
| LSTM wrapper | ✅ | Bidirectional support, TF cleanup |
| Ensemble wrapper | ✅ | Parallel training, weighted voting |
| Calibration engine | ✅ | Temperature, isotonic, conformal |
| ModelTrainer | ✅ | WFO, CV, HPO orchestration |
| Leakage prevention | ✅ | Scalers fitted ONLY on train data |
| Memory guard | ✅ | `_cleanup()` after every trial/fold |
| Type strictness | ✅ | `np.ndarray` type hints throughout |
| Divide & Conquer | ✅ | New files only, originals untouched |

---

## 🔜 Integration with Phases 1 & 2

**Complete End-to-End Pipeline:**

```python
# Phase 1: Configuration & Data
from src.core.config import load_default_config
from src.data.factory import DataFactory

config = load_default_config()
factory = DataFactory(config)

# Phase 2: Feature Engineering
from src.features.pipeline import FeaturePipeline

pipeline = FeaturePipeline(config, factory)
df_features, features = pipeline.build_features(df)

# Phase 3: Model Training
from src.models import ModelTrainer, XGBoostStrategy

trainer = ModelTrainer(config, pipeline, factory)
strategy = XGBoostStrategy(xgb_config)

trained_strategy, metrics = trainer.train_with_wfo(
    strategy=strategy,
    train_start='2023-01-01',
    train_end='2023-06-30',
    test_start='2023-07-01',
    test_end='2023-12-31',
    calibrate=True
)

# Complete modular pipeline!
```

---

**Phase 3 Complete** ✅  
Your forex bot now has a complete modular architecture: Config → Data → Features → Models

**Next Steps:**
- Integrate with existing MLBacktester
- Run full backtests with new wrappers
- Deploy to production with confidence
- Phase 4: UI/API Layer (optional)

---

## 🙏 Acknowledgments

Extracted from:
- `MLBacktesterNoWFO.py` lines 10231-10600 (model construction)
- `MLBacktesterNoWFO.py` lines 4983-5700 (training logic)
- `MLBacktesterNoWFO.py` lines 4027-4106 (calibration)
- `utilsNoWFO.py` lines 4360-4390 (calibration utilities)
- `tuningNoWFO.py` (Optuna HPO logic)
- `models/cnn.py`, `models/lstm.py`, `models/ensemble_cnn_lstm_xgboost.py`
