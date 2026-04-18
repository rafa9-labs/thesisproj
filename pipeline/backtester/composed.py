"""MLBacktester composed from modular mixins."""
from __future__ import annotations
from pipeline._imports import *  # noqa: F401,F403
from pipeline.backtester.core_mixin import CoreMixin
from pipeline.backtester.data_mixin import DataMixin
from pipeline.backtester.features_mixin import FeaturesMixin
from pipeline.backtester.deep_mixin import DeepMixin
from pipeline.backtester.strategy_mixin import StrategyMixin
from pipeline.backtester.ensemble_mixin import EnsembleMixin
from pipeline.backtester.dqn_mixin import DQNMixin
from pipeline.backtester.model_factory_mixin import ModelFactoryMixin
from pipeline.backtester.evaluation_mixin import EvaluationMixin
from pipeline.backtester.run_mixin import RunMixin
from pipeline.backtester.real_trading_mixin import RealTradingMixin


class MLBacktester(
    CoreMixin,
    DataMixin,
    FeaturesMixin,
    DeepMixin,
    StrategyMixin,
    EnsembleMixin,
    DQNMixin,
    ModelFactoryMixin,
    EvaluationMixin,
    RunMixin,
    RealTradingMixin,
):
    """MLBacktester - composed from modular mixins."""
    pass