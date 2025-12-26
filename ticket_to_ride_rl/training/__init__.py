"""
Training infrastructure for Ticket to Ride RL.
"""
from .metrics import MetricsCollector, GameMetrics, GameLogger, compute_metrics_from_game
from .self_play import SelfPlayManager, SelfPlayGame, PlayerSetup
from .trainer import Trainer, TrainingConfig, run_training
from .parallel import (
    ParallelGameRunner, StrategyTracker, GameResult,
    get_optimal_workers, print_strategy_evolution
)
from .parallel_trainer import (
    ParallelTrainer, ParallelTrainingConfig, run_parallel_training
)

__all__ = [
    'MetricsCollector', 'GameMetrics', 'GameLogger', 'compute_metrics_from_game',
    'SelfPlayManager', 'SelfPlayGame', 'PlayerSetup',
    'Trainer', 'TrainingConfig', 'run_training',
    'ParallelGameRunner', 'StrategyTracker', 'GameResult',
    'get_optimal_workers', 'print_strategy_evolution',
    'ParallelTrainer', 'ParallelTrainingConfig', 'run_parallel_training',
]
