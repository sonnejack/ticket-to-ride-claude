"""
Ticket to Ride Reinforcement Learning System

A complete RL training pipeline to discover optimal strategies for
playing Ticket to Ride (USA map).

Usage:
    python -m ticket_to_ride_rl.main --games 50000 --output results/
"""

__version__ = "1.0.0"
__author__ = "TTR RL Project"

from .game import GameState, GameConfig, Board, Player
from .agents import PPOAgent, PPOConfig, ArchetypeType
from .training import Trainer, TrainingConfig
from .analysis import analyze_results, create_visualizations
