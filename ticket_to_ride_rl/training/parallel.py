"""
Parallel game simulation for faster training.
Automatically scales to available CPU cores.
"""
import os
import multiprocessing as mp
from multiprocessing import Pool, Queue, Manager
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
import torch

from ticket_to_ride_rl.game import GameState, GameConfig, GamePhase
from ticket_to_ride_rl.agents import (
    PPOAgent, PPOConfig, ArchetypeType, get_archetype_types, get_archetype_config
)
from .self_play import PlayerSetup
from .metrics import GameMetrics, compute_metrics_from_game


def get_optimal_workers(requested: int = None) -> int:
    """
    Determine optimal number of worker processes.
    Leaves 1-2 cores free for the main process and OS.
    """
    available = mp.cpu_count()

    if requested is not None:
        return min(requested, available - 1)

    if available <= 4:
        return max(1, available - 1)
    elif available <= 8:
        return available - 2
    else:
        # For 30+ cores, use most of them
        return available - 2


@dataclass
class GameResult:
    """Result from a single game simulation."""
    winner: int
    scores: List[int]
    archetypes: List[str]
    tracking: List[bool]
    turn_count: int
    hand_histories: List[List[int]]
    action_counts: Dict[str, Dict[str, int]]  # Per player, per action type


def _play_single_game(args) -> GameResult:
    """
    Play a single game in a worker process.
    This function runs in a separate process.
    """
    game_id, num_players, archetype_names, use_tracking_list, obs_dim, ppo_config_dict = args

    # Recreate PPO config
    ppo_config = PPOConfig(**ppo_config_dict)

    # Create agents for this game
    players = []
    agents = []

    for i, (arch_name, use_tracking) in enumerate(zip(archetype_names, use_tracking_list)):
        archetype = ArchetypeType(arch_name)
        agent = PPOAgent(
            obs_dim=obs_dim,
            config=ppo_config,
            archetype=archetype,
            use_tracking=use_tracking,
            device='cpu',
            num_players=num_players
        )
        agents.append(agent)
        players.append(PlayerSetup(
            agent=agent,
            archetype=archetype,
            use_tracking=use_tracking
        ))

    # Initialize game
    config = GameConfig(num_players=num_players)
    game = GameState(config)
    game.setup_game()

    # Initialize agents
    for i, agent in enumerate(agents):
        agent.init_tracker(num_players, i)
        agent.reset_episode()

    # Handle initial destinations
    for i, player_setup in enumerate(players):
        archetype_config = get_archetype_config(player_setup.archetype)
        cards = game.deal_initial_destinations(i)

        game_min = game.config.min_destinations_keep
        n_keep = min(max(archetype_config.min_destinations_keep, game_min), len(cards))

        policy = player_setup.agent.archetype_policy
        if policy and cards:
            scores = policy.get_destination_keep_preferences(cards, game, i)
            sorted_indices = np.argsort(scores)[::-1]
            keep_ids = [cards[idx].card_id for idx in sorted_indices[:n_keep]]
        else:
            keep_ids = [c.card_id for c in cards[:n_keep]]

        if keep_ids:
            game.keep_initial_destinations(i, keep_ids)

    game.phase = GamePhase.PLAYING

    # Track hand sizes and action counts
    hand_histories = [[] for _ in range(num_players)]
    action_counts = {f"player_{i}": {} for i in range(num_players)}

    # Play game
    max_turns = 500
    turn_count = 0

    while not game.is_game_over() and turn_count < max_turns:
        current_player = game.current_player
        agent = agents[current_player]

        # Record hand size
        hand_histories[current_player].append(
            game.players[current_player].hand.total()
        )

        # Get valid actions
        valid_actions = game.get_valid_actions()
        if not valid_actions:
            game.end_turn()
            turn_count += 1
            continue

        # Select action
        action_idx, action, log_prob, value = agent.select_action(
            game, current_player, valid_actions
        )

        # Track action type
        action_type = action.action_type.name
        player_key = f"player_{current_player}"
        action_counts[player_key][action_type] = action_counts[player_key].get(action_type, 0) + 1

        # Execute action
        turn_complete, info = game.execute_action(action)

        if turn_complete:
            game.end_turn()
            turn_count += 1

    # Get results
    winner, _ = game.get_winner()
    final_scores = game.get_final_scores()
    scores = [0] * num_players
    for idx, score, _ in final_scores:
        scores[idx] = score

    return GameResult(
        winner=winner,
        scores=scores,
        archetypes=archetype_names,
        tracking=use_tracking_list,
        turn_count=turn_count,
        hand_histories=hand_histories,
        action_counts=action_counts
    )


class ParallelGameRunner:
    """
    Runs multiple games in parallel using multiprocessing.
    """

    def __init__(self, num_players: int = 4, num_workers: int = None,
                 obs_dim: int = 598, ppo_config: PPOConfig = None):
        self.num_players = num_players
        self.num_workers = get_optimal_workers(num_workers)
        self.obs_dim = obs_dim
        self.ppo_config = ppo_config or PPOConfig()

        # Convert config to dict for pickling
        self.ppo_config_dict = {
            'learning_rate': self.ppo_config.learning_rate,
            'lr_decay': self.ppo_config.lr_decay,
            'min_lr': self.ppo_config.min_lr,
            'clip_epsilon': self.ppo_config.clip_epsilon,
            'value_loss_coef': self.ppo_config.value_loss_coef,
            'entropy_coef': self.ppo_config.entropy_coef,
            'max_grad_norm': self.ppo_config.max_grad_norm,
            'gamma': self.ppo_config.gamma,
            'gae_lambda': self.ppo_config.gae_lambda,
            'batch_size': self.ppo_config.batch_size,
            'num_epochs': self.ppo_config.num_epochs,
            'num_minibatches': self.ppo_config.num_minibatches,
            'hidden_dim': self.ppo_config.hidden_dim,
            'use_lstm': self.ppo_config.use_lstm,
            'lstm_dim': self.ppo_config.lstm_dim,
            'max_actions': self.ppo_config.max_actions,
        }

        self.game_counter = 0

    def _create_game_args(self, num_games: int) -> List[tuple]:
        """Create arguments for parallel game execution."""
        args_list = []
        archetypes = list(get_archetype_types())

        for i in range(num_games):
            # Cycle through archetypes
            game_archetypes = []
            use_tracking = []

            for j in range(self.num_players):
                arch = archetypes[(i + j) % len(archetypes)]
                game_archetypes.append(arch.value)
                use_tracking.append(j % 2 == 0)

            # Shuffle for variety
            combined = list(zip(game_archetypes, use_tracking))
            np.random.shuffle(combined)
            game_archetypes, use_tracking = zip(*combined)

            args_list.append((
                self.game_counter + i,
                self.num_players,
                list(game_archetypes),
                list(use_tracking),
                self.obs_dim,
                self.ppo_config_dict
            ))

        return args_list

    def run_games(self, num_games: int) -> List[GameResult]:
        """Run multiple games in parallel."""
        args_list = self._create_game_args(num_games)

        # Use context to ensure clean process creation
        ctx = mp.get_context('spawn')

        with ctx.Pool(processes=self.num_workers) as pool:
            results = pool.map(_play_single_game, args_list)

        self.game_counter += num_games
        return results


class StrategyTracker:
    """
    Tracks how strategies evolve over training.
    Records action distributions and key metrics at intervals.
    """

    def __init__(self):
        self.snapshots: List[Dict] = []
        self.action_history: Dict[str, List[Dict]] = {}  # Per archetype

        for arch in get_archetype_types():
            self.action_history[arch.value] = []

    def record_snapshot(self, games_played: int, results: List[GameResult]):
        """Record a snapshot of current strategy statistics."""
        # Aggregate action distributions per archetype
        archetype_actions: Dict[str, Dict[str, int]] = {}
        archetype_wins: Dict[str, int] = {}
        archetype_games: Dict[str, int] = {}

        for result in results:
            for i, arch in enumerate(result.archetypes):
                if arch not in archetype_actions:
                    archetype_actions[arch] = {}
                    archetype_wins[arch] = 0
                    archetype_games[arch] = 0

                # Merge action counts
                player_actions = result.action_counts.get(f"player_{i}", {})
                for action, count in player_actions.items():
                    archetype_actions[arch][action] = archetype_actions[arch].get(action, 0) + count

                archetype_games[arch] += 1
                if i == result.winner:
                    archetype_wins[arch] += 1

        # Calculate action distributions (percentages)
        action_distributions = {}
        for arch, actions in archetype_actions.items():
            total = sum(actions.values())
            if total > 0:
                action_distributions[arch] = {
                    action: count / total for action, count in actions.items()
                }
            else:
                action_distributions[arch] = {}

        # Calculate win rates
        win_rates = {
            arch: archetype_wins.get(arch, 0) / archetype_games.get(arch, 1)
            for arch in archetype_games
        }

        snapshot = {
            'games_played': games_played,
            'action_distributions': action_distributions,
            'win_rates': win_rates,
            'timestamp': time.time(),
        }

        self.snapshots.append(snapshot)

        # Update per-archetype history
        for arch in action_distributions:
            self.action_history[arch].append({
                'games': games_played,
                'actions': action_distributions[arch],
                'win_rate': win_rates.get(arch, 0),
            })

    def get_strategy_evolution(self, archetype: str) -> Dict:
        """Get how a specific archetype's strategy evolved."""
        history = self.action_history.get(archetype, [])

        if len(history) < 2:
            return {'error': 'Not enough data'}

        early = history[:len(history)//3]  # First third
        late = history[-len(history)//3:]  # Last third

        # Average action distributions
        def avg_actions(records):
            combined = {}
            for r in records:
                for action, pct in r['actions'].items():
                    combined[action] = combined.get(action, [])
                    combined[action].append(pct)
            return {k: np.mean(v) for k, v in combined.items()}

        early_avg = avg_actions(early)
        late_avg = avg_actions(late)

        # Calculate changes
        changes = {}
        all_actions = set(early_avg.keys()) | set(late_avg.keys())
        for action in all_actions:
            early_val = early_avg.get(action, 0)
            late_val = late_avg.get(action, 0)
            changes[action] = {
                'early': early_val,
                'late': late_val,
                'change': late_val - early_val,
            }

        # Win rate evolution
        early_wr = np.mean([r['win_rate'] for r in early])
        late_wr = np.mean([r['win_rate'] for r in late])

        return {
            'archetype': archetype,
            'action_changes': changes,
            'win_rate_early': early_wr,
            'win_rate_late': late_wr,
            'win_rate_change': late_wr - early_wr,
        }

    def get_all_evolutions(self) -> Dict[str, Dict]:
        """Get strategy evolution for all archetypes."""
        evolutions = {}
        for arch in get_archetype_types():
            evolutions[arch.value] = self.get_strategy_evolution(arch.value)
        return evolutions

    def to_dict(self) -> Dict:
        """Convert to dictionary for saving."""
        return {
            'snapshots': self.snapshots,
            'action_history': self.action_history,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'StrategyTracker':
        """Load from dictionary."""
        tracker = cls()
        tracker.snapshots = data.get('snapshots', [])
        tracker.action_history = data.get('action_history', {})
        return tracker


def print_strategy_evolution(tracker: StrategyTracker):
    """Print strategy evolution analysis with rich formatting."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box

        console = Console()
        evolutions = tracker.get_all_evolutions()

        console.print("\n[bold cyan]STRATEGY EVOLUTION ANALYSIS[/bold cyan]")
        console.print("[dim]Comparing early training vs late training behavior[/dim]\n")

        for arch, evolution in evolutions.items():
            if 'error' in evolution:
                continue

            # Create table for this archetype
            table = Table(
                title=f"{arch.replace('_', ' ').title()}",
                box=box.ROUNDED
            )
            table.add_column("Action", style="cyan")
            table.add_column("Early", justify="right")
            table.add_column("Late", justify="right")
            table.add_column("Change", justify="right")

            for action, data in sorted(evolution['action_changes'].items()):
                change = data['change']
                if abs(change) < 0.01:
                    change_style = "dim"
                    change_str = f"{change:+.1%}"
                elif change > 0:
                    change_style = "green"
                    change_str = f"[bold]+{change:.1%}[/bold]"
                else:
                    change_style = "red"
                    change_str = f"[bold]{change:.1%}[/bold]"

                table.add_row(
                    action.replace('_', ' ').title(),
                    f"{data['early']:.1%}",
                    f"{data['late']:.1%}",
                    f"[{change_style}]{change_str}[/{change_style}]"
                )

            # Win rate row
            wr_change = evolution['win_rate_change']
            wr_style = "green" if wr_change > 0 else "red" if wr_change < 0 else "dim"
            table.add_row(
                "[bold]Win Rate[/bold]",
                f"{evolution['win_rate_early']:.1%}",
                f"{evolution['win_rate_late']:.1%}",
                f"[{wr_style}]{wr_change:+.1%}[/{wr_style}]",
                style="bold"
            )

            console.print(table)
            console.print()

    except ImportError:
        # Fallback without rich
        print("\n=== STRATEGY EVOLUTION ANALYSIS ===")
        evolutions = tracker.get_all_evolutions()

        for arch, evolution in evolutions.items():
            if 'error' in evolution:
                continue

            print(f"\n{arch.upper()}")
            print("-" * 40)

            for action, data in evolution['action_changes'].items():
                print(f"  {action}: {data['early']:.1%} -> {data['late']:.1%} ({data['change']:+.1%})")

            print(f"  Win Rate: {evolution['win_rate_early']:.1%} -> {evolution['win_rate_late']:.1%}")
