"""
Main training loop for Ticket to Ride RL.
"""
import os
import time
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Rich CLI imports
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from ticket_to_ride_rl.game import GameState, GameConfig
from ticket_to_ride_rl.agents import (
    PPOAgent, PPOConfig, ArchetypeType, get_archetype_types
)
from .self_play import SelfPlayManager, SelfPlayGame, PlayerSetup
from .metrics import MetricsCollector, compute_metrics_from_game


# Initialize console
console = Console() if RICH_AVAILABLE else None


def print_rich(message: str, style: str = None):
    """Print with rich if available, else regular print."""
    if console:
        console.print(message, style=style)
    else:
        print(message)


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Game settings
    num_players: int = 4
    num_games: int = 50000

    # Training settings
    games_per_update: int = 16
    evaluation_interval: int = 1000
    checkpoint_interval: int = 5000
    log_interval: int = 100

    # Parallelization
    num_workers: int = 4

    # Output
    output_dir: str = "results"
    experiment_name: str = "ttr_rl"

    # PPO config
    ppo_config: PPOConfig = None

    def __post_init__(self):
        if self.ppo_config is None:
            self.ppo_config = PPOConfig()


class Trainer:
    """
    Main trainer class that orchestrates training.
    """

    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.metrics = MetricsCollector()

        # Create output directories
        self.output_dir = os.path.join(
            self.config.output_dir,
            self.config.experiment_name
        )
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'checkpoints'), exist_ok=True)

        # Get observation dimension from a test game
        test_game = GameState(GameConfig(num_players=self.config.num_players))
        test_game.setup_game()
        self.obs_dim = len(test_game.get_flat_observation(0))

        # Create agents for each archetype (blind and tracking versions)
        self.agents: Dict[ArchetypeType, Tuple[PPOAgent, PPOAgent]] = {}
        self._create_agents()

        # Training state
        self.total_games = 0
        self.total_updates = 0
        self.training_stats: List[Dict] = []

    def _create_agents(self):
        """Create all agents for training."""
        for archetype in get_archetype_types():
            # Blind agent
            blind_agent = PPOAgent(
                obs_dim=self.obs_dim,
                config=self.config.ppo_config,
                archetype=archetype,
                use_tracking=False,
                device='cpu',
                num_players=self.config.num_players
            )

            # Tracking agent
            tracking_agent = PPOAgent(
                obs_dim=self.obs_dim,
                config=self.config.ppo_config,
                archetype=archetype,
                use_tracking=True,
                device='cpu',
                num_players=self.config.num_players
            )

            self.agents[archetype] = (blind_agent, tracking_agent)

    def _create_config_table(self) -> 'Table':
        """Create a rich table showing configuration."""
        if not RICH_AVAILABLE:
            return None

        table = Table(title="Training Configuration", box=box.ROUNDED)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Games", f"{self.config.num_games:,}")
        table.add_row("Players per Game", str(self.config.num_players))
        table.add_row("Observation Dim", str(self.obs_dim))
        table.add_row("Output Directory", self.output_dir)
        table.add_row("Learning Rate", f"{self.config.ppo_config.learning_rate:.0e}")
        table.add_row("Batch Size", str(self.config.ppo_config.batch_size))

        return table

    def _create_win_rate_table(self, win_rates: Dict[str, float]) -> 'Table':
        """Create a rich table showing win rates."""
        if not RICH_AVAILABLE:
            return None

        table = Table(title="Win Rates by Archetype", box=box.ROUNDED)
        table.add_column("Archetype", style="cyan")
        table.add_column("Win Rate", justify="right")
        table.add_column("Bar", justify="left")

        # Sort by win rate
        sorted_rates = sorted(win_rates.items(), key=lambda x: -x[1])

        for arch, rate in sorted_rates:
            # Create a simple bar
            bar_width = int(rate * 40)
            bar = "[green]" + "█" * bar_width + "[/green]" + "░" * (40 - bar_width)
            rate_str = f"{rate:.1%}"

            # Color based on performance
            if rate >= 0.30:
                style = "bold green"
            elif rate >= 0.20:
                style = "yellow"
            else:
                style = "red"

            table.add_row(arch.replace('_', ' ').title(), f"[{style}]{rate_str}[/{style}]", bar)

        return table

    def _create_tracking_table(self) -> 'Table':
        """Create table comparing tracking vs blind performance."""
        if not RICH_AVAILABLE:
            return None

        comparison = self.metrics.get_tracking_comparison()
        if not comparison:
            return None

        table = Table(title="Tracking vs Blind Performance", box=box.ROUNDED)
        table.add_column("Archetype", style="cyan")
        table.add_column("Blind", justify="right")
        table.add_column("Tracking", justify="right")
        table.add_column("Advantage", justify="right")

        for arch, modes in sorted(comparison.items()):
            blind_rate = modes.get('blind', {}).get('win_rate', 0)
            tracking_rate = modes.get('tracking', {}).get('win_rate', 0)
            advantage = tracking_rate - blind_rate

            adv_style = "green" if advantage > 0 else "red" if advantage < 0 else "white"

            table.add_row(
                arch.replace('_', ' ').title(),
                f"{blind_rate:.1%}",
                f"{tracking_rate:.1%}",
                f"[{adv_style}]{advantage:+.1%}[/{adv_style}]"
            )

        return table

    def train(self, progress_callback=None):
        """
        Main training loop with rich CLI output.
        """
        # Print header
        if RICH_AVAILABLE:
            console.print(Panel.fit(
                "[bold blue]TICKET TO RIDE[/bold blue]\n"
                "[dim]Reinforcement Learning Training System[/dim]",
                border_style="blue"
            ))
            console.print()
            console.print(self._create_config_table())
            console.print()

            # Show archetypes
            archetypes = ", ".join([a.value.replace('_', ' ').title() for a in get_archetype_types()])
            console.print(f"[cyan]Archetypes:[/cyan] {archetypes}")
            console.print()
        else:
            print("=" * 60)
            print("TICKET TO RIDE - REINFORCEMENT LEARNING TRAINING")
            print("=" * 60)
            print(f"Games: {self.config.num_games:,}, Players: {self.config.num_players}")

        manager = SelfPlayManager(
            agents_by_archetype=self.agents,
            num_players=self.config.num_players,
            metrics=self.metrics
        )

        start_time = time.time()

        # Training loop with rich progress
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=40),
                TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
                TextColumn("•"),
                TextColumn("[cyan]{task.completed:,}[/cyan]/[cyan]{task.total:,}[/cyan] games"),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                console=console,
                refresh_per_second=2
            ) as progress:
                task = progress.add_task("[green]Training", total=self.config.num_games)

                while self.total_games < self.config.num_games:
                    batch_size = min(
                        self.config.games_per_update,
                        self.config.num_games - self.total_games
                    )

                    record_samples = 1 if self.total_games % 5000 < batch_size else 0
                    game_metrics = manager.play_games(batch_size, record_samples=record_samples)
                    self.total_games += batch_size
                    progress.update(task, completed=self.total_games)

                    # Update agents
                    update_stats = manager.update_agents()
                    self.total_updates += 1

                    # Logging
                    if self.total_games % self.config.log_interval < batch_size:
                        self._log_progress_rich(game_metrics, update_stats)

                    # Evaluation
                    if self.total_games % self.config.evaluation_interval < batch_size:
                        self._run_evaluation()

                    # Checkpointing
                    if self.total_games % self.config.checkpoint_interval < batch_size:
                        self._save_checkpoint()
                        console.print(f"[dim]Checkpoint saved at {self.total_games:,} games[/dim]")

        else:
            # Fallback to tqdm
            from tqdm import tqdm
            pbar = tqdm(total=self.config.num_games, desc="Training")

            while self.total_games < self.config.num_games:
                batch_size = min(
                    self.config.games_per_update,
                    self.config.num_games - self.total_games
                )

                record_samples = 1 if self.total_games % 5000 < batch_size else 0
                game_metrics = manager.play_games(batch_size, record_samples=record_samples)
                self.total_games += batch_size
                pbar.update(batch_size)

                update_stats = manager.update_agents()
                self.total_updates += 1

                if self.total_games % self.config.log_interval < batch_size:
                    self._log_progress(game_metrics, update_stats)

                if self.total_games % self.config.evaluation_interval < batch_size:
                    self._run_evaluation()

                if self.total_games % self.config.checkpoint_interval < batch_size:
                    self._save_checkpoint()

            pbar.close()

        # Final save
        self._save_checkpoint()
        self._save_final_results()

        elapsed = time.time() - start_time

        # Print summary
        self._print_summary_rich(elapsed)

    def _log_progress_rich(self, recent_metrics, update_stats):
        """Log training progress with rich formatting."""
        win_rates = self.metrics.get_win_rates()

        stats = {
            'games': self.total_games,
            'updates': self.total_updates,
            'win_rates': win_rates,
            'update_stats': update_stats,
        }
        self.training_stats.append(stats)

        if RICH_AVAILABLE:
            console.print()
            console.print(self._create_win_rate_table(win_rates))

    def _log_progress(self, recent_metrics, update_stats):
        """Log training progress (fallback)."""
        win_rates = self.metrics.get_win_rates()

        stats = {
            'games': self.total_games,
            'updates': self.total_updates,
            'win_rates': win_rates,
            'update_stats': update_stats,
        }
        self.training_stats.append(stats)

        win_rate_str = ", ".join(
            f"{arch}: {rate:.1%}"
            for arch, rate in sorted(win_rates.items())
        )
        print(f"Games: {self.total_games:,} | Win rates: {win_rate_str}")

    def _run_evaluation(self):
        """Run evaluation games."""
        eval_wins = {arch.value: 0 for arch in get_archetype_types()}
        eval_games = 20

        for _ in range(eval_games):
            archetypes = list(get_archetype_types())
            np.random.shuffle(archetypes)
            selected = archetypes[:self.config.num_players]

            players = []
            for arch in selected:
                agent = self.agents[arch][0]
                players.append(PlayerSetup(
                    agent=agent,
                    archetype=arch,
                    use_tracking=False
                ))

            game = SelfPlayGame(players, GameConfig(num_players=len(players)))
            winner, scores, _ = game.play_game()
            eval_wins[players[winner].archetype.value] += 1

    def _save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')

        for archetype, (blind_agent, tracking_agent) in self.agents.items():
            blind_path = os.path.join(
                checkpoint_dir,
                f"{archetype.value}_blind_{self.total_games}.pt"
            )
            tracking_path = os.path.join(
                checkpoint_dir,
                f"{archetype.value}_tracking_{self.total_games}.pt"
            )
            blind_agent.save(blind_path)
            tracking_agent.save(tracking_path)

        metrics_path = os.path.join(self.output_dir, 'metrics.json')
        self.metrics.save(metrics_path)

        stats_path = os.path.join(self.output_dir, 'training_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)

    def _save_final_results(self):
        """Save final training results."""
        results = {
            'config': {
                'num_games': self.config.num_games,
                'num_players': self.config.num_players,
                'obs_dim': self.obs_dim,
            },
            'final_win_rates': self.metrics.get_win_rates(),
            'tracking_comparison': self.metrics.get_tracking_comparison(),
            'summary': self.metrics.get_summary(),
        }

        results_path = os.path.join(self.output_dir, 'final_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

    def _print_summary_rich(self, elapsed: float):
        """Print training summary with rich formatting."""
        if RICH_AVAILABLE:
            console.print()
            console.print(Panel.fit(
                "[bold green]TRAINING COMPLETE[/bold green]",
                border_style="green"
            ))

            # Stats table
            stats_table = Table(box=box.ROUNDED)
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="green")
            stats_table.add_row("Total Games", f"{self.total_games:,}")
            stats_table.add_row("Total Updates", f"{self.total_updates:,}")
            stats_table.add_row("Training Time", f"{elapsed/60:.1f} minutes")
            stats_table.add_row("Games/Second", f"{self.total_games/elapsed:.1f}")
            console.print(stats_table)
            console.print()

            # Final win rates
            console.print(self._create_win_rate_table(self.metrics.get_win_rates()))
            console.print()

            # Tracking comparison
            tracking_table = self._create_tracking_table()
            if tracking_table:
                console.print(tracking_table)
                console.print()

            console.print(f"[dim]Results saved to: {self.output_dir}[/dim]")
        else:
            self._print_summary()

    def _print_summary(self):
        """Print training summary (fallback)."""
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)

        print(f"\nTotal games: {self.total_games:,}")
        print(f"Total updates: {self.total_updates:,}")

        print("\nFinal Win Rates:")
        for arch, rate in sorted(self.metrics.get_win_rates().items()):
            print(f"  {arch}: {rate:.2%}")

        print("\nTracking vs Blind Comparison:")
        comparison = self.metrics.get_tracking_comparison()
        for arch, modes in sorted(comparison.items()):
            blind_rate = modes.get('blind', {}).get('win_rate', 0)
            tracking_rate = modes.get('tracking', {}).get('win_rate', 0)
            diff = tracking_rate - blind_rate
            print(f"  {arch}: Blind={blind_rate:.2%}, Tracking={tracking_rate:.2%}, Diff={diff:+.2%}")

        print(f"\nResults saved to: {self.output_dir}")

    def load_checkpoint(self, games: int):
        """Load agents from a specific checkpoint."""
        checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')

        for archetype, (blind_agent, tracking_agent) in self.agents.items():
            blind_path = os.path.join(
                checkpoint_dir,
                f"{archetype.value}_blind_{games}.pt"
            )
            tracking_path = os.path.join(
                checkpoint_dir,
                f"{archetype.value}_tracking_{games}.pt"
            )

            if os.path.exists(blind_path):
                blind_agent.load(blind_path)
            if os.path.exists(tracking_path):
                tracking_agent.load(tracking_path)

        self.total_games = games


def run_training(num_games: int = 50000, output_dir: str = "results",
                 num_players: int = 4, num_workers: int = 4):
    """
    Convenience function to run training.
    """
    config = TrainingConfig(
        num_games=num_games,
        output_dir=output_dir,
        num_players=num_players,
        num_workers=num_workers,
    )

    trainer = Trainer(config)
    trainer.train()

    return trainer


if __name__ == "__main__":
    run_training(num_games=1000)
