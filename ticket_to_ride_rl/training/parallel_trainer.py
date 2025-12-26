"""
Parallel training with strategy evolution tracking.
Scales automatically to available CPU cores.
"""
import os
import time
import json
import numpy as np
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

# Rich CLI imports
try:
    from rich.console import Console, Group
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn, TaskProgressColumn
    from rich.panel import Panel
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from ticket_to_ride_rl.game import GameState, GameConfig
from ticket_to_ride_rl.agents import (
    PPOAgent, PPOConfig, ArchetypeType, get_archetype_types
)
from .parallel import (
    ParallelGameRunner, StrategyTracker, GameResult,
    get_optimal_workers, print_strategy_evolution
)
from .metrics import MetricsCollector, GameMetrics


console = Console() if RICH_AVAILABLE else None


@dataclass
class ParallelTrainingConfig:
    """Configuration for parallel training."""
    # Game settings
    num_players: int = 4
    num_games: int = 50000

    # Parallel settings
    num_workers: int = None  # Auto-detect if None
    games_per_batch: int = 100  # Games per parallel batch

    # Training settings
    log_interval: int = 500
    checkpoint_interval: int = 5000
    strategy_snapshot_interval: int = 1000

    # Output
    output_dir: str = "results"
    experiment_name: str = "ttr_parallel"

    # PPO config (for reference, not used in parallel simulation)
    ppo_config: PPOConfig = None

    def __post_init__(self):
        if self.ppo_config is None:
            self.ppo_config = PPOConfig()
        if self.num_workers is None:
            self.num_workers = get_optimal_workers()


class ParallelTrainer:
    """
    Trainer that runs games in parallel across multiple CPU cores.
    Tracks strategy evolution over training.
    """

    def __init__(self, config: ParallelTrainingConfig = None):
        self.config = config or ParallelTrainingConfig()

        # Create output directories
        self.output_dir = os.path.join(
            self.config.output_dir,
            self.config.experiment_name
        )
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'strategy_snapshots'), exist_ok=True)

        # Get observation dimension
        test_game = GameState(GameConfig(num_players=self.config.num_players))
        test_game.setup_game()
        self.obs_dim = len(test_game.get_flat_observation(0))

        # Initialize parallel runner
        self.runner = ParallelGameRunner(
            num_players=self.config.num_players,
            num_workers=self.config.num_workers,
            obs_dim=self.obs_dim,
            ppo_config=self.config.ppo_config
        )

        # Strategy tracker
        self.strategy_tracker = StrategyTracker()

        # Metrics
        self.metrics = MetricsCollector()

        # Statistics
        self.total_games = 0
        self.archetype_wins: Dict[str, int] = defaultdict(int)
        self.archetype_games: Dict[str, int] = defaultdict(int)
        self.tracking_wins: Dict[str, int] = defaultdict(int)
        self.tracking_games: Dict[str, int] = defaultdict(int)

        # Training history
        self.win_rate_history: List[Dict] = []

    def _process_results(self, results: List[GameResult]):
        """Process game results and update statistics."""
        for result in results:
            # Update archetype stats
            winner_arch = result.archetypes[result.winner]
            self.archetype_wins[winner_arch] += 1

            for i, arch in enumerate(result.archetypes):
                self.archetype_games[arch] += 1

                # Tracking stats
                tracking_key = f"{arch}_{'tracking' if result.tracking[i] else 'blind'}"
                self.tracking_games[tracking_key] += 1
                if i == result.winner:
                    self.tracking_wins[tracking_key] += 1

    def _get_win_rates(self) -> Dict[str, float]:
        """Calculate current win rates."""
        return {
            arch: self.archetype_wins[arch] / self.archetype_games[arch]
            if self.archetype_games[arch] > 0 else 0
            for arch in self.archetype_games
        }

    def _get_tracking_comparison(self) -> Dict[str, Dict]:
        """Get tracking vs blind comparison."""
        results = {}
        for key in self.tracking_games:
            parts = key.rsplit('_', 1)
            if len(parts) == 2:
                archetype, mode = parts
                if archetype not in results:
                    results[archetype] = {}
                results[archetype][mode] = {
                    'games': self.tracking_games[key],
                    'wins': self.tracking_wins[key],
                    'win_rate': self.tracking_wins[key] / self.tracking_games[key]
                    if self.tracking_games[key] > 0 else 0
                }
        return results

    def _create_config_table(self) -> 'Table':
        """Create configuration display table."""
        if not RICH_AVAILABLE:
            return None

        table = Table(title="Parallel Training Configuration", box=box.ROUNDED)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Games", f"{self.config.num_games:,}")
        table.add_row("Players per Game", str(self.config.num_players))
        table.add_row("CPU Workers", f"{self.config.num_workers} / {mp.cpu_count()} cores")
        table.add_row("Games per Batch", str(self.config.games_per_batch))
        table.add_row("Output Directory", self.output_dir)

        # Estimate speedup
        estimated_speedup = min(self.config.num_workers, self.config.games_per_batch)
        table.add_row("Est. Speedup", f"~{estimated_speedup}x")

        return table

    def _create_win_rate_table(self, win_rates: Dict[str, float]) -> 'Table':
        """Create win rates display table."""
        if not RICH_AVAILABLE:
            return None

        table = Table(title=f"Final Results ({self.total_games:,} games)", box=box.ROUNDED)
        table.add_column("Archetype", style="cyan")
        table.add_column("Win Rate", justify="right")
        table.add_column("Games", justify="right")
        table.add_column("", justify="left")

        sorted_rates = sorted(win_rates.items(), key=lambda x: -x[1])

        for arch, rate in sorted_rates:
            games = self.archetype_games.get(arch, 0)
            bar_width = int(rate * 40)
            bar = "[green]" + "█" * bar_width + "[/green]" + "░" * (40 - bar_width)

            if rate >= 0.28:
                style = "bold green"
            elif rate >= 0.22:
                style = "yellow"
            else:
                style = "red"

            table.add_row(
                arch.replace('_', ' ').title(),
                f"[{style}]{rate:.0%}[/{style}]",
                f"{games:,}",
                bar
            )

        return table

    def _create_tracking_table(self) -> 'Table':
        """Create tracking comparison table."""
        if not RICH_AVAILABLE:
            return None

        comparison = self._get_tracking_comparison()
        if not comparison:
            return None

        # Archetype icons
        icons = {
            'six_shooter': '🎯',
            'instant_gratification': '⚡',
            'hoarder': '💰',
            'blocker': '🛡️',
            'wildcard': '🃏',
        }

        table = Table(box=box.ROUNDED, border_style="bright_blue")
        table.add_column("", width=3)
        table.add_column("Archetype", style="bold")
        table.add_column("👁️ Blind", justify="right")
        table.add_column("🔍 Track", justify="right")
        table.add_column("Δ Advantage", justify="right")

        for arch, modes in sorted(comparison.items()):
            blind_rate = modes.get('blind', {}).get('win_rate', 0)
            tracking_rate = modes.get('tracking', {}).get('win_rate', 0)
            advantage = tracking_rate - blind_rate
            icon = icons.get(arch, '•')

            if advantage > 0.02:
                adv_str = f"[bold green]▲ +{round(advantage*100)}%[/bold green]"
            elif advantage < -0.02:
                adv_str = f"[bold red]▼ {round(advantage*100)}%[/bold red]"
            else:
                adv_str = f"[dim]━ 0%[/dim]"

            table.add_row(
                icon,
                arch.replace('_', ' ').title(),
                f"{round(blind_rate*100)}%",
                f"{round(tracking_rate*100)}%",
                adv_str
            )

        return table

    def _create_live_dashboard(self, elapsed: float) -> 'Table':
        """Create a beautiful live dashboard display."""
        if not RICH_AVAILABLE:
            return None

        win_rates = self._get_win_rates()
        sorted_rates = sorted(win_rates.items(), key=lambda x: -x[1])

        # Archetype display config with colors and icons
        arch_config = {
            'six_shooter': {'name': 'Six Shooter', 'icon': '🎯', 'color': 'bright_red'},
            'instant_gratification': {'name': 'Instant', 'icon': '⚡', 'color': 'bright_yellow'},
            'hoarder': {'name': 'Hoarder', 'icon': '💰', 'color': 'bright_green'},
            'blocker': {'name': 'Blocker', 'icon': '🛡️', 'color': 'bright_blue'},
            'wildcard': {'name': 'Wildcard', 'icon': '🃏', 'color': 'bright_magenta'},
        }

        # Create the dashboard table
        table = Table(box=box.DOUBLE_EDGE, border_style="bright_blue", padding=(0, 1))
        table.add_column("", justify="center", width=3)
        table.add_column("Archetype", style="bold", width=12)
        table.add_column("Win Rate", justify="right", width=8)
        table.add_column("Progress", width=30)

        for rank, (arch, rate) in enumerate(sorted_rates, 1):
            cfg = arch_config.get(arch, {'name': arch, 'icon': '•', 'color': 'white'})
            pct = round(rate * 100)

            # Medal for top 3
            if rank == 1:
                medal = "🥇"
            elif rank == 2:
                medal = "🥈"
            elif rank == 3:
                medal = "🥉"
            else:
                medal = f"[dim]#{rank}[/dim]"

            # Progress bar with gradient
            bar_width = int(rate * 25)
            bar = f"[{cfg['color']}]{'█' * bar_width}[/{cfg['color']}][dim]{'░' * (25 - bar_width)}[/dim]"

            table.add_row(
                medal,
                f"{cfg['icon']} [{cfg['color']}]{cfg['name']}[/{cfg['color']}]",
                f"[bold white]{pct}%[/bold white]",
                bar
            )

        return table

    def _create_stats_line(self, elapsed: float) -> Text:
        """Create stats line."""
        games_per_sec = self.total_games / elapsed if elapsed > 0 else 0
        pct_complete = (self.total_games / self.config.num_games) * 100

        text = Text()
        text.append("  📊 ", style="dim")
        text.append(f"{self.total_games:,}", style="bold cyan")
        text.append(f" / {self.config.num_games:,} games  ", style="dim")
        text.append("⚡ ", style="dim")
        text.append(f"{games_per_sec:.1f}", style="bold green")
        text.append(" games/sec  ", style="dim")
        text.append("⏱️ ", style="dim")
        text.append(f"{elapsed:.0f}s", style="bold yellow")

        return text

    def _make_live_display(self, progress: 'Progress', task_id, elapsed: float) -> 'Group':
        """Create the live display group with dashboard and progress bar."""
        dashboard = self._create_live_dashboard(elapsed)
        stats = self._create_stats_line(elapsed)
        return Group(dashboard, stats, progress)

    def _print_header(self):
        """Print beautiful header."""
        if not RICH_AVAILABLE:
            print("=" * 60)
            print("TICKET TO RIDE - PARALLEL TRAINING")
            print("=" * 60)
            return

        # ASCII art title
        title = """
[bold bright_blue]╔════════════════════════════════════════════════════════════════╗
║[/bold bright_blue] [bold white]🚂 TICKET TO RIDE[/bold white] [bold bright_cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold bright_cyan] [bold bright_blue]║
║[/bold bright_blue] [dim]Reinforcement Learning Arena[/dim]                                  [bold bright_blue]║
╚════════════════════════════════════════════════════════════════╝[/bold bright_blue]
"""
        console.print(title)

        # Config in a nice grid
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold cyan")
        grid.add_column(style="bold white")
        grid.add_column(style="bold cyan")
        grid.add_column(style="bold white")

        grid.add_row(
            "🎮 Games:", f"{self.config.num_games:,}",
            "👥 Players:", str(self.config.num_players)
        )
        grid.add_row(
            "⚙️  Workers:", f"{self.config.num_workers}/{mp.cpu_count()} cores",
            "📦 Batch:", str(self.config.games_per_batch)
        )

        console.print(Panel(grid, border_style="dim", padding=(0, 1)))
        console.print()

    def train(self):
        """Run parallel training."""
        start_time = time.time()

        # Print header
        if RICH_AVAILABLE:
            self._print_header()
        else:
            print("=" * 60)
            print("TICKET TO RIDE - PARALLEL TRAINING")
            print("=" * 60)
            print(f"Games: {self.config.num_games:,}, Workers: {self.config.num_workers}")

        # Training loop
        if RICH_AVAILABLE:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Training[/bold blue]"),
                BarColumn(bar_width=50, complete_style="green", finished_style="green"),
                TaskProgressColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
            )
            task = progress.add_task("Training", total=self.config.num_games)

            with Live(progress, console=console, refresh_per_second=4) as live:
                while self.total_games < self.config.num_games:
                    batch_size = min(
                        self.config.games_per_batch,
                        self.config.num_games - self.total_games
                    )

                    # Run parallel games
                    results = self.runner.run_games(batch_size)

                    # Process results
                    self._process_results(results)
                    self.total_games += batch_size
                    progress.update(task, completed=self.total_games)

                    # Update live display with status
                    elapsed = time.time() - start_time
                    live.update(self._make_live_display(progress, task, elapsed))

                    # Strategy snapshot
                    if self.total_games % self.config.strategy_snapshot_interval < batch_size:
                        self.strategy_tracker.record_snapshot(self.total_games, results)

                    # Record history (silently)
                    if self.total_games % self.config.log_interval < batch_size:
                        win_rates = self._get_win_rates()
                        self.win_rate_history.append({
                            'games': self.total_games,
                            'win_rates': win_rates.copy()
                        })

                    # Checkpointing (silently)
                    if self.total_games % self.config.checkpoint_interval < batch_size:
                        self._save_checkpoint()

        else:
            # Fallback without rich
            from tqdm import tqdm
            pbar = tqdm(total=self.config.num_games, desc="Training")

            while self.total_games < self.config.num_games:
                batch_size = min(
                    self.config.games_per_batch,
                    self.config.num_games - self.total_games
                )

                results = self.runner.run_games(batch_size)
                self._process_results(results)
                self.total_games += batch_size
                pbar.update(batch_size)

                if self.total_games % self.config.strategy_snapshot_interval < batch_size:
                    self.strategy_tracker.record_snapshot(self.total_games, results)

                if self.total_games % self.config.log_interval < batch_size:
                    win_rates = self._get_win_rates()
                    self.win_rate_history.append({
                        'games': self.total_games,
                        'win_rates': win_rates.copy()
                    })

                if self.total_games % self.config.checkpoint_interval < batch_size:
                    self._save_checkpoint()

            pbar.close()

        # Final save and summary
        elapsed = time.time() - start_time
        self._save_checkpoint()
        self._save_final_results()
        self._print_summary(elapsed)

        # Strategy evolution analysis
        if RICH_AVAILABLE:
            print_strategy_evolution(self.strategy_tracker)

    def _save_checkpoint(self):
        """Save training checkpoint."""
        # Save strategy tracker
        strategy_path = os.path.join(
            self.output_dir, 'strategy_snapshots',
            f'strategy_{self.total_games}.json'
        )
        with open(strategy_path, 'w') as f:
            json.dump(self.strategy_tracker.to_dict(), f, indent=2)

        # Save win rate history
        history_path = os.path.join(self.output_dir, 'win_rate_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.win_rate_history, f, indent=2)

        # Save current stats
        stats_path = os.path.join(self.output_dir, 'current_stats.json')
        with open(stats_path, 'w') as f:
            json.dump({
                'total_games': self.total_games,
                'win_rates': self._get_win_rates(),
                'tracking_comparison': self._get_tracking_comparison(),
            }, f, indent=2)

    def _save_final_results(self):
        """Save final training results."""
        results = {
            'config': {
                'num_games': self.config.num_games,
                'num_players': self.config.num_players,
                'num_workers': self.config.num_workers,
            },
            'total_games': self.total_games,
            'final_win_rates': self._get_win_rates(),
            'tracking_comparison': self._get_tracking_comparison(),
            'strategy_evolution': self.strategy_tracker.get_all_evolutions(),
        }

        results_path = os.path.join(self.output_dir, 'final_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

    def _print_summary(self, elapsed: float):
        """Print beautiful training summary."""
        games_per_sec = self.total_games / elapsed

        if RICH_AVAILABLE:
            # Victory banner
            console.print()
            banner = """
[bold bright_green]╔════════════════════════════════════════════════════════════════╗
║[/bold bright_green] [bold white]✨ TRAINING COMPLETE[/bold white] [bold bright_green]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold bright_green] [bold bright_green]║
╚════════════════════════════════════════════════════════════════╝[/bold bright_green]
"""
            console.print(banner)

            # Stats in a nice grid
            stats_grid = Table.grid(padding=(0, 3))
            stats_grid.add_column(justify="center")
            stats_grid.add_column(justify="center")
            stats_grid.add_column(justify="center")
            stats_grid.add_column(justify="center")

            stats_grid.add_row(
                f"[bold cyan]🎮 {self.total_games:,}[/bold cyan]\n[dim]games[/dim]",
                f"[bold green]⚡ {games_per_sec:.1f}[/bold green]\n[dim]games/sec[/dim]",
                f"[bold yellow]⏱️  {elapsed/60:.1f}m[/bold yellow]\n[dim]elapsed[/dim]",
                f"[bold magenta]🚀 {self.config.num_workers}x[/bold magenta]\n[dim]parallel[/dim]",
            )

            console.print(Panel(stats_grid, border_style="green"))
            console.print()

            # Final leaderboard
            console.print("[bold bright_white]🏆 FINAL LEADERBOARD[/bold bright_white]")
            console.print(self._create_live_dashboard(elapsed))
            console.print()

            # Tracking comparison
            console.print("[bold bright_white]🔍 TRACKING ANALYSIS[/bold bright_white]")
            console.print(self._create_tracking_table())
            console.print()

            console.print(f"[dim]📁 Results saved to: {self.output_dir}[/dim]")
        else:
            print("\n" + "=" * 60)
            print("TRAINING COMPLETE")
            print("=" * 60)
            print(f"Total games: {self.total_games:,}")
            print(f"Training time: {elapsed/60:.1f} minutes")
            print(f"Games/second: {games_per_sec:.1f}")
            print(f"\nFinal win rates: {self._get_win_rates()}")


def run_parallel_training(num_games: int = 50000, output_dir: str = "results",
                          num_players: int = 4, num_workers: int = None):
    """
    Convenience function to run parallel training.
    """
    config = ParallelTrainingConfig(
        num_games=num_games,
        output_dir=output_dir,
        num_players=num_players,
        num_workers=num_workers,
    )

    trainer = ParallelTrainer(config)
    trainer.train()

    return trainer
