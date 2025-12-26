#!/usr/bin/env python3
"""
Ticket to Ride Reinforcement Learning - Main Entry Point

This script runs the complete RL training pipeline to discover optimal
strategies for playing Ticket to Ride.

Usage:
    python main.py --games 50000 --output results/
    python main.py --games 50000 --parallel              # Use all CPU cores
    python main.py --games 50000 --parallel --workers 16 # Use 16 cores
    python main.py --analyze results/                    # Analyze existing results
    python main.py --quick-test                          # Run quick validation test
"""
import argparse
import os
import sys
import time
import multiprocessing as mp
from datetime import timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ticket_to_ride_rl.game import GameState, GameConfig, play_random_game
from ticket_to_ride_rl.agents import PPOAgent, PPOConfig, ArchetypeType, get_archetype_types
from ticket_to_ride_rl.training import (
    Trainer, TrainingConfig,
    ParallelTrainer, ParallelTrainingConfig,
    get_optimal_workers, print_strategy_evolution
)
from ticket_to_ride_rl.analysis import analyze_results, create_visualizations


def run_quick_test():
    """Run a quick test to validate the system works."""
    print("=" * 60)
    print("QUICK VALIDATION TEST")
    print("=" * 60)

    # Test 1: Game engine
    print("\n1. Testing game engine...")
    try:
        winner, scores = play_random_game(num_players=4, verbose=False)
        print(f"   Random game completed. Winner: Player {winner}, Scores: {scores}")
        print("   [PASS] Game engine works!")
    except Exception as e:
        print(f"   [FAIL] Game engine error: {e}")
        return False

    # Test 2: Agent creation
    print("\n2. Testing agent creation...")
    try:
        test_game = GameState(GameConfig(num_players=4))
        test_game.setup_game()
        obs_dim = len(test_game.get_flat_observation(0))

        agent = PPOAgent(
            obs_dim=obs_dim,
            archetype=ArchetypeType.ARCHITECT,
            use_tracking=True,
            num_players=4
        )
        print(f"   Created agent with obs_dim={obs_dim}")
        print("   [PASS] Agent creation works!")
    except Exception as e:
        print(f"   [FAIL] Agent creation error: {e}")
        return False

    # Test 3: Self-play
    print("\n3. Testing self-play simulation...")
    try:
        from ticket_to_ride_rl.training import SelfPlayGame, PlayerSetup

        players = []
        archetypes = list(ArchetypeType)[:4]
        for i, arch in enumerate(archetypes):
            agent = PPOAgent(
                obs_dim=obs_dim,
                archetype=arch,
                use_tracking=(i % 2 == 0),
                num_players=4
            )
            players.append(PlayerSetup(
                agent=agent,
                archetype=arch,
                use_tracking=(i % 2 == 0)
            ))

        game = SelfPlayGame(players)
        winner, scores, info = game.play_game()
        print(f"   Self-play game completed. Winner: Player {winner}")
        print("   [PASS] Self-play works!")
    except Exception as e:
        print(f"   [FAIL] Self-play error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 4: Parallel simulation
    print("\n4. Testing parallel game simulation...")
    try:
        from ticket_to_ride_rl.training import ParallelGameRunner
        runner = ParallelGameRunner(num_players=4, num_workers=2, obs_dim=obs_dim)
        results = runner.run_games(4)
        print(f"   Ran 4 parallel games with 2 workers")
        print("   [PASS] Parallel simulation works!")
    except Exception as e:
        print(f"   [FAIL] Parallel simulation error: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    print(f"\nCPU cores available: {mp.cpu_count()}")
    print(f"Recommended workers: {get_optimal_workers()}")
    print("\nThe system is ready for training. Run with:")
    print("  python main.py --games 50000 --parallel --output results/")
    return True


def run_training(args):
    """Run the full training pipeline (sequential mode)."""
    # Create training config
    ppo_config = PPOConfig(
        learning_rate=3e-4,
        batch_size=64,
        num_epochs=4,
        hidden_dim=256,
        use_lstm=False,
    )

    config = TrainingConfig(
        num_games=args.games,
        num_players=args.players,
        games_per_update=16,
        evaluation_interval=1000,
        checkpoint_interval=max(1000, args.games // 10),
        log_interval=100,
        output_dir=args.output,
        experiment_name="ttr_rl_experiment",
        ppo_config=ppo_config,
    )

    # Start training
    start_time = time.time()
    trainer = Trainer(config)
    trainer.train()
    elapsed = time.time() - start_time

    print(f"\nTraining completed in {timedelta(seconds=int(elapsed))}")

    # Run analysis
    results_dir = os.path.join(args.output, "ttr_rl_experiment")
    run_post_analysis(results_dir, args)

    return trainer


def run_parallel_training(args):
    """Run parallel training pipeline."""
    num_workers = args.workers if args.workers else get_optimal_workers()

    config = ParallelTrainingConfig(
        num_games=args.games,
        num_players=args.players,
        num_workers=num_workers,
        games_per_batch=min(num_workers * 4, 200),  # Scale batch with workers
        log_interval=max(100, num_workers * 10),
        checkpoint_interval=max(1000, args.games // 10),
        strategy_snapshot_interval=max(500, args.games // 50),
        output_dir=args.output,
        experiment_name="ttr_parallel",
    )

    # Start training
    start_time = time.time()
    trainer = ParallelTrainer(config)
    trainer.train()
    elapsed = time.time() - start_time

    print(f"\nTraining completed in {timedelta(seconds=int(elapsed))}")
    print(f"Average speed: {args.games / elapsed:.1f} games/second")

    # Run analysis
    results_dir = os.path.join(args.output, "ttr_parallel")
    run_post_analysis(results_dir, args)

    return trainer


def run_post_analysis(results_dir: str, args):
    """Run post-training analysis."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        console.print()
        console.print(Panel.fit(
            "[bold cyan]RUNNING ANALYSIS[/bold cyan]",
            border_style="cyan"
        ))
    except ImportError:
        print("\n" + "=" * 60)
        print("RUNNING ANALYSIS")
        print("=" * 60)

    if os.path.exists(results_dir):
        analyze_results(results_dir, print_report=True)

        if not args.no_plots:
            try:
                create_visualizations(results_dir)
            except ImportError:
                print("\nNote: Install matplotlib for visualizations: pip install matplotlib")


def run_analysis(args):
    """Run analysis on existing results."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        console.print(Panel.fit(
            "[bold cyan]ANALYZING RESULTS[/bold cyan]",
            border_style="cyan"
        ))
    except ImportError:
        print("=" * 60)
        print("ANALYZING RESULTS")
        print("=" * 60)

    if not os.path.exists(args.analyze):
        print(f"Error: Results directory not found: {args.analyze}")
        return

    analyze_results(args.analyze, print_report=True)

    # Check for strategy evolution data
    strategy_file = os.path.join(args.analyze, 'strategy_snapshots')
    if os.path.exists(strategy_file):
        try:
            import json
            from ticket_to_ride_rl.training import StrategyTracker, print_strategy_evolution

            # Find latest snapshot
            snapshots = sorted([f for f in os.listdir(strategy_file) if f.endswith('.json')])
            if snapshots:
                latest = os.path.join(strategy_file, snapshots[-1])
                with open(latest) as f:
                    data = json.load(f)
                tracker = StrategyTracker.from_dict(data)
                print_strategy_evolution(tracker)
        except Exception as e:
            print(f"Could not load strategy evolution data: {e}")

    if not args.no_plots:
        try:
            create_visualizations(args.analyze)
        except ImportError:
            print("\nNote: Install matplotlib for visualizations: pip install matplotlib")


def main():
    parser = argparse.ArgumentParser(
        description="Ticket to Ride RL Training System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --games 50000 --output results/           # Sequential training
  python main.py --games 50000 --parallel --output results/ # Parallel training (auto cores)
  python main.py --games 50000 --parallel --workers 16      # Use 16 CPU cores
  python main.py --analyze results/ttr_parallel             # Analyze results
  python main.py --quick-test                               # Validate installation

Research Questions:
  1. Destination Card Importance - How many destinations are optimal?
  2. Hand Tracking Value - Does tracking opponent hands help?
  3. Timing Strategies - When to build vs when to gather cards?
        """
    )

    parser.add_argument('--games', type=int, default=50000,
                       help='Number of games to train (default: 50000)')
    parser.add_argument('--players', type=int, default=4,
                       help='Number of players per game (default: 4)')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory (default: results)')
    parser.add_argument('--analyze', type=str, default=None,
                       help='Path to results directory to analyze')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick validation test')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating visualization plots')

    # Parallel training options
    parser.add_argument('--parallel', action='store_true',
                       help='Use parallel game simulation (faster)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of CPU workers for parallel mode (default: auto-detect)')

    args = parser.parse_args()

    # Route to appropriate function
    if args.quick_test:
        success = run_quick_test()
        sys.exit(0 if success else 1)
    elif args.analyze:
        run_analysis(args)
    elif args.parallel:
        run_parallel_training(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
