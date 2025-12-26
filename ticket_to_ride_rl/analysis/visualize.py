"""
Visualization of training results and analysis.
"""
import json
import os
from typing import Dict, List, Optional
import numpy as np

# Check for matplotlib availability
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def check_matplotlib():
    """Check if matplotlib is available."""
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not installed. Visualization disabled.")
        print("Install with: pip install matplotlib")
        return False
    return True


class Visualizer:
    """
    Creates visualizations of training results.
    """

    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.output_dir = os.path.join(results_dir, 'plots')
        os.makedirs(self.output_dir, exist_ok=True)

        self.metrics = None
        self.training_stats = None
        self.final_results = None

        self._load_data()

    def _load_data(self):
        """Load data from results directory."""
        metrics_path = os.path.join(self.results_dir, 'metrics.json')
        stats_path = os.path.join(self.results_dir, 'training_stats.json')
        results_path = os.path.join(self.results_dir, 'final_results.json')

        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                self.metrics = json.load(f)

        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                self.training_stats = json.load(f)

        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                self.final_results = json.load(f)

    def plot_win_rates_over_time(self, save: bool = True):
        """Plot win rates for each archetype over training."""
        if not check_matplotlib() or not self.training_stats:
            return None

        # Extract win rates over time
        archetypes = set()
        for stat in self.training_stats:
            if 'win_rates' in stat:
                archetypes.update(stat['win_rates'].keys())

        games = []
        rates = {arch: [] for arch in archetypes}

        for stat in self.training_stats:
            games.append(stat.get('games', 0))
            for arch in archetypes:
                rate = stat.get('win_rates', {}).get(arch, None)
                rates[arch].append(rate)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))

        colors = plt.cm.tab10(np.linspace(0, 1, len(archetypes)))

        for (arch, rate_list), color in zip(sorted(rates.items()), colors):
            valid_games = [g for g, r in zip(games, rate_list) if r is not None]
            valid_rates = [r for r in rate_list if r is not None]
            if valid_rates:
                ax.plot(valid_games, valid_rates, label=arch, color=color, linewidth=2)

        ax.set_xlabel('Games Played', fontsize=12)
        ax.set_ylabel('Win Rate', fontsize=12)
        ax.set_title('Win Rates by Archetype Over Training', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 0.6)

        plt.tight_layout()

        if save:
            path = os.path.join(self.output_dir, 'win_rates_over_time.png')
            plt.savefig(path, dpi=150)
            print(f"Saved: {path}")

        return fig

    def plot_tracking_comparison(self, save: bool = True):
        """Plot tracking vs blind performance comparison."""
        if not check_matplotlib() or not self.final_results:
            return None

        comparison = self.final_results.get('tracking_comparison', {})
        if not comparison:
            return None

        archetypes = sorted(comparison.keys())
        blind_rates = []
        tracking_rates = []

        for arch in archetypes:
            modes = comparison[arch]
            blind_rates.append(modes.get('blind', {}).get('win_rate', 0))
            tracking_rates.append(modes.get('tracking', {}).get('win_rate', 0))

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(archetypes))
        width = 0.35

        bars1 = ax.bar(x - width/2, blind_rates, width, label='Blind', color='#3498db')
        bars2 = ax.bar(x + width/2, tracking_rates, width, label='Tracking', color='#e74c3c')

        ax.set_xlabel('Archetype', fontsize=12)
        ax.set_ylabel('Win Rate', fontsize=12)
        ax.set_title('Blind vs Tracking Performance by Archetype', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([a.replace('_', '\n') for a in archetypes], fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1%}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        if save:
            path = os.path.join(self.output_dir, 'tracking_comparison.png')
            plt.savefig(path, dpi=150)
            print(f"Saved: {path}")

        return fig

    def plot_destination_analysis(self, save: bool = True):
        """Plot destination card analysis."""
        if not check_matplotlib() or not self.metrics:
            return None

        games = self.metrics.get('games', [])
        if not games:
            return None

        # Calculate win rate by number of destinations
        dest_stats = {}
        for game in games:
            for i, num_dest in enumerate(game.get('destinations_drawn', [])):
                bucket = min(num_dest, 6)
                if bucket not in dest_stats:
                    dest_stats[bucket] = {'wins': 0, 'games': 0}
                dest_stats[bucket]['games'] += 1
                if i == game.get('winner', -1):
                    dest_stats[bucket]['wins'] += 1

        x = sorted(dest_stats.keys())
        rates = [dest_stats[k]['wins'] / dest_stats[k]['games']
                if dest_stats[k]['games'] > 0 else 0 for k in x]
        counts = [dest_stats[k]['games'] for k in x]

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Win rate by destinations
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(x)))
        bars = ax1.bar(x, rates, color=colors, edgecolor='black')
        ax1.set_xlabel('Number of Destinations', fontsize=12)
        ax1.set_ylabel('Win Rate', fontsize=12)
        ax1.set_title('Win Rate by Number of Destinations', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'{n}' if n < 6 else '6+' for n in x])
        ax1.grid(True, alpha=0.3, axis='y')

        for bar, rate in zip(bars, rates):
            ax1.annotate(f'{rate:.1%}',
                        xy=(bar.get_x() + bar.get_width() / 2, rate),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

        # Distribution of destinations
        ax2.bar(x, counts, color='#2ecc71', edgecolor='black')
        ax2.set_xlabel('Number of Destinations', fontsize=12)
        ax2.set_ylabel('Number of Games', fontsize=12)
        ax2.set_title('Distribution of Destinations Drawn', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'{n}' if n < 6 else '6+' for n in x])
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save:
            path = os.path.join(self.output_dir, 'destination_analysis.png')
            plt.savefig(path, dpi=150)
            print(f"Saved: {path}")

        return fig

    def plot_timing_analysis(self, save: bool = True):
        """Plot timing strategy analysis."""
        if not check_matplotlib() or not self.metrics:
            return None

        games = self.metrics.get('games', [])
        if not games:
            return None

        # Win rate by first route timing
        timing_stats = {}
        for game in games:
            for i, turn in enumerate(game.get('first_route_turns', [])):
                if turn is not None:
                    if turn <= 3:
                        bucket = 'Very Early\n(1-3)'
                    elif turn <= 7:
                        bucket = 'Early\n(4-7)'
                    elif turn <= 12:
                        bucket = 'Mid\n(8-12)'
                    else:
                        bucket = 'Late\n(13+)'

                    if bucket not in timing_stats:
                        timing_stats[bucket] = {'wins': 0, 'games': 0}
                    timing_stats[bucket]['games'] += 1
                    if i == game.get('winner', -1):
                        timing_stats[bucket]['wins'] += 1

        # Hand size over time for winners vs losers
        winner_hands = []
        loser_hands = []

        for game in games:
            winner = game.get('winner', 0)
            for i in range(game.get('num_players', 4)):
                avg_hand = game.get('avg_hand_sizes', [0]*4)[i]
                if i == winner:
                    winner_hands.append(avg_hand)
                else:
                    loser_hands.append(avg_hand)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Timing win rates
        order = ['Very Early\n(1-3)', 'Early\n(4-7)', 'Mid\n(8-12)', 'Late\n(13+)']
        x = [b for b in order if b in timing_stats]
        rates = [timing_stats[b]['wins'] / timing_stats[b]['games']
                if timing_stats[b]['games'] > 0 else 0 for b in x]

        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(x)))
        bars = ax1.bar(range(len(x)), rates, color=colors, edgecolor='black')
        ax1.set_xlabel('First Route Timing', fontsize=12)
        ax1.set_ylabel('Win Rate', fontsize=12)
        ax1.set_title('Win Rate by First Route Timing', fontsize=14)
        ax1.set_xticks(range(len(x)))
        ax1.set_xticklabels(x)
        ax1.grid(True, alpha=0.3, axis='y')

        for bar, rate in zip(bars, rates):
            ax1.annotate(f'{rate:.1%}',
                        xy=(bar.get_x() + bar.get_width() / 2, rate),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

        # Hand size distribution
        if winner_hands and loser_hands:
            ax2.hist([winner_hands, loser_hands], bins=20, label=['Winners', 'Losers'],
                    color=['#27ae60', '#c0392b'], alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Average Hand Size', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title('Hand Size Distribution: Winners vs Losers', fontsize=14)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            path = os.path.join(self.output_dir, 'timing_analysis.png')
            plt.savefig(path, dpi=150)
            print(f"Saved: {path}")

        return fig

    def plot_archetype_performance(self, save: bool = True):
        """Plot final archetype performance summary."""
        if not check_matplotlib() or not self.final_results:
            return None

        win_rates = self.final_results.get('final_win_rates', {})
        if not win_rates:
            return None

        # Sort by win rate
        sorted_archs = sorted(win_rates.items(), key=lambda x: -x[1])
        archetypes = [a for a, _ in sorted_archs]
        rates = [r for _, r in sorted_archs]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(archetypes)))[::-1]
        bars = ax.barh(range(len(archetypes)), rates, color=colors, edgecolor='black')

        ax.set_xlabel('Win Rate', fontsize=12)
        ax.set_ylabel('Archetype', fontsize=12)
        ax.set_title('Archetype Win Rates (Ranked)', fontsize=14)
        ax.set_yticks(range(len(archetypes)))
        ax.set_yticklabels([a.replace('_', ' ').title() for a in archetypes])
        ax.grid(True, alpha=0.3, axis='x')

        # Add expected rate line
        expected = 1.0 / len(archetypes) if archetypes else 0.25
        ax.axvline(x=expected, color='red', linestyle='--', linewidth=2,
                  label=f'Expected ({expected:.1%})')
        ax.legend()

        for bar, rate in zip(bars, rates):
            ax.annotate(f'{rate:.1%}',
                       xy=(rate, bar.get_y() + bar.get_height() / 2),
                       xytext=(5, 0),
                       textcoords="offset points",
                       ha='left', va='center', fontsize=10, fontweight='bold')

        plt.tight_layout()

        if save:
            path = os.path.join(self.output_dir, 'archetype_performance.png')
            plt.savefig(path, dpi=150)
            print(f"Saved: {path}")

        return fig

    def generate_all_plots(self):
        """Generate all visualization plots."""
        print("\nGenerating visualizations...")

        plots = [
            ('Win Rates Over Time', self.plot_win_rates_over_time),
            ('Tracking Comparison', self.plot_tracking_comparison),
            ('Destination Analysis', self.plot_destination_analysis),
            ('Timing Analysis', self.plot_timing_analysis),
            ('Archetype Performance', self.plot_archetype_performance),
        ]

        for name, plot_func in plots:
            try:
                fig = plot_func()
                if fig:
                    plt.close(fig)
                    print(f"  Generated: {name}")
            except Exception as e:
                print(f"  Error generating {name}: {e}")

        print(f"\nPlots saved to: {self.output_dir}")


def create_visualizations(results_dir: str):
    """
    Convenience function to create all visualizations.
    """
    visualizer = Visualizer(results_dir)
    visualizer.generate_all_plots()


def print_sample_game(sample_game: Dict):
    """Print a sample game playback."""
    if not sample_game:
        print("No sample game available.")
        return

    print("\n" + "=" * 60)
    print("SAMPLE GAME PLAYBACK")
    print("=" * 60)

    actions = sample_game.get('actions', [])
    total_turns = sample_game.get('total_turns', 0)

    print(f"\nTotal turns: {total_turns}")
    print("\nKey moments:\n")

    # Find interesting moments
    route_claims = [a for a in actions if 'CLAIM_ROUTE' in a.get('action', '')]
    dest_draws = [a for a in actions if 'DRAW_DESTINATIONS' in a.get('action', '')]

    print(f"Routes claimed: {len(route_claims)}")
    print(f"Destination draws: {len(dest_draws)}")

    # Show first few route claims
    print("\nFirst 5 route claims:")
    for i, action in enumerate(route_claims[:5]):
        print(f"  Turn {action['turn']}, Player {action['player']}: {action['action']}")

    # Show hand size progression if available
    hand_sizes = sample_game.get('hand_sizes', {})
    if hand_sizes:
        print("\nHand size progression (player 0):")
        sizes = hand_sizes.get('0', hand_sizes.get(0, []))[:20]
        if sizes:
            print(f"  First 20 turns: {sizes}")

    print("\n" + "=" * 60)
