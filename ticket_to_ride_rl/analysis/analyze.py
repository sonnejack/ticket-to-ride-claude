"""
Statistical analysis of training results.
"""
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import os


@dataclass
class DestinationAnalysis:
    """Analysis of destination card strategies."""
    optimal_initial_destinations: int
    win_rate_by_num_destinations: Dict[int, float]
    mid_game_destination_value: float
    completion_rate_correlation: float
    avg_destinations_winners: float
    avg_destinations_losers: float


@dataclass
class TrackingAnalysis:
    """Analysis of hand tracking effectiveness."""
    overall_tracking_advantage: float
    tracking_vs_blind_by_archetype: Dict[str, Dict[str, float]]
    blocking_effectiveness: float


@dataclass
class TimingAnalysis:
    """Analysis of timing strategies."""
    optimal_first_route_timing: str
    win_rate_by_timing: Dict[str, float]
    avg_hand_size_winners: float
    avg_hand_size_losers: float
    gathering_phase_length_winners: float


@dataclass
class ArchetypeAnalysis:
    """Analysis of archetype performance."""
    win_rates: Dict[str, float]
    avg_scores: Dict[str, float]
    strongest_matchups: Dict[str, str]
    weakest_matchups: Dict[str, str]


@dataclass
class FullAnalysis:
    """Complete analysis results."""
    destination: DestinationAnalysis
    tracking: TrackingAnalysis
    timing: TimingAnalysis
    archetype: ArchetypeAnalysis
    total_games: int
    key_insights: List[str]


class ResultsAnalyzer:
    """
    Analyzes training results to answer key research questions.
    """

    def __init__(self, results_dir: str):
        self.results_dir = results_dir
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

    def analyze_destinations(self) -> DestinationAnalysis:
        """
        Analyze destination card importance.

        Key questions:
        - Optimal number of destinations at game start?
        - Value of drawing destinations mid-game?
        - Correlation between completion rate and winning?
        """
        games = self.metrics.get('games', []) if self.metrics else []

        # Win rate by number of destinations
        dest_wins = defaultdict(lambda: {'wins': 0, 'games': 0})

        for game in games:
            for i, num_dest in enumerate(game.get('destinations_drawn', [])):
                bucket = min(num_dest, 6)
                dest_wins[bucket]['games'] += 1
                if i == game.get('winner', -1):
                    dest_wins[bucket]['wins'] += 1

        win_rate_by_dest = {
            k: v['wins'] / v['games'] if v['games'] > 0 else 0
            for k, v in dest_wins.items()
        }

        # Find optimal number
        if win_rate_by_dest:
            optimal = max(win_rate_by_dest.items(), key=lambda x: x[1])[0]
        else:
            optimal = 2

        # Completion rate correlation
        completion_rates = []
        win_flags = []

        for game in games:
            for i in range(game.get('num_players', 4)):
                drawn = game.get('destinations_drawn', [0]*4)[i]
                completed = game.get('destinations_completed', [0]*4)[i]
                if drawn > 0:
                    completion_rates.append(completed / drawn)
                    win_flags.append(1 if i == game.get('winner', -1) else 0)

        if completion_rates:
            correlation = np.corrcoef(completion_rates, win_flags)[0, 1]
        else:
            correlation = 0.0

        # Average destinations for winners vs losers
        winner_dests = []
        loser_dests = []

        for game in games:
            winner = game.get('winner', 0)
            for i in range(game.get('num_players', 4)):
                dest = game.get('destinations_drawn', [0]*4)[i]
                if i == winner:
                    winner_dests.append(dest)
                else:
                    loser_dests.append(dest)

        return DestinationAnalysis(
            optimal_initial_destinations=optimal,
            win_rate_by_num_destinations=win_rate_by_dest,
            mid_game_destination_value=0.0,  # Would need more detailed logs
            completion_rate_correlation=float(correlation),
            avg_destinations_winners=np.mean(winner_dests) if winner_dests else 0,
            avg_destinations_losers=np.mean(loser_dests) if loser_dests else 0,
        )

    def analyze_tracking(self) -> TrackingAnalysis:
        """
        Analyze hand tracking effectiveness.

        Key questions:
        - Do tracking bots outperform blind bots?
        - By how much per archetype?
        - Does it improve blocking effectiveness?
        """
        comparison = {}
        if self.final_results:
            comparison = self.final_results.get('tracking_comparison', {})

        # Calculate overall advantage
        tracking_wins = 0
        tracking_games = 0
        blind_wins = 0
        blind_games = 0

        for arch, modes in comparison.items():
            if 'tracking' in modes:
                tracking_wins += modes['tracking'].get('wins', 0)
                tracking_games += modes['tracking'].get('games', 0)
            if 'blind' in modes:
                blind_wins += modes['blind'].get('wins', 0)
                blind_games += modes['blind'].get('games', 0)

        tracking_rate = tracking_wins / tracking_games if tracking_games > 0 else 0
        blind_rate = blind_wins / blind_games if blind_games > 0 else 0

        # Per-archetype comparison
        by_archetype = {}
        for arch, modes in comparison.items():
            by_archetype[arch] = {
                'blind_rate': modes.get('blind', {}).get('win_rate', 0),
                'tracking_rate': modes.get('tracking', {}).get('win_rate', 0),
                'advantage': modes.get('tracking', {}).get('win_rate', 0) -
                            modes.get('blind', {}).get('win_rate', 0)
            }

        return TrackingAnalysis(
            overall_tracking_advantage=tracking_rate - blind_rate,
            tracking_vs_blind_by_archetype=by_archetype,
            blocking_effectiveness=0.0,  # Would need specific blocking metrics
        )

    def analyze_timing(self) -> TimingAnalysis:
        """
        Analyze timing strategies.

        Key questions:
        - When should you play your first route?
        - How long should the gathering phase be?
        - What's the optimal hand size to maintain?
        """
        games = self.metrics.get('games', []) if self.metrics else []

        # First route timing vs win rate
        timing_wins = defaultdict(lambda: {'wins': 0, 'games': 0})

        for game in games:
            for i, turn in enumerate(game.get('first_route_turns', [None]*4)):
                if turn is not None:
                    if turn <= 3:
                        bucket = 'very_early'
                    elif turn <= 7:
                        bucket = 'early'
                    elif turn <= 12:
                        bucket = 'mid'
                    else:
                        bucket = 'late'

                    timing_wins[bucket]['games'] += 1
                    if i == game.get('winner', -1):
                        timing_wins[bucket]['wins'] += 1

        win_rate_by_timing = {
            k: v['wins'] / v['games'] if v['games'] > 0 else 0
            for k, v in timing_wins.items()
        }

        # Find optimal timing
        if win_rate_by_timing:
            optimal_timing = max(win_rate_by_timing.items(), key=lambda x: x[1])[0]
        else:
            optimal_timing = 'mid'

        # Hand sizes
        winner_hand_sizes = []
        loser_hand_sizes = []
        winner_first_route = []

        for game in games:
            winner = game.get('winner', 0)
            for i in range(game.get('num_players', 4)):
                avg_hand = game.get('avg_hand_sizes', [0]*4)[i]
                first_route = game.get('first_route_turns', [None]*4)[i]

                if i == winner:
                    winner_hand_sizes.append(avg_hand)
                    if first_route is not None:
                        winner_first_route.append(first_route)
                else:
                    loser_hand_sizes.append(avg_hand)

        return TimingAnalysis(
            optimal_first_route_timing=optimal_timing,
            win_rate_by_timing=win_rate_by_timing,
            avg_hand_size_winners=np.mean(winner_hand_sizes) if winner_hand_sizes else 0,
            avg_hand_size_losers=np.mean(loser_hand_sizes) if loser_hand_sizes else 0,
            gathering_phase_length_winners=np.mean(winner_first_route) if winner_first_route else 0,
        )

    def analyze_archetypes(self) -> ArchetypeAnalysis:
        """
        Analyze archetype performance.
        """
        win_rates = {}
        avg_scores = {}

        if self.final_results:
            win_rates = self.final_results.get('final_win_rates', {})

        if self.metrics and 'summary' in self.metrics:
            avg_scores = self.metrics['summary'].get('avg_scores', {})

        # Matchup analysis would require more detailed game logs
        # For now, return basic stats
        return ArchetypeAnalysis(
            win_rates=win_rates,
            avg_scores=avg_scores,
            strongest_matchups={},
            weakest_matchups={},
        )

    def generate_insights(self, dest: DestinationAnalysis,
                          tracking: TrackingAnalysis,
                          timing: TimingAnalysis) -> List[str]:
        """Generate key insights from analysis."""
        insights = []

        # Destination insights
        if dest.optimal_initial_destinations:
            insights.append(
                f"Optimal initial destinations: {dest.optimal_initial_destinations} "
                f"(win rate: {dest.win_rate_by_num_destinations.get(dest.optimal_initial_destinations, 0):.1%})"
            )

        if abs(dest.completion_rate_correlation) > 0.1:
            direction = "positively" if dest.completion_rate_correlation > 0 else "negatively"
            insights.append(
                f"Destination completion rate is {direction} correlated with winning "
                f"(r={dest.completion_rate_correlation:.2f})"
            )

        # Tracking insights
        if abs(tracking.overall_tracking_advantage) > 0.01:
            better = "tracking" if tracking.overall_tracking_advantage > 0 else "blind"
            insights.append(
                f"Hand tracking provides {'advantage' if tracking.overall_tracking_advantage > 0 else 'disadvantage'} "
                f"({tracking.overall_tracking_advantage:+.1%} win rate)"
            )

        # Timing insights
        insights.append(
            f"Optimal first route timing: {timing.optimal_first_route_timing} "
            f"(win rate: {timing.win_rate_by_timing.get(timing.optimal_first_route_timing, 0):.1%})"
        )

        if abs(timing.avg_hand_size_winners - timing.avg_hand_size_losers) > 0.5:
            if timing.avg_hand_size_winners > timing.avg_hand_size_losers:
                insights.append(
                    f"Winners maintain larger hands on average "
                    f"({timing.avg_hand_size_winners:.1f} vs {timing.avg_hand_size_losers:.1f} cards)"
                )
            else:
                insights.append(
                    f"Winners play more aggressively (smaller hands: "
                    f"{timing.avg_hand_size_winners:.1f} vs {timing.avg_hand_size_losers:.1f} cards)"
                )

        return insights

    def run_full_analysis(self) -> FullAnalysis:
        """Run complete analysis."""
        dest_analysis = self.analyze_destinations()
        tracking_analysis = self.analyze_tracking()
        timing_analysis = self.analyze_timing()
        archetype_analysis = self.analyze_archetypes()

        insights = self.generate_insights(dest_analysis, tracking_analysis, timing_analysis)

        total_games = len(self.metrics.get('games', [])) if self.metrics else 0

        return FullAnalysis(
            destination=dest_analysis,
            tracking=tracking_analysis,
            timing=timing_analysis,
            archetype=archetype_analysis,
            total_games=total_games,
            key_insights=insights,
        )

    def print_report(self):
        """Print formatted analysis report."""
        analysis = self.run_full_analysis()

        print("\n" + "=" * 70)
        print("TICKET TO RIDE RL - ANALYSIS REPORT")
        print("=" * 70)
        print(f"\nTotal games analyzed: {analysis.total_games:,}")

        # Question 1: Destination Cards
        print("\n" + "-" * 70)
        print("QUESTION 1: DESTINATION CARD IMPORTANCE")
        print("-" * 70)
        dest = analysis.destination
        print(f"\nOptimal number of initial destinations: {dest.optimal_initial_destinations}")
        print("\nWin rate by number of destinations:")
        for n, rate in sorted(dest.win_rate_by_num_destinations.items()):
            print(f"  {n} destinations: {rate:.1%}")
        print(f"\nDestination completion correlation with winning: {dest.completion_rate_correlation:.3f}")
        print(f"Average destinations (winners): {dest.avg_destinations_winners:.1f}")
        print(f"Average destinations (losers): {dest.avg_destinations_losers:.1f}")

        # Question 2: Hand Tracking
        print("\n" + "-" * 70)
        print("QUESTION 2: HAND TRACKING VALUE")
        print("-" * 70)
        track = analysis.tracking
        print(f"\nOverall tracking advantage: {track.overall_tracking_advantage:+.2%}")
        print("\nBy archetype:")
        for arch, stats in sorted(track.tracking_vs_blind_by_archetype.items()):
            print(f"  {arch}: Blind={stats['blind_rate']:.1%}, "
                  f"Tracking={stats['tracking_rate']:.1%}, "
                  f"Advantage={stats['advantage']:+.1%}")

        # Question 3: Timing
        print("\n" + "-" * 70)
        print("QUESTION 3: TIMING STRATEGIES")
        print("-" * 70)
        timing = analysis.timing
        print(f"\nOptimal first route timing: {timing.optimal_first_route_timing}")
        print("\nWin rate by first route timing:")
        for bucket, rate in sorted(timing.win_rate_by_timing.items()):
            print(f"  {bucket}: {rate:.1%}")
        print(f"\nAverage hand size (winners): {timing.avg_hand_size_winners:.1f}")
        print(f"Average hand size (losers): {timing.avg_hand_size_losers:.1f}")
        print(f"Average gathering phase length (winners): {timing.gathering_phase_length_winners:.1f} turns")

        # Archetype Performance
        print("\n" + "-" * 70)
        print("ARCHETYPE PERFORMANCE")
        print("-" * 70)
        arch = analysis.archetype
        print("\nWin rates:")
        for name, rate in sorted(arch.win_rates.items(), key=lambda x: -x[1]):
            print(f"  {name}: {rate:.1%}")
        print("\nAverage scores:")
        for name, score in sorted(arch.avg_scores.items(), key=lambda x: -x[1]):
            print(f"  {name}: {score:.1f}")

        # Key Insights
        print("\n" + "-" * 70)
        print("KEY INSIGHTS")
        print("-" * 70)
        for insight in analysis.key_insights:
            print(f"  * {insight}")

        print("\n" + "=" * 70)

    def save_report(self, path: str):
        """Save analysis report to file."""
        analysis = self.run_full_analysis()

        report = {
            'total_games': analysis.total_games,
            'destination_analysis': {
                'optimal_initial': analysis.destination.optimal_initial_destinations,
                'win_rate_by_count': analysis.destination.win_rate_by_num_destinations,
                'completion_correlation': analysis.destination.completion_rate_correlation,
            },
            'tracking_analysis': {
                'overall_advantage': analysis.tracking.overall_tracking_advantage,
                'by_archetype': analysis.tracking.tracking_vs_blind_by_archetype,
            },
            'timing_analysis': {
                'optimal_timing': analysis.timing.optimal_first_route_timing,
                'win_rate_by_timing': analysis.timing.win_rate_by_timing,
                'avg_hand_winners': analysis.timing.avg_hand_size_winners,
                'avg_hand_losers': analysis.timing.avg_hand_size_losers,
            },
            'archetype_performance': {
                'win_rates': analysis.archetype.win_rates,
                'avg_scores': analysis.archetype.avg_scores,
            },
            'key_insights': analysis.key_insights,
        }

        with open(path, 'w') as f:
            json.dump(report, f, indent=2)


def analyze_results(results_dir: str, print_report: bool = True) -> FullAnalysis:
    """
    Convenience function to analyze results.
    """
    analyzer = ResultsAnalyzer(results_dir)

    if print_report:
        analyzer.print_report()

    return analyzer.run_full_analysis()
