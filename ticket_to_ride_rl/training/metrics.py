"""
Metrics collection and tracking for training analysis.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import json
import os


@dataclass
class GameMetrics:
    """Metrics collected from a single game."""
    game_id: int
    num_players: int
    winner: int
    final_scores: List[int]
    turn_count: int

    # Per-player metrics
    player_archetypes: List[str]
    player_tracking: List[bool]

    # Route metrics
    routes_claimed: List[int]  # Per player
    route_points: List[int]

    # Destination metrics
    destinations_drawn: List[int]
    destinations_completed: List[int]
    destination_points: List[int]
    destination_penalties: List[int]

    # Hand metrics
    max_hand_sizes: List[int]
    avg_hand_sizes: List[float]

    # Timing metrics
    first_route_turns: List[Optional[int]]  # Turn when first route was claimed

    # Longest route
    longest_routes: List[int]
    longest_route_winner: int

    # Cards drawn
    cards_drawn: List[int]

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'game_id': self.game_id,
            'num_players': self.num_players,
            'winner': self.winner,
            'final_scores': self.final_scores,
            'turn_count': self.turn_count,
            'player_archetypes': self.player_archetypes,
            'player_tracking': self.player_tracking,
            'routes_claimed': self.routes_claimed,
            'route_points': self.route_points,
            'destinations_drawn': self.destinations_drawn,
            'destinations_completed': self.destinations_completed,
            'destination_points': self.destination_points,
            'destination_penalties': self.destination_penalties,
            'max_hand_sizes': self.max_hand_sizes,
            'avg_hand_sizes': self.avg_hand_sizes,
            'first_route_turns': self.first_route_turns,
            'longest_routes': self.longest_routes,
            'longest_route_winner': self.longest_route_winner,
            'cards_drawn': self.cards_drawn,
        }


class MetricsCollector:
    """Collects and aggregates training metrics."""

    def __init__(self):
        self.games: List[GameMetrics] = []
        self.game_counter = 0

        # Aggregate statistics
        self.archetype_wins: Dict[str, int] = defaultdict(int)
        self.archetype_games: Dict[str, int] = defaultdict(int)
        self.archetype_scores: Dict[str, List[int]] = defaultdict(list)

        # Tracking vs blind comparison
        self.tracking_wins: Dict[str, int] = defaultdict(int)
        self.tracking_games: Dict[str, int] = defaultdict(int)

        # Temporal statistics (per batch of games)
        self.batch_stats: List[Dict] = []
        self.current_batch: List[GameMetrics] = []
        self.batch_size = 100

        # Detailed game logs for sample playback
        self.sample_games: List[Dict] = []
        self.max_sample_games = 10

    def record_game(self, metrics: GameMetrics):
        """Record metrics from a completed game."""
        self.games.append(metrics)
        self.current_batch.append(metrics)
        self.game_counter += 1

        # Update archetype statistics
        winner_archetype = metrics.player_archetypes[metrics.winner]
        self.archetype_wins[winner_archetype] += 1

        for i, archetype in enumerate(metrics.player_archetypes):
            self.archetype_games[archetype] += 1
            self.archetype_scores[archetype].append(metrics.final_scores[i])

            # Tracking stats
            tracking_key = f"{archetype}_{'tracking' if metrics.player_tracking[i] else 'blind'}"
            self.tracking_games[tracking_key] += 1
            if i == metrics.winner:
                self.tracking_wins[tracking_key] += 1

        # Check for batch completion
        if len(self.current_batch) >= self.batch_size:
            self._compute_batch_stats()

    def _compute_batch_stats(self):
        """Compute statistics for the current batch."""
        if not self.current_batch:
            return

        # Win rates by archetype
        batch_wins = defaultdict(int)
        batch_games = defaultdict(int)

        for game in self.current_batch:
            winner_archetype = game.player_archetypes[game.winner]
            batch_wins[winner_archetype] += 1
            for archetype in game.player_archetypes:
                batch_games[archetype] += 1

        win_rates = {
            arch: batch_wins[arch] / batch_games[arch]
            for arch in batch_games
        }

        # Average scores
        avg_scores = {}
        for arch in batch_games:
            scores = [
                game.final_scores[i]
                for game in self.current_batch
                for i, a in enumerate(game.player_archetypes)
                if a == arch
            ]
            avg_scores[arch] = np.mean(scores) if scores else 0

        # Destination statistics
        dest_stats = self._compute_destination_stats(self.current_batch)

        # Timing statistics
        timing_stats = self._compute_timing_stats(self.current_batch)

        stats = {
            'game_range': (self.game_counter - len(self.current_batch), self.game_counter),
            'win_rates': win_rates,
            'avg_scores': avg_scores,
            'destination_stats': dest_stats,
            'timing_stats': timing_stats,
        }

        self.batch_stats.append(stats)
        self.current_batch = []

    def _compute_destination_stats(self, games: List[GameMetrics]) -> Dict:
        """Compute destination-related statistics."""
        # Group by number of destinations taken
        dest_win_rates = defaultdict(lambda: {'wins': 0, 'games': 0})

        for game in games:
            for i, num_dest in enumerate(game.destinations_drawn):
                key = min(num_dest, 6)  # Cap at 6+
                dest_win_rates[key]['games'] += 1
                if i == game.winner:
                    dest_win_rates[key]['wins'] += 1

        # Completion rate vs win rate correlation
        completion_rates = []
        win_flags = []

        for game in games:
            for i in range(game.num_players):
                if game.destinations_drawn[i] > 0:
                    rate = game.destinations_completed[i] / game.destinations_drawn[i]
                    completion_rates.append(rate)
                    win_flags.append(1 if i == game.winner else 0)

        correlation = np.corrcoef(completion_rates, win_flags)[0, 1] if completion_rates else 0

        return {
            'win_rate_by_destinations': {
                k: v['wins'] / v['games'] if v['games'] > 0 else 0
                for k, v in dest_win_rates.items()
            },
            'completion_win_correlation': float(correlation),
            'avg_destinations_winner': np.mean([
                g.destinations_drawn[g.winner] for g in games
            ]),
            'avg_destinations_loser': np.mean([
                g.destinations_drawn[i]
                for g in games
                for i in range(g.num_players)
                if i != g.winner
            ]) if games else 0,
        }

    def _compute_timing_stats(self, games: List[GameMetrics]) -> Dict:
        """Compute timing-related statistics."""
        # First route timing vs win rate
        first_route_win = defaultdict(lambda: {'wins': 0, 'games': 0})

        for game in games:
            for i, turn in enumerate(game.first_route_turns):
                if turn is not None:
                    # Bucket into early (0-5), mid (6-15), late (16+)
                    if turn <= 5:
                        bucket = 'early'
                    elif turn <= 15:
                        bucket = 'mid'
                    else:
                        bucket = 'late'

                    first_route_win[bucket]['games'] += 1
                    if i == game.winner:
                        first_route_win[bucket]['wins'] += 1

        # Average hand size for winners vs losers
        winner_hand_sizes = [g.avg_hand_sizes[g.winner] for g in games]
        loser_hand_sizes = [
            g.avg_hand_sizes[i]
            for g in games
            for i in range(g.num_players)
            if i != g.winner
        ]

        return {
            'first_route_timing_win_rate': {
                k: v['wins'] / v['games'] if v['games'] > 0 else 0
                for k, v in first_route_win.items()
            },
            'avg_hand_size_winner': np.mean(winner_hand_sizes) if winner_hand_sizes else 0,
            'avg_hand_size_loser': np.mean(loser_hand_sizes) if loser_hand_sizes else 0,
        }

    def record_sample_game(self, game_log: Dict):
        """Record a detailed game log for later playback."""
        if len(self.sample_games) < self.max_sample_games:
            self.sample_games.append(game_log)

    def get_win_rates(self) -> Dict[str, float]:
        """Get current win rates by archetype."""
        return {
            arch: self.archetype_wins[arch] / self.archetype_games[arch]
            for arch in self.archetype_games
            if self.archetype_games[arch] > 0
        }

    def get_tracking_comparison(self) -> Dict[str, Dict]:
        """Compare tracking vs blind performance."""
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

    def get_summary(self) -> Dict:
        """Get summary of all collected metrics."""
        return {
            'total_games': self.game_counter,
            'win_rates': self.get_win_rates(),
            'tracking_comparison': self.get_tracking_comparison(),
            'avg_scores': {
                arch: np.mean(scores)
                for arch, scores in self.archetype_scores.items()
            },
            'batch_stats': self.batch_stats[-10:] if self.batch_stats else [],
        }

    def save(self, path: str):
        """Save metrics to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        data = {
            'games': [g.to_dict() for g in self.games[-1000:]],  # Last 1000 games
            'summary': self.get_summary(),
            'sample_games': self.sample_games,
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        """Load metrics from file."""
        with open(path, 'r') as f:
            data = json.load(f)

        # Reconstruct aggregates from loaded games
        for game_dict in data.get('games', []):
            metrics = GameMetrics(**game_dict)
            self.record_game(metrics)

        self.sample_games = data.get('sample_games', [])


class GameLogger:
    """Logs detailed game actions for replay and analysis."""

    def __init__(self, game_id: int):
        self.game_id = game_id
        self.actions: List[Dict] = []
        self.hand_sizes: Dict[int, List[int]] = defaultdict(list)
        self.turn = 0

    def log_action(self, player_idx: int, action_type: str, details: Dict):
        """Log an action."""
        self.actions.append({
            'turn': self.turn,
            'player': player_idx,
            'action': action_type,
            'details': details,
        })

    def log_hand_sizes(self, hand_sizes: List[int]):
        """Log hand sizes at turn end."""
        for i, size in enumerate(hand_sizes):
            self.hand_sizes[i].append(size)

    def end_turn(self):
        """Mark end of turn."""
        self.turn += 1

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'game_id': self.game_id,
            'actions': self.actions,
            'hand_sizes': dict(self.hand_sizes),
            'total_turns': self.turn,
        }


def compute_metrics_from_game(game_state, player_archetypes: List[str],
                               player_tracking: List[bool],
                               hand_histories: List[List[int]],
                               game_id: int) -> GameMetrics:
    """
    Compute GameMetrics from a completed game state.
    """
    num_players = len(game_state.players)
    final_scores_data = game_state.get_final_scores()
    winner, _ = game_state.get_winner()

    # Extract per-player data
    final_scores = [0] * num_players
    route_points = [0] * num_players
    destinations_completed = [0] * num_players
    destination_points = [0] * num_players
    destination_penalties = [0] * num_players
    longest_routes = [0] * num_players

    for player_idx, score, breakdown in final_scores_data:
        final_scores[player_idx] = score
        route_points[player_idx] = breakdown['route_points']
        destination_points[player_idx] = breakdown['completed_destinations']
        destination_penalties[player_idx] = breakdown['incomplete_penalty']
        longest_routes[player_idx] = breakdown['longest_route_length']

    # Find longest route winner
    max_length = max(longest_routes)
    longest_route_winner = longest_routes.index(max_length) if max_length > 0 else -1

    # Player stats
    routes_claimed = [len(p.claimed_routes) for p in game_state.players]
    destinations_drawn = [len(p.destinations) for p in game_state.players]
    destinations_completed = [len(p.completed_destinations) for p in game_state.players]
    first_route_turns = [p.turn_first_route for p in game_state.players]
    cards_drawn = [p.cards_drawn for p in game_state.players]

    # Hand size metrics
    max_hand_sizes = [max(h) if h else 0 for h in hand_histories]
    avg_hand_sizes = [np.mean(h) if h else 0 for h in hand_histories]

    return GameMetrics(
        game_id=game_id,
        num_players=num_players,
        winner=winner,
        final_scores=final_scores,
        turn_count=game_state.turn_number,
        player_archetypes=player_archetypes,
        player_tracking=player_tracking,
        routes_claimed=routes_claimed,
        route_points=route_points,
        destinations_drawn=destinations_drawn,
        destinations_completed=destinations_completed,
        destination_points=destination_points,
        destination_penalties=destination_penalties,
        max_hand_sizes=max_hand_sizes,
        avg_hand_sizes=avg_hand_sizes,
        first_route_turns=first_route_turns,
        longest_routes=longest_routes,
        longest_route_winner=longest_route_winner,
        cards_drawn=cards_drawn,
    )
