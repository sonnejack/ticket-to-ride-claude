"""
Player state management for Ticket to Ride.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Set, Optional
from .cards import Hand, DestinationCard, CardColor
from .board import Route


@dataclass
class Player:
    """A player's complete state in the game."""
    player_idx: int
    trains_remaining: int = 45
    hand: Hand = field(default_factory=Hand)
    destinations: List[DestinationCard] = field(default_factory=list)
    completed_destinations: Set[int] = field(default_factory=set)  # card_ids
    claimed_routes: List[Route] = field(default_factory=list)
    score: int = 0

    # Tracking for metrics
    cards_drawn: int = 0
    routes_claimed: int = 0
    destinations_drawn: int = 0
    turn_first_route: Optional[int] = None  # Turn number when first route was claimed

    def add_destination(self, card: DestinationCard):
        """Add a destination card to the player's hand."""
        self.destinations.append(card)
        self.destinations_drawn += 1

    def add_card(self, color: CardColor):
        """Add a resource card to the player's hand."""
        self.hand.add(color)
        self.cards_drawn += 1

    def claim_route(self, route: Route, cards_used: List[CardColor], turn_number: int) -> int:
        """
        Claim a route, removing cards and trains.
        Returns points earned.
        """
        # Remove cards from hand
        for card in cards_used:
            self.hand.remove(card)

        # Use trains
        self.trains_remaining -= route.length

        # Track route
        self.claimed_routes.append(route)
        self.routes_claimed += 1

        # Track first route timing
        if self.turn_first_route is None:
            self.turn_first_route = turn_number

        # Add points
        points = route.points()
        self.score += points

        return points

    def check_destinations(self, board) -> tuple:
        """
        Check which destinations are completed.
        Returns (completed_points, incomplete_penalty).
        """
        completed_points = 0
        incomplete_penalty = 0

        for dest in self.destinations:
            if board.cities_connected(self.player_idx, dest.city1, dest.city2):
                if dest.card_id not in self.completed_destinations:
                    self.completed_destinations.add(dest.card_id)
                completed_points += dest.points
            else:
                incomplete_penalty += dest.points

        return completed_points, incomplete_penalty

    def final_score(self, board, longest_route_bonus: bool = False) -> int:
        """Calculate final score including destinations and bonuses."""
        # Route points already tracked in self.score
        completed_pts, incomplete_penalty = self.check_destinations(board)

        final = self.score + completed_pts - incomplete_penalty
        if longest_route_bonus:
            final += 10

        return final

    def can_take_turn(self) -> bool:
        """Check if player can take any action (has trains and game isn't over)."""
        return self.trains_remaining > 0

    def get_hand_size(self) -> int:
        """Get total cards in hand."""
        return self.hand.total()

    def to_observation(self) -> dict:
        """Convert player state to observation dict."""
        return {
            'hand': self.hand.to_numpy(),
            'trains': self.trains_remaining,
            'score': self.score,
            'num_destinations': len(self.destinations),
            'num_completed': len(self.completed_destinations),
            'num_routes': len(self.claimed_routes),
        }

    def reset(self):
        """Reset player for new game."""
        self.trains_remaining = 45
        self.hand = Hand()
        self.destinations = []
        self.completed_destinations = set()
        self.claimed_routes = []
        self.score = 0
        self.cards_drawn = 0
        self.routes_claimed = 0
        self.destinations_drawn = 0
        self.turn_first_route = None

    def copy(self) -> 'Player':
        """Create a copy of this player state."""
        new_player = Player(
            player_idx=self.player_idx,
            trains_remaining=self.trains_remaining,
            hand=self.hand.copy(),
            destinations=self.destinations.copy(),
            completed_destinations=self.completed_destinations.copy(),
            claimed_routes=self.claimed_routes.copy(),
            score=self.score,
            cards_drawn=self.cards_drawn,
            routes_claimed=self.routes_claimed,
            destinations_drawn=self.destinations_drawn,
            turn_first_route=self.turn_first_route,
        )
        return new_player

    def __repr__(self):
        return (f"Player({self.player_idx}: {self.trains_remaining} trains, "
                f"{self.hand.total()} cards, {self.score} pts)")


class PlayerObservation:
    """What a player can observe about another player."""

    def __init__(self, player: Player, is_self: bool = False):
        self.player_idx = player.player_idx
        self.trains_remaining = player.trains_remaining
        self.num_cards = player.hand.total()
        self.num_destinations = len(player.destinations)
        self.score = player.score  # Route points are public
        self.num_routes = len(player.claimed_routes)

        # Only visible to self
        if is_self:
            self.hand = player.hand.to_numpy()
            self.destinations = player.destinations.copy()
            self.completed_destinations = player.completed_destinations.copy()
        else:
            self.hand = None
            self.destinations = None
            self.completed_destinations = None

    def to_numpy(self, include_private: bool = False) -> np.ndarray:
        """Convert to numpy array for state representation."""
        # Public info: [trains, num_cards, num_destinations, score, num_routes]
        public = np.array([
            self.trains_remaining,
            self.num_cards,
            self.num_destinations,
            self.score,
            self.num_routes,
        ], dtype=np.float32)

        if include_private and self.hand is not None:
            return np.concatenate([public, self.hand.astype(np.float32)])
        return public
