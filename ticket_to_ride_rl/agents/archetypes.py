"""
Archetype definitions for different playing styles.
Each archetype has initial policy biases that influence early training.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from enum import Enum
import numpy as np

from ticket_to_ride_rl.game import ActionType, Action, GameState, CardColor


class ArchetypeType(Enum):
    """The five distinct archetypes."""
    SIX_SHOOTER = "six_shooter"
    INSTANT_GRATIFICATION = "instant_gratification"
    HOARDER = "hoarder"
    BLOCKER = "blocker"
    WILDCARD = "wildcard"


@dataclass
class ArchetypeConfig:
    """Configuration for an archetype's initial behavior."""
    name: str
    description: str

    # Initial destination preferences
    initial_destinations: int = 3  # How many to draw at start
    min_destinations_keep: int = 2  # Minimum to keep

    # Action type biases (higher = more likely initially)
    # These are added to policy logits before softmax
    draw_deck_bias: float = 0.0
    draw_face_up_bias: float = 0.0
    claim_route_bias: float = 0.0
    draw_destinations_bias: float = 0.0

    # Card preferences
    prefer_wilds: bool = False
    prefer_long_routes: bool = False

    # Route length preferences (for Six Shooter)
    min_route_length: int = 1  # Minimum route length to consider
    preferred_route_lengths: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])

    # Strategic biases
    uses_hand_tracking: bool = False
    blocking_priority: float = 0.0  # 0-1, how much to prioritize blocking
    mimic_leader: bool = False  # Copy the leading player's style

    # Timing preferences
    min_cards_before_playing: int = 0  # Minimum hand size before claiming routes
    end_game_aggression: float = 0.5  # How aggressively to end the game

    # Learning parameters
    initial_exploration: float = 0.3  # Initial epsilon for exploration


ARCHETYPE_CONFIGS = {
    ArchetypeType.SIX_SHOOTER: ArchetypeConfig(
        name="Six Shooter",
        description="Only claims 5-6 length routes, then 4-3 if necessary. Goes for big points.",
        initial_destinations=2,
        min_destinations_keep=2,
        draw_deck_bias=0.6,
        draw_face_up_bias=0.4,
        claim_route_bias=0.5,
        draw_destinations_bias=-0.3,
        prefer_long_routes=True,
        min_route_length=5,  # Start with 5-6 only
        preferred_route_lengths=[6, 5, 4, 3],  # Priority order
        min_cards_before_playing=5,
        end_game_aggression=0.7,
        initial_exploration=0.2,
    ),

    ArchetypeType.INSTANT_GRATIFICATION: ArchetypeConfig(
        name="Instant Gratification",
        description="Plays routes immediately when possible, minimal planning",
        initial_destinations=2,
        min_destinations_keep=2,
        draw_deck_bias=-0.2,
        draw_face_up_bias=0.0,
        claim_route_bias=1.0,
        draw_destinations_bias=-0.3,
        prefer_long_routes=False,
        min_cards_before_playing=0,
        end_game_aggression=0.8,
        initial_exploration=0.25,
    ),

    ArchetypeType.HOARDER: ArchetypeConfig(
        name="The Hoarder",
        description="Focuses on long routes, accumulates specific colors, draws 2 destinations at start",
        initial_destinations=2,
        min_destinations_keep=2,
        draw_deck_bias=0.8,
        draw_face_up_bias=0.5,
        claim_route_bias=-0.5,
        draw_destinations_bias=0.0,
        prefer_long_routes=True,
        prefer_wilds=True,
        min_cards_before_playing=12,
        end_game_aggression=0.4,
        initial_exploration=0.2,
    ),

    ArchetypeType.BLOCKER: ArchetypeConfig(
        name="The Blocker",
        description="Actively models opponents, prioritizes blocking contested routes",
        initial_destinations=2,
        min_destinations_keep=2,
        draw_deck_bias=0.2,
        draw_face_up_bias=0.3,
        claim_route_bias=0.3,
        draw_destinations_bias=0.0,
        uses_hand_tracking=True,
        blocking_priority=0.8,
        min_cards_before_playing=4,
        end_game_aggression=0.6,
        initial_exploration=0.25,
    ),

    ArchetypeType.WILDCARD: ArchetypeConfig(
        name="The Wildcard",
        description="Draws exclusively from deck (never face-up), mimics the leading player's strategy",
        initial_destinations=2,
        min_destinations_keep=2,
        draw_deck_bias=2.0,  # Strong preference for deck
        draw_face_up_bias=-5.0,  # Never draw face-up
        claim_route_bias=0.0,
        draw_destinations_bias=0.0,
        prefer_wilds=False,
        mimic_leader=True,  # Copy winning player's style
        min_cards_before_playing=0,
        end_game_aggression=0.5,
        initial_exploration=0.3,
    ),
}


class ArchetypePolicy:
    """
    Applies archetype-specific biases to action selection.
    Used to initialize and guide policy network behavior.
    """

    def __init__(self, archetype_type: ArchetypeType):
        self.archetype = archetype_type
        self.config = ARCHETYPE_CONFIGS[archetype_type]
        self._leader_archetype = None  # For Wildcard mimicking

    def _get_leader_config(self, game_state: GameState, player_idx: int) -> Optional[ArchetypeConfig]:
        """Get the config of the leading player (for Wildcard mimic)."""
        if not self.config.mimic_leader:
            return None

        # Find player with highest score (excluding self)
        best_score = -999
        best_idx = None
        for i, p in enumerate(game_state.players):
            if i != player_idx and p.score > best_score:
                best_score = p.score
                best_idx = i

        if best_idx is None:
            return None

        # Default to Hoarder-like behavior (most successful archetype)
        return ARCHETYPE_CONFIGS[ArchetypeType.HOARDER]

    def get_action_biases(self, actions: List[Action], game_state: GameState,
                           player_idx: int) -> np.ndarray:
        """
        Get bias values for each action based on archetype preferences.
        These are added to network logits before softmax.
        """
        biases = np.zeros(len(actions))
        player = game_state.players[player_idx]

        # For Wildcard, get leader's config to mimic (except for card drawing)
        mimic_config = self._get_leader_config(game_state, player_idx) if self.config.mimic_leader else None

        for i, action in enumerate(actions):
            bias = 0.0

            # Base action type biases
            if action.action_type == ActionType.DRAW_DECK:
                bias += self.config.draw_deck_bias

                # Wildcard always draws from deck
                if self.archetype == ArchetypeType.WILDCARD:
                    bias += 2.0  # Very strong preference

            elif action.action_type == ActionType.DRAW_FACE_UP:
                bias += self.config.draw_face_up_bias

                # Wild card preference
                if self.config.prefer_wilds and action.face_up_index is not None:
                    face_up = game_state.resource_deck.face_up
                    if action.face_up_index < len(face_up):
                        if face_up[action.face_up_index] == CardColor.WILD:
                            bias += 0.5

                # Wildcard archetype NEVER draws face-up
                if self.archetype == ArchetypeType.WILDCARD:
                    bias -= 10.0  # Massive penalty

            elif action.action_type == ActionType.CLAIM_ROUTE:
                # Use mimic config for route claiming if Wildcard
                active_config = mimic_config if mimic_config else self.config

                bias += active_config.claim_route_bias

                # Check hand size threshold
                if player.hand.total() < active_config.min_cards_before_playing:
                    bias -= 0.5

                # Six Shooter: Strong preference for long routes
                if self.archetype == ArchetypeType.SIX_SHOOTER and action.route_id is not None:
                    route = game_state.board.routes[action.route_id]
                    # Check if any 5-6 routes are available
                    has_long_routes = any(
                        a.action_type == ActionType.CLAIM_ROUTE and
                        a.route_id is not None and
                        game_state.board.routes[a.route_id].length >= 5
                        for a in actions
                    )

                    if route.length == 6:
                        bias += 2.0  # Highest priority
                    elif route.length == 5:
                        bias += 1.5
                    elif route.length == 4:
                        if has_long_routes:
                            bias -= 1.0  # Avoid if long routes available
                        else:
                            bias += 0.5  # OK if no long routes
                    elif route.length == 3:
                        if has_long_routes:
                            bias -= 1.5
                        else:
                            bias += 0.2
                    else:  # 1-2 length
                        bias -= 2.0  # Strongly avoid short routes

                # General long route preference
                elif self.config.prefer_long_routes and action.route_id is not None:
                    route = game_state.board.routes[action.route_id]
                    if route.length >= 5:
                        bias += 0.3
                    elif route.length <= 2:
                        bias -= 0.2

                # Instant Gratification always wants to claim
                if self.archetype == ArchetypeType.INSTANT_GRATIFICATION:
                    bias += 0.5

            elif action.action_type == ActionType.DRAW_DESTINATIONS:
                # Use mimic config for destinations if Wildcard
                active_config = mimic_config if mimic_config else self.config
                bias += active_config.draw_destinations_bias

            biases[i] = bias

        return biases

    def get_destination_keep_preferences(self, destinations, game_state: GameState,
                                          player_idx: int) -> List[float]:
        """
        Score each destination for keeping.
        Higher score = more likely to keep.
        """
        scores = []
        player = game_state.players[player_idx]

        for dest in destinations:
            score = 0.0

            # Base points value
            score += dest.points / 20.0  # Normalize

            # Six Shooter likes medium-high destinations (long routes = high points)
            if self.archetype == ArchetypeType.SIX_SHOOTER:
                if dest.points >= 15:
                    score += 0.4
                elif dest.points >= 11:
                    score += 0.2

            # Hoarder likes long destinations (higher points usually = longer)
            if self.archetype == ArchetypeType.HOARDER:
                if dest.points >= 17:
                    score += 0.3

            # Instant Gratification prefers easier destinations
            if self.archetype == ArchetypeType.INSTANT_GRATIFICATION:
                if dest.points <= 8:
                    score += 0.3
                elif dest.points >= 15:
                    score -= 0.2

            # Wildcard mimics leader - prefer moderate destinations
            if self.archetype == ArchetypeType.WILDCARD:
                if 8 <= dest.points <= 15:
                    score += 0.2

            # Check synergy with existing destinations
            existing_cities = set()
            for d in player.destinations:
                existing_cities.add(d.city1)
                existing_cities.add(d.city2)

            if dest.city1 in existing_cities or dest.city2 in existing_cities:
                score += 0.2  # Synergy bonus

            scores.append(score)

        return scores


def get_archetype_types() -> List[ArchetypeType]:
    """Get all archetype types."""
    return list(ArchetypeType)


def create_archetype_policy(archetype_type: ArchetypeType) -> ArchetypePolicy:
    """Factory function to create an archetype policy."""
    return ArchetypePolicy(archetype_type)


def get_archetype_config(archetype_type: ArchetypeType) -> ArchetypeConfig:
    """Get configuration for an archetype."""
    return ARCHETYPE_CONFIGS[archetype_type]
