"""
Probabilistic opponent hand tracking using Bayesian inference.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from ticket_to_ride_rl.game import CardColor, RouteColor, ROUTE_TO_CARD


@dataclass
class CardDistribution:
    """Probability distribution over card counts for a single color."""
    # Probabilities for having 0, 1, 2, ... cards of this color
    probs: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    max_cards: int = 20

    def expected_count(self) -> float:
        """Expected number of cards."""
        return np.sum(np.arange(len(self.probs)) * self.probs)

    def probability_at_least(self, n: int) -> float:
        """Probability of having at least n cards."""
        if n >= len(self.probs):
            return 0.0
        return np.sum(self.probs[n:])

    def update_draw(self, color_drawn: bool):
        """Update distribution after observing a draw."""
        if color_drawn:
            # Shift distribution right (gained a card)
            new_probs = np.zeros(min(len(self.probs) + 1, self.max_cards + 1))
            new_probs[1:len(self.probs)+1] = self.probs[:min(len(self.probs), self.max_cards)]
            self.probs = new_probs
        # If not this color, no update needed for this distribution

    def update_play(self, cards_played: int):
        """Update distribution after observing cards played of this color."""
        if cards_played <= 0:
            return

        # Shift distribution left (lost cards)
        new_probs = np.zeros(len(self.probs))
        for i in range(len(self.probs)):
            if i + cards_played < len(self.probs):
                new_probs[i] = self.probs[i + cards_played]

        # Normalize
        total = np.sum(new_probs)
        if total > 0:
            self.probs = new_probs / total
        else:
            # Reset to uniform if we lost all probability
            self.probs = np.array([1.0])


class HandTracker:
    """
    Tracks probability distributions of opponent hands based on observations.
    Uses Bayesian updating as cards are drawn and played.
    """

    # Card counts in deck (12 each color, 14 wilds)
    CARDS_PER_COLOR = 12
    NUM_WILDS = 14

    def __init__(self, num_players: int, player_idx: int):
        """
        Initialize tracker for a specific player's perspective.

        Args:
            num_players: Total players in game
            player_idx: Index of the player doing the tracking (observer)
        """
        self.num_players = num_players
        self.player_idx = player_idx

        # Track each opponent's hand distribution
        # For each opponent, for each card color, we have a distribution
        self.opponent_distributions: Dict[int, Dict[CardColor, CardDistribution]] = {}

        # Track cards known to be out of deck (in hands, discarded, etc.)
        self.cards_in_play: np.ndarray = np.zeros(10, dtype=np.int32)  # By color

        # Track total cards each opponent has
        self.opponent_card_counts: Dict[int, int] = {}

        self.reset()

    def reset(self):
        """Reset all tracking."""
        self.opponent_distributions = {}
        self.opponent_card_counts = {}

        for i in range(self.num_players):
            if i != self.player_idx:
                self.opponent_distributions[i] = {}
                for color in CardColor:
                    self.opponent_distributions[i][color] = CardDistribution()
                self.opponent_card_counts[i] = 4  # Initial hand size

        self.cards_in_play = np.zeros(10, dtype=np.int32)

    def observe_blind_draw(self, opponent_idx: int):
        """
        Observe an opponent drawing a blind card.
        We don't know what they got, but we know they have one more card.
        """
        if opponent_idx == self.player_idx:
            return

        self.opponent_card_counts[opponent_idx] += 1

        # Update probability distributions
        # Each color has some probability of being drawn based on remaining deck
        deck_composition = self._estimate_deck_composition()
        total_deck = sum(deck_composition.values())

        if total_deck == 0:
            return

        # For each color, probability it was drawn
        for color in CardColor:
            prob_this_color = deck_composition.get(color, 0) / total_deck
            if prob_this_color > 0:
                self._probabilistic_add_card(opponent_idx, color, prob_this_color)

    def observe_face_up_draw(self, opponent_idx: int, card: CardColor):
        """
        Observe an opponent drawing a specific face-up card.
        This is known information.
        """
        if opponent_idx == self.player_idx:
            return

        self.opponent_card_counts[opponent_idx] += 1
        self.cards_in_play[int(card)] += 1

        # Definitely add this card to their distribution
        dist = self.opponent_distributions[opponent_idx][card]
        dist.update_draw(True)

    def observe_route_claim(self, opponent_idx: int, route_length: int,
                            route_color: RouteColor, cards_used: Optional[List[CardColor]] = None):
        """
        Observe an opponent claiming a route.

        Args:
            opponent_idx: Which opponent
            route_length: Length of route (cards needed)
            route_color: Color of the route
            cards_used: If known, the exact cards used
        """
        if opponent_idx == self.player_idx:
            return

        self.opponent_card_counts[opponent_idx] -= route_length

        if cards_used is not None:
            # We know exactly what was played
            for card in cards_used:
                self.opponent_distributions[opponent_idx][card].update_play(1)
                self.cards_in_play[int(card)] -= 1  # Back to discard
        else:
            # Infer what was likely played
            self._infer_cards_played(opponent_idx, route_length, route_color)

    def _infer_cards_played(self, opponent_idx: int, route_length: int, route_color: RouteColor):
        """Infer which cards were likely used to claim a route."""
        distributions = self.opponent_distributions[opponent_idx]

        if route_color == RouteColor.ANY:
            # Gray route - could be any color + wilds
            # Use expected values to estimate
            best_color = None
            best_expected = 0

            for color in [CardColor.RED, CardColor.ORANGE, CardColor.YELLOW,
                          CardColor.GREEN, CardColor.BLUE, CardColor.PURPLE,
                          CardColor.BLACK, CardColor.WHITE]:
                expected = distributions[color].expected_count()
                if expected > best_expected:
                    best_expected = expected
                    best_color = color

            if best_color:
                # Assume they used this color + wilds if needed
                color_count = min(route_length, int(best_expected))
                wild_count = route_length - color_count

                distributions[best_color].update_play(color_count)
                distributions[CardColor.WILD].update_play(wild_count)
        else:
            # Colored route
            card_color = ROUTE_TO_CARD[route_color]
            expected = distributions[card_color].expected_count()
            color_count = min(route_length, int(expected))
            wild_count = route_length - color_count

            distributions[card_color].update_play(color_count)
            distributions[CardColor.WILD].update_play(wild_count)

    def _probabilistic_add_card(self, opponent_idx: int, color: CardColor, probability: float):
        """Add a card to opponent's distribution with given probability."""
        dist = self.opponent_distributions[opponent_idx][color]

        # Create new probability distribution
        new_probs = np.zeros(len(dist.probs) + 1)

        # P(n cards now) = P(n-1 before) * prob_drew + P(n before) * (1-prob_drew)
        for i in range(len(new_probs)):
            if i > 0 and i - 1 < len(dist.probs):
                new_probs[i] += dist.probs[i-1] * probability
            if i < len(dist.probs):
                new_probs[i] += dist.probs[i] * (1 - probability)

        dist.probs = new_probs / np.sum(new_probs)

    def _estimate_deck_composition(self) -> Dict[CardColor, int]:
        """Estimate remaining cards in deck based on what we've tracked."""
        composition = {}
        for color in CardColor:
            if color == CardColor.WILD:
                total = self.NUM_WILDS
            else:
                total = self.CARDS_PER_COLOR
            remaining = max(0, total - self.cards_in_play[int(color)])
            composition[color] = remaining
        return composition

    def get_opponent_hand_estimate(self, opponent_idx: int) -> np.ndarray:
        """
        Get expected hand composition for an opponent.
        Returns array of expected counts per color.
        """
        if opponent_idx == self.player_idx or opponent_idx not in self.opponent_distributions:
            return np.zeros(10)

        estimates = np.zeros(10)
        for color in CardColor:
            estimates[int(color)] = self.opponent_distributions[opponent_idx][color].expected_count()

        return estimates

    def probability_opponent_can_claim(self, opponent_idx: int, route_length: int,
                                        route_color: RouteColor) -> float:
        """
        Estimate probability an opponent can claim a specific route.
        """
        if opponent_idx == self.player_idx:
            return 0.0

        dist = self.opponent_distributions[opponent_idx]

        if route_color == RouteColor.ANY:
            # Need route_length of any single color + wilds
            max_prob = 0.0
            for color in [CardColor.RED, CardColor.ORANGE, CardColor.YELLOW,
                          CardColor.GREEN, CardColor.BLUE, CardColor.PURPLE,
                          CardColor.BLACK, CardColor.WHITE]:
                prob = self._probability_can_pay(dist, color, route_length)
                max_prob = max(max_prob, prob)
            return max_prob
        else:
            card_color = ROUTE_TO_CARD[route_color]
            return self._probability_can_pay(dist, card_color, route_length)

    def _probability_can_pay(self, distributions: Dict[CardColor, CardDistribution],
                              color: CardColor, length: int) -> float:
        """Calculate probability of having enough cards to pay for a route."""
        color_dist = distributions[color]
        wild_dist = distributions[CardColor.WILD]

        # Sum over all valid combinations
        prob = 0.0
        for num_color in range(length + 1):
            wilds_needed = length - num_color
            p_color = color_dist.probability_at_least(num_color)
            p_wilds = wild_dist.probability_at_least(wilds_needed)
            prob += p_color * p_wilds

        return min(1.0, prob)

    def to_observation(self) -> np.ndarray:
        """
        Convert tracking information to observation vector.
        Used as input to neural network.
        """
        obs = []

        for i in range(self.num_players):
            if i != self.player_idx:
                # Expected hand composition
                estimates = self.get_opponent_hand_estimate(i)
                obs.extend(estimates)

                # Card count
                obs.append(self.opponent_card_counts.get(i, 0))

        return np.array(obs, dtype=np.float32)

    def get_blocking_priorities(self, routes: List, opponent_destinations: Optional[Dict] = None
                                  ) -> List[Tuple[int, float]]:
        """
        Get routes that should be prioritized for blocking.
        Returns list of (route_id, priority_score) sorted by priority.
        """
        priorities = []

        for route in routes:
            if route.claimed_by is not None:
                continue

            # Check probability that each opponent wants/needs this route
            block_score = 0.0

            for opp_idx in range(self.num_players):
                if opp_idx == self.player_idx:
                    continue

                # Can they claim it?
                can_claim_prob = self.probability_opponent_can_claim(
                    opp_idx, route.length, route.color
                )

                # Weight by route value (longer routes more valuable to block)
                route_value = route.points()
                block_score += can_claim_prob * route_value

            priorities.append((route.route_id, block_score))

        return sorted(priorities, key=lambda x: -x[1])


class SimpleHandTracker:
    """
    Simplified hand tracker that just counts cards without full Bayesian inference.
    Faster but less accurate.
    """

    def __init__(self, num_players: int, player_idx: int):
        self.num_players = num_players
        self.player_idx = player_idx
        self.opponent_card_counts = {}
        self.opponent_known_cards = {}
        self.reset()

    def reset(self):
        for i in range(self.num_players):
            if i != self.player_idx:
                self.opponent_card_counts[i] = 4
                self.opponent_known_cards[i] = np.zeros(10, dtype=np.int32)

    def observe_blind_draw(self, opponent_idx: int):
        if opponent_idx != self.player_idx:
            self.opponent_card_counts[opponent_idx] += 1

    def observe_face_up_draw(self, opponent_idx: int, card: CardColor):
        if opponent_idx != self.player_idx:
            self.opponent_card_counts[opponent_idx] += 1
            self.opponent_known_cards[opponent_idx][int(card)] += 1

    def observe_route_claim(self, opponent_idx: int, route_length: int,
                            route_color: RouteColor, cards_used: Optional[List[CardColor]] = None):
        if opponent_idx != self.player_idx:
            self.opponent_card_counts[opponent_idx] -= route_length
            if cards_used:
                for card in cards_used:
                    self.opponent_known_cards[opponent_idx][int(card)] = max(
                        0, self.opponent_known_cards[opponent_idx][int(card)] - 1
                    )

    def to_observation(self) -> np.ndarray:
        obs = []
        for i in range(self.num_players):
            if i != self.player_idx:
                obs.extend(self.opponent_known_cards[i])
                obs.append(self.opponent_card_counts[i])
        return np.array(obs, dtype=np.float32)
