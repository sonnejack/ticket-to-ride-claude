"""
Ticket to Ride card system - Resource cards and Destination cards.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import IntEnum
import random

from .board import RouteColor, COLOR_NAMES


class CardColor(IntEnum):
    """Card colors match route colors, plus wild."""
    RED = 1
    ORANGE = 2
    YELLOW = 3
    GREEN = 4
    BLUE = 5
    PURPLE = 6
    BLACK = 7
    WHITE = 8
    WILD = 9


# Mapping from route color to card color needed
ROUTE_TO_CARD = {
    RouteColor.RED: CardColor.RED,
    RouteColor.ORANGE: CardColor.ORANGE,
    RouteColor.YELLOW: CardColor.YELLOW,
    RouteColor.GREEN: CardColor.GREEN,
    RouteColor.BLUE: CardColor.BLUE,
    RouteColor.PURPLE: CardColor.PURPLE,
    RouteColor.BLACK: CardColor.BLACK,
    RouteColor.WHITE: CardColor.WHITE,
}


@dataclass
class DestinationCard:
    """A destination ticket with two cities and point value."""
    city1: str
    city2: str
    points: int
    card_id: int

    def __hash__(self):
        return hash(self.card_id)

    def __eq__(self, other):
        if isinstance(other, DestinationCard):
            return self.card_id == other.card_id
        return False

    def __repr__(self):
        return f"Dest({self.city1}-{self.city2}: {self.points}pts)"


class ResourceDeck:
    """The deck of resource (train) cards."""

    # Standard distribution: 12 of each color, 14 wilds = 110 cards total
    CARDS_PER_COLOR = 12
    NUM_WILDS = 14

    def __init__(self):
        """Initialize and shuffle the deck."""
        self.draw_pile: List[CardColor] = []
        self.discard_pile: List[CardColor] = []
        self.face_up: List[CardColor] = []  # 5 face-up cards
        self._initialize_deck()

    def _initialize_deck(self):
        """Create initial deck with proper distribution."""
        self.draw_pile = []
        self.discard_pile = []

        # Add colored cards
        for color in [CardColor.RED, CardColor.ORANGE, CardColor.YELLOW,
                      CardColor.GREEN, CardColor.BLUE, CardColor.PURPLE,
                      CardColor.BLACK, CardColor.WHITE]:
            self.draw_pile.extend([color] * self.CARDS_PER_COLOR)

        # Add wilds
        self.draw_pile.extend([CardColor.WILD] * self.NUM_WILDS)

        # Shuffle
        random.shuffle(self.draw_pile)

        # Deal face-up cards
        self.face_up = []
        for _ in range(5):
            if self.draw_pile:
                self.face_up.append(self.draw_pile.pop())

        # Handle 3+ wilds in face-up (reshuffle rule)
        self._check_face_up_wilds()

    def _check_face_up_wilds(self):
        """If 3+ wilds are face up, reshuffle all face-up and redraw."""
        while sum(1 for c in self.face_up if c == CardColor.WILD) >= 3:
            # Put face-up back
            self.discard_pile.extend(self.face_up)
            self.face_up = []
            # Reshuffle if needed
            self._reshuffle_if_needed()
            # Draw new face-up
            for _ in range(5):
                if self.draw_pile:
                    self.face_up.append(self.draw_pile.pop())
                else:
                    break

    def _reshuffle_if_needed(self):
        """Shuffle discard into draw pile if draw pile is empty."""
        if not self.draw_pile and self.discard_pile:
            self.draw_pile = self.discard_pile[:]
            self.discard_pile = []
            random.shuffle(self.draw_pile)

    def draw_blind(self) -> Optional[CardColor]:
        """Draw a card from the deck (blind draw)."""
        self._reshuffle_if_needed()
        if self.draw_pile:
            return self.draw_pile.pop()
        return None

    def draw_face_up(self, index: int) -> Optional[CardColor]:
        """Draw a face-up card and replace it."""
        if 0 <= index < len(self.face_up):
            card = self.face_up[index]

            # Replace the card
            self._reshuffle_if_needed()
            if self.draw_pile:
                self.face_up[index] = self.draw_pile.pop()
            else:
                self.face_up.pop(index)

            # Check for 3 wilds rule
            self._check_face_up_wilds()

            return card
        return None

    def discard(self, cards: List[CardColor]):
        """Add cards to the discard pile."""
        self.discard_pile.extend(cards)

    def cards_remaining(self) -> int:
        """Total cards available to draw."""
        return len(self.draw_pile) + len(self.discard_pile)

    def get_face_up_state(self) -> List[CardColor]:
        """Get current face-up cards."""
        return self.face_up.copy()

    def to_numpy(self) -> np.ndarray:
        """Face-up cards as numpy array (5 elements, 0-9 values)."""
        state = np.zeros(5, dtype=np.int8)
        for i, card in enumerate(self.face_up):
            state[i] = int(card)
        return state

    def reset(self):
        """Reset the deck."""
        self._initialize_deck()


class DestinationDeck:
    """The deck of destination (ticket) cards."""

    # All 30 destination cards from the USA map
    DESTINATIONS = [
        ("Boston", "Miami", 12),
        ("Calgary", "Phoenix", 13),
        ("Calgary", "Salt Lake City", 7),
        ("Chicago", "New Orleans", 7),
        ("Chicago", "Santa Fe", 9),
        ("Dallas", "New York", 11),
        ("Denver", "El Paso", 4),
        ("Denver", "Pittsburgh", 11),
        ("Duluth", "El Paso", 10),
        ("Duluth", "Houston", 8),
        ("Helena", "Los Angeles", 8),
        ("Kansas City", "Houston", 5),
        ("Los Angeles", "Chicago", 16),
        ("Los Angeles", "Miami", 20),
        ("Los Angeles", "New York", 21),
        ("Montreal", "Atlanta", 9),
        ("Montreal", "New Orleans", 13),
        ("New York", "Atlanta", 6),
        ("Portland", "Nashville", 17),
        ("Portland", "Phoenix", 11),
        ("San Francisco", "Atlanta", 17),
        ("Sault St. Marie", "Nashville", 8),
        ("Sault St. Marie", "Oklahoma City", 9),
        ("Seattle", "Los Angeles", 9),
        ("Seattle", "New York", 22),
        ("Toronto", "Miami", 10),
        ("Vancouver", "Montreal", 20),
    ]

    def __init__(self):
        """Initialize the destination deck."""
        self.cards: List[DestinationCard] = []
        self.draw_pile: List[DestinationCard] = []
        self._initialize_deck()

    def _initialize_deck(self):
        """Create all destination cards and shuffle."""
        self.cards = []
        for i, (city1, city2, points) in enumerate(self.DESTINATIONS):
            card = DestinationCard(city1=city1, city2=city2, points=points, card_id=i)
            self.cards.append(card)

        self.draw_pile = self.cards.copy()
        random.shuffle(self.draw_pile)

    def draw(self, count: int = 3) -> List[DestinationCard]:
        """Draw destination cards (typically 3 at start, or when drawing)."""
        drawn = []
        for _ in range(min(count, len(self.draw_pile))):
            drawn.append(self.draw_pile.pop())
        return drawn

    def return_cards(self, cards: List[DestinationCard]):
        """Return unchosen cards to bottom of deck."""
        # In standard rules, unchosen destination cards go to bottom
        for card in cards:
            self.draw_pile.insert(0, card)

    def cards_remaining(self) -> int:
        """Number of destination cards left."""
        return len(self.draw_pile)

    def reset(self):
        """Reset the deck."""
        self._initialize_deck()


class Hand:
    """A player's hand of resource cards."""

    def __init__(self):
        """Initialize empty hand."""
        # Count of each card color
        self.cards = np.zeros(10, dtype=np.int8)  # Index 0 unused, 1-9 for colors

    def add(self, color: CardColor, count: int = 1):
        """Add cards to hand."""
        self.cards[int(color)] += count

    def remove(self, color: CardColor, count: int = 1) -> bool:
        """Remove cards from hand. Returns True if successful."""
        if self.cards[int(color)] >= count:
            self.cards[int(color)] -= count
            return True
        return False

    def count(self, color: CardColor) -> int:
        """Count cards of a specific color."""
        return int(self.cards[int(color)])

    def total(self) -> int:
        """Total cards in hand."""
        return int(np.sum(self.cards))

    def can_claim_route(self, length: int, route_color: RouteColor) -> List[Tuple[CardColor, int, int]]:
        """
        Check if hand can claim a route of given length and color.
        Returns list of valid (card_color, num_colored, num_wilds) combinations.
        """
        valid_combos = []
        num_wilds = self.count(CardColor.WILD)

        if route_color == RouteColor.ANY:
            # Gray route - can use any single color
            for color in [CardColor.RED, CardColor.ORANGE, CardColor.YELLOW,
                          CardColor.GREEN, CardColor.BLUE, CardColor.PURPLE,
                          CardColor.BLACK, CardColor.WHITE]:
                num_color = self.count(color)
                for colored_used in range(min(num_color, length) + 1):
                    wilds_needed = length - colored_used
                    if wilds_needed <= num_wilds:
                        valid_combos.append((color, colored_used, wilds_needed))
            # All wilds
            if num_wilds >= length:
                valid_combos.append((CardColor.WILD, 0, length))
        else:
            # Colored route - must use matching color or wilds
            card_color = ROUTE_TO_CARD[route_color]
            num_color = self.count(card_color)

            for colored_used in range(min(num_color, length) + 1):
                wilds_needed = length - colored_used
                if wilds_needed <= num_wilds:
                    valid_combos.append((card_color, colored_used, wilds_needed))

        return valid_combos

    def to_numpy(self) -> np.ndarray:
        """Return hand as numpy array."""
        return self.cards.copy()

    def from_numpy(self, cards: np.ndarray):
        """Set hand from numpy array."""
        self.cards = cards.copy()

    def copy(self) -> 'Hand':
        """Create a copy of this hand."""
        new_hand = Hand()
        new_hand.cards = self.cards.copy()
        return new_hand

    def clear(self):
        """Remove all cards."""
        self.cards.fill(0)

    def get_cards_list(self) -> List[CardColor]:
        """Get cards as a list (for discarding)."""
        cards = []
        for color in CardColor:
            cards.extend([color] * self.count(color))
        return cards

    def __repr__(self):
        parts = []
        for color in CardColor:
            c = self.count(color)
            if c > 0:
                parts.append(f"{COLOR_NAMES.get(RouteColor(int(color)), str(color))}:{c}")
        return f"Hand({', '.join(parts) or 'empty'})"
