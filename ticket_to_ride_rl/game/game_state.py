"""
Ticket to Ride game engine - Complete game state and rules.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from enum import IntEnum
import random

from .board import Board, Route, RouteColor
from .cards import ResourceDeck, DestinationDeck, Hand, CardColor, DestinationCard, ROUTE_TO_CARD
from .player import Player, PlayerObservation


class ActionType(IntEnum):
    """Types of actions a player can take."""
    DRAW_DECK = 0       # Draw 2 cards from deck
    DRAW_FACE_UP = 1    # Draw face-up card (may need 2nd action)
    CLAIM_ROUTE = 2     # Claim a route
    DRAW_DESTINATIONS = 3  # Draw destination cards
    KEEP_DESTINATIONS = 4  # Choose which destinations to keep (sub-action)


class GamePhase(IntEnum):
    """Current phase of the game."""
    SETUP = 0
    PLAYING = 1
    LAST_ROUND = 2
    GAME_OVER = 3


@dataclass
class Action:
    """A game action."""
    action_type: ActionType
    route_id: Optional[int] = None
    face_up_index: Optional[int] = None
    card_color: Optional[CardColor] = None  # For gray routes
    num_colored: Optional[int] = None
    num_wilds: Optional[int] = None
    destination_choices: Optional[List[int]] = None  # card_ids to keep


@dataclass
class GameConfig:
    """Game configuration."""
    num_players: int = 4
    initial_cards: int = 4
    initial_destinations: int = 3
    min_destinations_keep: int = 2
    trains_per_player: int = 45
    end_game_train_threshold: int = 2  # Triggers last round


class GameState:
    """Complete game state and rules engine."""

    def __init__(self, config: GameConfig = None):
        self.config = config or GameConfig()
        self.board = Board()
        self.resource_deck = ResourceDeck()
        self.destination_deck = DestinationDeck()

        # Players
        self.players: List[Player] = []
        for i in range(self.config.num_players):
            self.players.append(Player(player_idx=i))

        # Game state
        self.current_player = 0
        self.turn_number = 0
        self.phase = GamePhase.SETUP
        self.last_round_trigger_player: Optional[int] = None

        # Pending actions (for multi-step turns)
        self.pending_draw_count = 0  # Cards left to draw this turn
        self.pending_destinations: List[DestinationCard] = []  # Destinations to choose from
        self.drew_wild_face_up = False  # Can't draw 2nd card if drew wild

        # Game history for analysis
        self.history: List[Dict[str, Any]] = []

    def setup_game(self):
        """Initialize a new game."""
        self.board.reset()
        self.resource_deck.reset()
        self.destination_deck.reset()

        for player in self.players:
            player.reset()

            # Deal initial cards
            for _ in range(self.config.initial_cards):
                card = self.resource_deck.draw_blind()
                if card:
                    player.hand.add(card)

        self.current_player = random.randint(0, self.config.num_players - 1)
        self.turn_number = 0
        self.phase = GamePhase.SETUP
        self.last_round_trigger_player = None
        self.pending_draw_count = 0
        self.pending_destinations = []
        self.drew_wild_face_up = False
        self.history = []

    def deal_initial_destinations(self, player_idx: int) -> List[DestinationCard]:
        """Deal initial destination cards to a player."""
        cards = self.destination_deck.draw(self.config.initial_destinations)
        self.pending_destinations = cards
        return cards

    def keep_initial_destinations(self, player_idx: int, keep_card_ids: List[int]):
        """Player chooses which initial destinations to keep (must keep >= 2)."""
        if len(keep_card_ids) < self.config.min_destinations_keep:
            raise ValueError(f"Must keep at least {self.config.min_destinations_keep} destinations")

        kept = []
        returned = []
        for card in self.pending_destinations:
            if card.card_id in keep_card_ids:
                self.players[player_idx].add_destination(card)
                kept.append(card)
            else:
                returned.append(card)

        self.destination_deck.return_cards(returned)
        self.pending_destinations = []

        # Check if all players have done setup
        if all(len(p.destinations) >= self.config.min_destinations_keep for p in self.players):
            self.phase = GamePhase.PLAYING

        return kept

    def get_current_player(self) -> Player:
        """Get the current player."""
        return self.players[self.current_player]

    def get_valid_actions(self, player_idx: Optional[int] = None) -> List[Action]:
        """Get all valid actions for a player."""
        if player_idx is None:
            player_idx = self.current_player

        player = self.players[player_idx]
        actions = []

        # If we're in destination selection mode
        if self.pending_destinations:
            # Generate all valid keep combinations
            min_keep = 1 if self.phase == GamePhase.PLAYING else self.config.min_destinations_keep
            actions.extend(self._get_destination_keep_actions(min_keep))
            return actions

        # If we're mid-draw (drew 1 card, need to draw 2nd)
        if self.pending_draw_count > 0:
            if not self.drew_wild_face_up:
                # Can draw from deck
                actions.append(Action(action_type=ActionType.DRAW_DECK))
                # Can draw non-wild face-up
                for i, card in enumerate(self.resource_deck.face_up):
                    if card != CardColor.WILD:
                        actions.append(Action(action_type=ActionType.DRAW_FACE_UP, face_up_index=i))
            return actions

        # Normal turn - all action types available

        # 1. Draw from deck (always available if cards exist)
        if self.resource_deck.cards_remaining() > 0 or self.resource_deck.face_up:
            actions.append(Action(action_type=ActionType.DRAW_DECK))

        # 2. Draw face-up cards
        for i, card in enumerate(self.resource_deck.face_up):
            actions.append(Action(action_type=ActionType.DRAW_FACE_UP, face_up_index=i))

        # 3. Claim routes
        available_routes = self.board.get_available_routes_for_player(
            player_idx, self.config.num_players
        )
        for route in available_routes:
            if route.length <= player.trains_remaining:
                combos = player.hand.can_claim_route(route.length, route.color)
                for card_color, num_colored, num_wilds in combos:
                    actions.append(Action(
                        action_type=ActionType.CLAIM_ROUTE,
                        route_id=route.route_id,
                        card_color=card_color,
                        num_colored=num_colored,
                        num_wilds=num_wilds
                    ))

        # 4. Draw destinations (if cards available)
        if self.destination_deck.cards_remaining() > 0:
            actions.append(Action(action_type=ActionType.DRAW_DESTINATIONS))

        return actions

    def _get_destination_keep_actions(self, min_keep: int) -> List[Action]:
        """Generate all valid destination keep combinations."""
        from itertools import combinations

        actions = []
        card_ids = [c.card_id for c in self.pending_destinations]

        for n in range(min_keep, len(card_ids) + 1):
            for combo in combinations(card_ids, n):
                actions.append(Action(
                    action_type=ActionType.KEEP_DESTINATIONS,
                    destination_choices=list(combo)
                ))

        return actions

    def execute_action(self, action: Action) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute an action. Returns (turn_complete, info_dict).
        """
        player = self.get_current_player()
        info = {'action_type': action.action_type, 'player': self.current_player}

        if action.action_type == ActionType.DRAW_DECK:
            return self._execute_draw_deck(player, info)

        elif action.action_type == ActionType.DRAW_FACE_UP:
            return self._execute_draw_face_up(player, action, info)

        elif action.action_type == ActionType.CLAIM_ROUTE:
            return self._execute_claim_route(player, action, info)

        elif action.action_type == ActionType.DRAW_DESTINATIONS:
            return self._execute_draw_destinations(player, info)

        elif action.action_type == ActionType.KEEP_DESTINATIONS:
            return self._execute_keep_destinations(player, action, info)

        return False, {'error': 'Unknown action type'}

    def _execute_draw_deck(self, player: Player, info: Dict) -> Tuple[bool, Dict]:
        """Draw 2 cards from deck."""
        cards_drawn = []
        for _ in range(2 - self.pending_draw_count):
            card = self.resource_deck.draw_blind()
            if card:
                player.add_card(card)
                cards_drawn.append(card)

        info['cards_drawn'] = cards_drawn
        self.pending_draw_count = 0
        return True, info

    def _execute_draw_face_up(self, player: Player, action: Action, info: Dict) -> Tuple[bool, Dict]:
        """Draw a face-up card."""
        card = self.resource_deck.draw_face_up(action.face_up_index)
        if card is None:
            return False, {'error': 'Invalid face-up index'}

        player.add_card(card)
        info['card_drawn'] = card

        # Wild card rules
        if card == CardColor.WILD:
            if self.pending_draw_count == 0:
                # First draw - wild counts as both cards
                self.drew_wild_face_up = True
                return True, info
            else:
                # Shouldn't happen - wilds can't be 2nd draw
                return False, {'error': 'Cannot draw wild as second card'}
        else:
            if self.pending_draw_count == 0:
                # First draw - need second
                self.pending_draw_count = 1
                return False, info
            else:
                # Second draw - turn complete
                self.pending_draw_count = 0
                return True, info

    def _execute_claim_route(self, player: Player, action: Action, info: Dict) -> Tuple[bool, Dict]:
        """Claim a route."""
        route = self.board.routes[action.route_id]

        # Build cards to use
        cards_used = []
        if action.card_color and action.card_color != CardColor.WILD:
            cards_used.extend([action.card_color] * action.num_colored)
        cards_used.extend([CardColor.WILD] * action.num_wilds)

        # Claim the route
        points = player.claim_route(route, cards_used, self.turn_number)
        self.board.claim_route(action.route_id, self.current_player)

        # Discard used cards
        self.resource_deck.discard(cards_used)

        info['route'] = route
        info['points'] = points
        info['cards_used'] = cards_used

        # Check for end game trigger
        if player.trains_remaining <= self.config.end_game_train_threshold:
            if self.last_round_trigger_player is None:
                self.last_round_trigger_player = self.current_player
                self.phase = GamePhase.LAST_ROUND
                info['triggered_last_round'] = True

        return True, info

    def _execute_draw_destinations(self, player: Player, info: Dict) -> Tuple[bool, Dict]:
        """Draw destination cards."""
        cards = self.destination_deck.draw(3)
        self.pending_destinations = cards
        info['destinations_drawn'] = cards
        return False, info  # Need to choose which to keep

    def _execute_keep_destinations(self, player: Player, action: Action, info: Dict) -> Tuple[bool, Dict]:
        """Keep chosen destination cards."""
        kept = []
        returned = []

        for card in self.pending_destinations:
            if card.card_id in action.destination_choices:
                player.add_destination(card)
                kept.append(card)
            else:
                returned.append(card)

        self.destination_deck.return_cards(returned)
        self.pending_destinations = []

        info['destinations_kept'] = kept
        info['destinations_returned'] = len(returned)

        return True, info

    def end_turn(self):
        """End the current turn and advance to next player."""
        self.history.append({
            'turn': self.turn_number,
            'player': self.current_player,
            'phase': self.phase,
        })

        # Reset turn state
        self.pending_draw_count = 0
        self.drew_wild_face_up = False

        # Check for game over
        if self.phase == GamePhase.LAST_ROUND:
            next_player = (self.current_player + 1) % self.config.num_players
            if next_player == self.last_round_trigger_player:
                self.phase = GamePhase.GAME_OVER
                return

        # Advance to next player
        self.current_player = (self.current_player + 1) % self.config.num_players
        if self.current_player == 0:
            self.turn_number += 1

    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.phase == GamePhase.GAME_OVER

    def get_final_scores(self) -> List[Tuple[int, int, Dict]]:
        """
        Calculate final scores for all players.
        Returns list of (player_idx, final_score, breakdown).
        """
        # First, calculate longest route for each player
        longest_routes = [(i, self.board.longest_continuous_path(i))
                         for i in range(self.config.num_players)]
        max_length = max(lr[1] for lr in longest_routes)
        longest_route_winners = [i for i, length in longest_routes if length == max_length]

        results = []
        for player in self.players:
            completed_pts, incomplete_penalty = player.check_destinations(self.board)
            has_longest = player.player_idx in longest_route_winners and max_length > 0

            breakdown = {
                'route_points': player.score,
                'completed_destinations': completed_pts,
                'incomplete_penalty': incomplete_penalty,
                'longest_route_bonus': 10 if has_longest else 0,
                'longest_route_length': self.board.longest_continuous_path(player.player_idx),
            }

            final_score = (player.score + completed_pts - incomplete_penalty +
                          (10 if has_longest else 0))

            results.append((player.player_idx, final_score, breakdown))

        return results

    def get_winner(self) -> Tuple[int, int]:
        """Get the winning player and their score."""
        scores = self.get_final_scores()
        scores.sort(key=lambda x: (-x[1], -x[2]['completed_destinations']))  # Tiebreaker
        return scores[0][0], scores[0][1]

    def get_observation(self, player_idx: int) -> Dict[str, np.ndarray]:
        """
        Get the observation for a specific player.
        This is what the AI agent sees.
        """
        player = self.players[player_idx]

        # Own hand
        own_hand = player.hand.to_numpy().astype(np.float32)

        # Own destinations (one-hot encoding of which we have)
        own_destinations = np.zeros(len(self.destination_deck.cards), dtype=np.float32)
        for dest in player.destinations:
            own_destinations[dest.card_id] = 1

        # Completed destinations
        completed_destinations = np.zeros(len(self.destination_deck.cards), dtype=np.float32)
        for card_id in player.completed_destinations:
            completed_destinations[card_id] = 1

        # Board state - route ownership
        route_state = np.zeros((len(self.board.routes), self.config.num_players + 1), dtype=np.float32)
        for i, route in enumerate(self.board.routes):
            if route.claimed_by is None:
                route_state[i, 0] = 1  # Unclaimed
            else:
                route_state[i, route.claimed_by + 1] = 1

        # Face-up cards
        face_up = np.zeros(10, dtype=np.float32)  # Count per color
        for card in self.resource_deck.face_up:
            face_up[int(card)] += 1

        # Opponent info (public)
        opponent_info = []
        for i, p in enumerate(self.players):
            if i != player_idx:
                obs = PlayerObservation(p, is_self=False)
                opponent_info.append(obs.to_numpy())
        opponent_info = np.stack(opponent_info) if opponent_info else np.zeros((0, 5), dtype=np.float32)

        # Game state
        game_info = np.array([
            self.turn_number,
            self.current_player,
            int(self.phase),
            player.trains_remaining,
            player.score,
            len(player.destinations),
            len(player.completed_destinations),
            self.resource_deck.cards_remaining(),
            self.destination_deck.cards_remaining(),
        ], dtype=np.float32)

        return {
            'own_hand': own_hand,
            'own_destinations': own_destinations,
            'completed_destinations': completed_destinations,
            'route_state': route_state.flatten(),
            'face_up_cards': face_up,
            'opponent_info': opponent_info.flatten(),
            'game_info': game_info,
        }

    def get_flat_observation(self, player_idx: int) -> np.ndarray:
        """Get flattened observation vector."""
        obs = self.get_observation(player_idx)
        return np.concatenate([v.flatten() for v in obs.values()])

    def clone(self) -> 'GameState':
        """Create a deep copy of the game state."""
        import copy
        return copy.deepcopy(self)

    def __repr__(self):
        return (f"GameState(turn={self.turn_number}, phase={self.phase.name}, "
                f"current_player={self.current_player})")


def play_random_game(num_players: int = 4, verbose: bool = False) -> Tuple[int, List[int]]:
    """Play a game with random actions. Returns (winner, scores)."""
    config = GameConfig(num_players=num_players)
    game = GameState(config)
    game.setup_game()

    # Setup phase - deal and select destinations
    game.phase = GamePhase.SETUP
    for i in range(num_players):
        cards = game.deal_initial_destinations(i)
        # Keep all destinations for random play
        keep_ids = [c.card_id for c in cards]
        game.keep_initial_destinations(i, keep_ids)

    # Main game loop
    while not game.is_game_over():
        actions = game.get_valid_actions()
        if not actions:
            game.end_turn()
            continue

        action = random.choice(actions)
        turn_complete, info = game.execute_action(action)

        if verbose and turn_complete:
            print(f"Turn {game.turn_number}, Player {game.current_player}: {action.action_type.name}")

        if turn_complete:
            game.end_turn()

    # Get results
    scores = game.get_final_scores()
    winner, winning_score = game.get_winner()

    if verbose:
        print(f"\nGame over! Winner: Player {winner} with {winning_score} points")
        for pid, score, breakdown in scores:
            print(f"  Player {pid}: {score} (routes={breakdown['route_points']}, "
                  f"dest={breakdown['completed_destinations']}, "
                  f"penalty=-{breakdown['incomplete_penalty']})")

    return winner, [s[1] for s in sorted(scores, key=lambda x: x[0])]


if __name__ == "__main__":
    # Test game
    winner, scores = play_random_game(verbose=True)
