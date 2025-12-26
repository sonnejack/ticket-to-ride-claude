"""
Self-play game simulation for training.
"""
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import random

from ticket_to_ride_rl.game import (
    GameState, GameConfig, Action, ActionType, GamePhase
)
from ticket_to_ride_rl.agents import (
    PPOAgent, ArchetypeType, get_archetype_config
)
from .metrics import MetricsCollector, GameLogger, compute_metrics_from_game, GameMetrics


@dataclass
class PlayerSetup:
    """Configuration for a player in self-play."""
    agent: PPOAgent
    archetype: ArchetypeType
    use_tracking: bool


class SelfPlayGame:
    """
    Runs a single game with multiple agents.
    """

    def __init__(self, players: List[PlayerSetup], config: GameConfig = None,
                 record_log: bool = False):
        self.players = players
        self.config = config or GameConfig(num_players=len(players))
        self.game = GameState(self.config)
        self.record_log = record_log
        self.logger = None

        # Track hand sizes over time for metrics
        self.hand_histories: List[List[int]] = [[] for _ in range(len(players))]

        # Rewards accumulated per player this game
        self.episode_rewards: List[float] = [0.0] * len(players)

    def setup(self):
        """Initialize the game and deal initial cards."""
        self.game.setup_game()

        # Initialize trackers for each agent
        for i, player_setup in enumerate(self.players):
            player_setup.agent.init_tracker(len(self.players), i)
            player_setup.agent.reset_episode()

        if self.record_log:
            self.logger = GameLogger(game_id=0)

    def handle_initial_destinations(self):
        """Handle initial destination card selection for all players."""
        for i, player_setup in enumerate(self.players):
            archetype_config = get_archetype_config(player_setup.archetype)

            # Draw initial destinations
            cards = self.game.deal_initial_destinations(i)

            # Use archetype preferences to select
            # Ensure we keep at least the game minimum (2) but not more than available
            game_min = self.game.config.min_destinations_keep
            n_keep = min(max(archetype_config.min_destinations_keep, game_min), len(cards))

            policy = player_setup.agent.archetype_policy
            if policy and cards:
                scores = policy.get_destination_keep_preferences(
                    cards, self.game, i
                )
                sorted_indices = np.argsort(scores)[::-1]
                keep_ids = [cards[idx].card_id for idx in sorted_indices[:n_keep]]
            else:
                # Keep minimum required
                keep_ids = [c.card_id for c in cards[:n_keep]]

            if keep_ids:
                self.game.keep_initial_destinations(i, keep_ids)

    def play_game(self) -> Tuple[int, List[int], Dict]:
        """
        Play a complete game.

        Returns:
            winner: Index of winning player
            scores: Final scores for all players
            info: Additional game information
        """
        self.setup()
        self.handle_initial_destinations()

        # Start main game
        self.game.phase = GamePhase.PLAYING

        max_turns = 500  # Safety limit
        turn_count = 0

        while not self.game.is_game_over() and turn_count < max_turns:
            current_player = self.game.current_player
            player_setup = self.players[current_player]
            agent = player_setup.agent

            # Record hand size at start of turn
            self.hand_histories[current_player].append(
                self.game.players[current_player].hand.total()
            )

            # Get valid actions
            valid_actions = self.game.get_valid_actions()

            if not valid_actions:
                self.game.end_turn()
                turn_count += 1
                continue

            # Get observation
            obs = agent.get_observation(self.game, current_player)

            # Create action mask
            action_mask = np.zeros(agent.config.max_actions, dtype=bool)
            action_mask[:len(valid_actions)] = True

            # Select action
            action_idx, action, log_prob, value = agent.select_action(
                self.game, current_player, valid_actions
            )

            # Execute action
            turn_complete, info = self.game.execute_action(action)

            # Update other agents' trackers
            self._update_trackers(current_player, action, info)

            # Log action if recording
            if self.logger:
                self.logger.log_action(
                    current_player,
                    action.action_type.name,
                    {'info': str(info)}
                )

            # Calculate intermediate reward
            reward = self._calculate_reward(current_player, action, info)
            self.episode_rewards[current_player] += reward

            # Store transition
            tracking_obs = None
            if player_setup.use_tracking and agent.hand_tracker:
                tracking_obs = agent.hand_tracker.to_observation()

            agent.store_transition(
                obs=obs,
                action_idx=action_idx,
                action_mask=action_mask,
                log_prob=log_prob,
                value=value,
                reward=reward,
                done=False,
                tracking_obs=tracking_obs
            )

            if turn_complete:
                if self.logger:
                    hand_sizes = [p.hand.total() for p in self.game.players]
                    self.logger.log_hand_sizes(hand_sizes)
                    self.logger.end_turn()

                self.game.end_turn()
                turn_count += 1

        # Game ended - calculate final rewards
        winner, winning_score = self.game.get_winner()
        final_scores = self.game.get_final_scores()

        # Add final reward to last transition for each player
        for i, player_setup in enumerate(self.players):
            agent = player_setup.agent
            final_reward = self._calculate_final_reward(i, winner, final_scores)
            self.episode_rewards[i] += final_reward

            # Update last transition with done=True and final reward
            if agent.buffer.transitions:
                agent.buffer.transitions[-1].done = True
                agent.buffer.transitions[-1].reward += final_reward

        # Prepare return info
        game_info = {
            'turn_count': turn_count,
            'final_scores': final_scores,
            'archetypes': [p.archetype.value for p in self.players],
            'tracking': [p.use_tracking for p in self.players],
        }

        if self.logger:
            game_info['log'] = self.logger.to_dict()

        scores = [0] * len(self.players)
        for idx, score, _ in final_scores:
            scores[idx] = score

        return winner, scores, game_info

    def _update_trackers(self, acting_player: int, action: Action, info: Dict):
        """Update hand trackers for all players observing this action."""
        for i, player_setup in enumerate(self.players):
            if i == acting_player:
                continue

            agent = player_setup.agent
            if not agent.hand_tracker:
                continue

            if action.action_type == ActionType.DRAW_DECK:
                # Blind draw - don't know what card
                for _ in range(len(info.get('cards_drawn', []))):
                    agent.update_tracking('blind_draw', player_idx=acting_player)

            elif action.action_type == ActionType.DRAW_FACE_UP:
                card = info.get('card_drawn')
                if card is not None:
                    agent.update_tracking(
                        'face_up_draw',
                        player_idx=acting_player,
                        card=card
                    )

            elif action.action_type == ActionType.CLAIM_ROUTE:
                route = info.get('route')
                if route:
                    agent.update_tracking(
                        'route_claim',
                        player_idx=acting_player,
                        route_length=route.length,
                        route_color=route.color,
                        cards_used=info.get('cards_used')
                    )

    def _calculate_reward(self, player_idx: int, action: Action, info: Dict) -> float:
        """Calculate intermediate reward for an action."""
        reward = 0.0

        if action.action_type == ActionType.CLAIM_ROUTE:
            # Reward for claiming routes
            points = info.get('points', 0)
            reward += points * 0.1  # Scale down

            # Check if this helps complete a destination
            player = self.game.players[player_idx]
            completed_before = len(player.completed_destinations)
            player.check_destinations(self.game.board)
            completed_after = len(player.completed_destinations)

            if completed_after > completed_before:
                reward += 2.0  # Bonus for completing destination

        elif action.action_type == ActionType.DRAW_DESTINATIONS:
            # Small reward/penalty for drawing destinations mid-game
            if self.game.turn_number > 10:
                reward -= 0.1  # Slight penalty for late destination draws

        return reward

    def _calculate_final_reward(self, player_idx: int, winner: int,
                                  final_scores: List[Tuple[int, int, Dict]]) -> float:
        """Calculate final reward based on game outcome."""
        # Get player's rank
        sorted_scores = sorted(final_scores, key=lambda x: -x[1])
        rank = next(i for i, (idx, _, _) in enumerate(sorted_scores) if idx == player_idx)

        # Get player's score
        player_score = next(score for idx, score, _ in final_scores if idx == player_idx)

        # Win/lose reward
        if player_idx == winner:
            reward = 10.0
        else:
            reward = -5.0

        # Add score-based component
        reward += player_score * 0.01

        # Rank bonus
        reward += (len(self.players) - 1 - rank) * 2.0

        return reward


class SelfPlayManager:
    """
    Manages multiple self-play games for training.
    """

    def __init__(self, agents_by_archetype: Dict[ArchetypeType, Tuple[PPOAgent, PPOAgent]],
                 num_players: int = 4, metrics: MetricsCollector = None):
        """
        Args:
            agents_by_archetype: Dict mapping archetype to (blind_agent, tracking_agent)
            num_players: Number of players per game
            metrics: Metrics collector
        """
        self.agents_by_archetype = agents_by_archetype
        self.num_players = num_players
        self.metrics = metrics or MetricsCollector()
        self.game_counter = 0

    def create_player_setups(self) -> List[PlayerSetup]:
        """Create a diverse set of players for a game."""
        setups = []
        archetypes = list(self.agents_by_archetype.keys())

        for i in range(self.num_players):
            # Cycle through archetypes
            archetype = archetypes[i % len(archetypes)]

            # Alternate between blind and tracking
            use_tracking = (i % 2 == 0)

            if use_tracking:
                agent = self.agents_by_archetype[archetype][1]  # Tracking agent
            else:
                agent = self.agents_by_archetype[archetype][0]  # Blind agent

            setups.append(PlayerSetup(
                agent=agent,
                archetype=archetype,
                use_tracking=use_tracking
            ))

        # Shuffle to randomize positions
        random.shuffle(setups)
        return setups

    def play_game(self, record_log: bool = False) -> GameMetrics:
        """Play a single game and collect metrics."""
        players = self.create_player_setups()
        config = GameConfig(num_players=self.num_players)

        game = SelfPlayGame(players, config, record_log=record_log)
        winner, scores, info = game.play_game()

        # Create metrics
        metrics = compute_metrics_from_game(
            game.game,
            player_archetypes=[p.archetype.value for p in players],
            player_tracking=[p.use_tracking for p in players],
            hand_histories=game.hand_histories,
            game_id=self.game_counter
        )

        self.metrics.record_game(metrics)

        if record_log and 'log' in info:
            self.metrics.record_sample_game(info['log'])

        self.game_counter += 1
        return metrics

    def play_games(self, num_games: int, record_samples: int = 0) -> List[GameMetrics]:
        """Play multiple games."""
        all_metrics = []

        for i in range(num_games):
            record = i < record_samples
            metrics = self.play_game(record_log=record)
            all_metrics.append(metrics)

        return all_metrics

    def update_agents(self) -> Dict[str, Dict[str, float]]:
        """Update all agents using collected experience."""
        update_stats = {}

        for archetype, (blind_agent, tracking_agent) in self.agents_by_archetype.items():
            # Update blind agent
            stats = blind_agent.update()
            if stats:
                update_stats[f"{archetype.value}_blind"] = stats

            # Update tracking agent
            stats = tracking_agent.update()
            if stats:
                update_stats[f"{archetype.value}_tracking"] = stats

        return update_stats


def run_evaluation_game(agents: List[PPOAgent], archetypes: List[ArchetypeType],
                        use_tracking: List[bool]) -> Tuple[int, List[int]]:
    """
    Run a single evaluation game with deterministic action selection.
    """
    players = []
    for agent, archetype, tracking in zip(agents, archetypes, use_tracking):
        players.append(PlayerSetup(
            agent=agent,
            archetype=archetype,
            use_tracking=tracking
        ))

    config = GameConfig(num_players=len(players))
    game = SelfPlayGame(players, config, record_log=False)

    # Temporarily set agents to deterministic mode
    winner, scores, _ = game.play_game()

    return winner, scores
