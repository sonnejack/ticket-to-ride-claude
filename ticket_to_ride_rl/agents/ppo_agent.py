"""
PPO (Proximal Policy Optimization) agent implementation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from collections import deque

from .networks import ActorCritic, create_network
from .archetypes import ArchetypeType, ArchetypePolicy, get_archetype_config
from .hand_tracker import HandTracker, SimpleHandTracker
from ticket_to_ride_rl.game import GameState, Action, ActionType


@dataclass
class PPOConfig:
    """PPO hyperparameters."""
    # Learning
    learning_rate: float = 3e-4
    lr_decay: float = 0.9999
    min_lr: float = 1e-5

    # PPO specific
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    # GAE
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Training
    batch_size: int = 64
    num_epochs: int = 4
    num_minibatches: int = 4

    # Network
    hidden_dim: int = 256
    use_lstm: bool = False
    lstm_dim: int = 128

    # Action space
    max_actions: int = 600  # Maximum possible actions in any state (route claiming can have many combinations)


@dataclass
class Transition:
    """A single experience tuple."""
    observation: np.ndarray
    action: int
    action_mask: np.ndarray
    log_prob: float
    value: float
    reward: float
    done: bool
    tracking_obs: Optional[np.ndarray] = None


class RolloutBuffer:
    """Buffer for storing rollout experiences."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.transitions: List[Transition] = []
        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None

    def add(self, transition: Transition):
        """Add a transition to the buffer."""
        self.transitions.append(transition)
        if len(self.transitions) > self.max_size:
            self.transitions.pop(0)

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float):
        """Compute Generalized Advantage Estimation."""
        n = len(self.transitions)
        self.advantages = np.zeros(n, dtype=np.float32)
        self.returns = np.zeros(n, dtype=np.float32)

        last_gae = 0
        for t in reversed(range(n)):
            trans = self.transitions[t]

            if t == n - 1:
                next_value = last_value
                next_done = 1.0
            else:
                next_trans = self.transitions[t + 1]
                next_value = next_trans.value
                next_done = float(next_trans.done)

            delta = trans.reward + gamma * next_value * (1 - next_done) - trans.value
            last_gae = delta + gamma * gae_lambda * (1 - next_done) * last_gae
            self.advantages[t] = last_gae
            self.returns[t] = last_gae + trans.value

    def get_batches(self, batch_size: int, num_minibatches: int = 1):
        """Generate minibatches for training."""
        n = len(self.transitions)
        indices = np.random.permutation(n)

        minibatch_size = n // num_minibatches
        for start in range(0, n, minibatch_size):
            end = min(start + minibatch_size, n)
            batch_indices = indices[start:end]

            yield self._get_batch(batch_indices)

    def _get_batch(self, indices):
        """Extract a batch of data."""
        observations = np.array([self.transitions[i].observation for i in indices])
        actions = np.array([self.transitions[i].action for i in indices])
        action_masks = np.array([self.transitions[i].action_mask for i in indices])
        old_log_probs = np.array([self.transitions[i].log_prob for i in indices])
        advantages = self.advantages[indices]
        returns = self.returns[indices]

        # Include tracking observations if available
        tracking_obs = None
        if self.transitions[0].tracking_obs is not None:
            tracking_obs = np.array([self.transitions[i].tracking_obs for i in indices])

        return {
            'observations': observations,
            'actions': actions,
            'action_masks': action_masks,
            'old_log_probs': old_log_probs,
            'advantages': advantages,
            'returns': returns,
            'tracking_obs': tracking_obs,
        }

    def clear(self):
        """Clear the buffer."""
        self.transitions = []
        self.advantages = None
        self.returns = None

    def __len__(self):
        return len(self.transitions)


class PPOAgent:
    """
    PPO agent for playing Ticket to Ride.
    """

    def __init__(self, obs_dim: int, config: PPOConfig = None,
                 archetype: ArchetypeType = None, use_tracking: bool = False,
                 device: str = 'cpu', num_players: int = 4):
        self.config = config or PPOConfig()
        self.archetype = archetype
        self.use_tracking = use_tracking
        self.device = torch.device(device)
        self.num_players = num_players

        # Observation dimension
        self.obs_dim = obs_dim
        if use_tracking:
            # Add tracking observation dimensions
            # (num_players - 1) opponents * (10 card colors + 1 count) = extra dims
            tracking_dims = (num_players - 1) * 11
            self.obs_dim += tracking_dims

        # Create network
        archetype_name = archetype.value if archetype else None
        self.network = create_network(
            self.obs_dim,
            self.config.max_actions,
            archetype_name,
            self.config.hidden_dim,
            self.config.use_lstm,
            self.config.lstm_dim
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate
        )

        # Archetype policy for biases
        if archetype:
            self.archetype_policy = ArchetypePolicy(archetype)
            self.archetype_config = get_archetype_config(archetype)
        else:
            self.archetype_policy = None
            self.archetype_config = None

        # Experience buffer
        self.buffer = RolloutBuffer()

        # Hand tracker
        self.hand_tracker = None

        # LSTM hidden state
        self.hidden = None

        # Training stats
        self.total_updates = 0
        self.current_lr = self.config.learning_rate

    def init_tracker(self, num_players: int, player_idx: int):
        """Initialize hand tracker for a new game."""
        if self.use_tracking:
            self.hand_tracker = HandTracker(num_players, player_idx)
        else:
            self.hand_tracker = None

    def reset_episode(self):
        """Reset for a new episode/game."""
        self.hidden = None
        if self.hand_tracker:
            self.hand_tracker.reset()

    def get_observation(self, game_state: GameState, player_idx: int) -> np.ndarray:
        """Get observation including tracking info if enabled."""
        base_obs = game_state.get_flat_observation(player_idx)

        if self.use_tracking and self.hand_tracker:
            tracking_obs = self.hand_tracker.to_observation()
            return np.concatenate([base_obs, tracking_obs])

        return base_obs

    def select_action(self, game_state: GameState, player_idx: int,
                      valid_actions: List[Action], deterministic: bool = False
                      ) -> Tuple[int, Action, float, float]:
        """
        Select an action given the game state.

        Returns:
            action_idx: Index into valid_actions
            action: The selected Action object
            log_prob: Log probability of the action
            value: State value estimate
        """
        # Get observation
        obs = self.get_observation(game_state, player_idx)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        # Create action mask
        action_mask = torch.zeros(self.config.max_actions, dtype=torch.bool)
        action_mask[:len(valid_actions)] = True
        action_mask = action_mask.unsqueeze(0).to(self.device)

        # Get action from network
        with torch.no_grad():
            probs, value, self.hidden = self.network(obs_tensor, action_mask, self.hidden)

            # Apply archetype biases if available
            if self.archetype_policy and not deterministic:
                biases = self.archetype_policy.get_action_biases(
                    valid_actions, game_state, player_idx
                )
                # Scale biases based on training progress (decay over time)
                bias_scale = max(0.1, 1.0 - self.total_updates / 10000)
                biases = torch.FloatTensor(biases).to(self.device) * bias_scale

                # Pad biases to max_actions (truncate if more actions than expected)
                full_biases = torch.zeros(self.config.max_actions).to(self.device)
                num_biases = min(len(biases), self.config.max_actions)
                full_biases[:num_biases] = biases[:num_biases]

                # Apply to logits (before softmax)
                logits = torch.log(probs + 1e-10) + full_biases.unsqueeze(0)
                logits = logits.masked_fill(~action_mask, float('-inf'))
                probs = torch.softmax(logits, dim=-1)

            # Sample action
            dist = torch.distributions.Categorical(probs)
            if deterministic:
                action_idx = probs.argmax(dim=-1).item()
            else:
                action_idx = dist.sample().item()

            log_prob = dist.log_prob(torch.tensor(action_idx).to(self.device)).item()
            value = value.squeeze().item()

        return action_idx, valid_actions[action_idx], log_prob, value

    def store_transition(self, obs: np.ndarray, action_idx: int,
                         action_mask: np.ndarray, log_prob: float,
                         value: float, reward: float, done: bool,
                         tracking_obs: Optional[np.ndarray] = None):
        """Store a transition in the buffer."""
        transition = Transition(
            observation=obs,
            action=action_idx,
            action_mask=action_mask,
            log_prob=log_prob,
            value=value,
            reward=reward,
            done=done,
            tracking_obs=tracking_obs
        )
        self.buffer.add(transition)

    def update(self) -> Dict[str, float]:
        """
        Perform PPO update using collected experiences.
        Returns training statistics.
        """
        if len(self.buffer) < self.config.batch_size:
            return {}

        # Compute advantages
        last_obs = self.buffer.transitions[-1].observation
        last_obs_tensor = torch.FloatTensor(last_obs).unsqueeze(0).to(self.device)
        action_mask = torch.ones(1, self.config.max_actions, dtype=torch.bool).to(self.device)

        with torch.no_grad():
            _, last_value, _ = self.network(last_obs_tensor, action_mask, None)
            last_value = last_value.item()

        self.buffer.compute_gae(last_value, self.config.gamma, self.config.gae_lambda)

        # Training loop
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0

        for epoch in range(self.config.num_epochs):
            for batch in self.buffer.get_batches(
                self.config.batch_size,
                self.config.num_minibatches
            ):
                policy_loss, value_loss, entropy = self._update_batch(batch)
                total_policy_loss += policy_loss
                total_value_loss += value_loss
                total_entropy += entropy
                num_updates += 1

        self.buffer.clear()
        self.total_updates += 1

        # Learning rate decay
        self.current_lr = max(
            self.config.min_lr,
            self.config.learning_rate * (self.config.lr_decay ** self.total_updates)
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr

        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'lr': self.current_lr,
        }

    def _update_batch(self, batch: Dict) -> Tuple[float, float, float]:
        """Update on a single batch."""
        # Convert to tensors
        obs = torch.FloatTensor(batch['observations']).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        action_masks = torch.BoolTensor(batch['action_masks']).to(self.device)
        old_log_probs = torch.FloatTensor(batch['old_log_probs']).to(self.device)
        advantages = torch.FloatTensor(batch['advantages']).to(self.device)
        returns = torch.FloatTensor(batch['returns']).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Forward pass
        log_probs, values, entropy = self.network.evaluate_actions(
            obs, action_masks, actions
        )

        # Policy loss (PPO clip objective)
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon,
                            1 + self.config.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = nn.functional.mse_loss(values, returns)

        # Entropy bonus
        entropy_loss = -entropy.mean()

        # Total loss
        loss = (policy_loss +
                self.config.value_loss_coef * value_loss +
                self.config.entropy_coef * entropy_loss)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        return policy_loss.item(), value_loss.item(), entropy.mean().item()

    def update_tracking(self, event_type: str, **kwargs):
        """Update hand tracker with an observed event."""
        if not self.hand_tracker:
            return

        if event_type == 'blind_draw':
            self.hand_tracker.observe_blind_draw(kwargs['player_idx'])
        elif event_type == 'face_up_draw':
            self.hand_tracker.observe_face_up_draw(
                kwargs['player_idx'], kwargs['card']
            )
        elif event_type == 'route_claim':
            self.hand_tracker.observe_route_claim(
                kwargs['player_idx'],
                kwargs['route_length'],
                kwargs['route_color'],
                kwargs.get('cards_used')
            )

    def save(self, path: str):
        """Save agent to file."""
        torch.save({
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config,
            'archetype': self.archetype,
            'use_tracking': self.use_tracking,
            'total_updates': self.total_updates,
            'current_lr': self.current_lr,
        }, path)

    def load(self, path: str):
        """Load agent from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.total_updates = checkpoint.get('total_updates', 0)
        self.current_lr = checkpoint.get('current_lr', self.config.learning_rate)


class MultiAgentPPO:
    """
    Manages multiple PPO agents for multi-agent training.
    """

    def __init__(self, obs_dim: int, num_agents: int, config: PPOConfig = None,
                 archetypes: List[ArchetypeType] = None,
                 use_tracking: List[bool] = None, device: str = 'cpu'):
        self.num_agents = num_agents
        self.agents: List[PPOAgent] = []

        for i in range(num_agents):
            archetype = archetypes[i] if archetypes else None
            tracking = use_tracking[i] if use_tracking else False

            agent = PPOAgent(
                obs_dim=obs_dim,
                config=config,
                archetype=archetype,
                use_tracking=tracking,
                device=device
            )
            self.agents.append(agent)

    def get_agent(self, idx: int) -> PPOAgent:
        """Get agent by index."""
        return self.agents[idx]

    def save_all(self, base_path: str):
        """Save all agents."""
        for i, agent in enumerate(self.agents):
            path = f"{base_path}_agent_{i}.pt"
            agent.save(path)

    def load_all(self, base_path: str):
        """Load all agents."""
        for i, agent in enumerate(self.agents):
            path = f"{base_path}_agent_{i}.pt"
            agent.load(path)
