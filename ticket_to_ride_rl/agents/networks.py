"""
Neural network architectures for PPO agents.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class FeatureExtractor(nn.Module):
    """
    Shared feature extraction layers for policy and value networks.
    Processes game observations into learned representations.
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        # Layer normalization for stability
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        return x


class LSTMFeatureExtractor(nn.Module):
    """
    Feature extractor with LSTM for temporal patterns and opponent modeling.
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 256, lstm_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.lstm = nn.LSTM(hidden_dim, lstm_dim, batch_first=True)
        self.lstm_dim = lstm_dim

        self.fc2 = nn.Linear(lstm_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, obs_dim) or (batch, obs_dim)
            hidden: LSTM hidden state tuple (h, c)

        Returns:
            features: Extracted features
            hidden: New LSTM hidden state
        """
        # Handle non-sequence input
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        batch_size = x.size(0)

        # Initial feature extraction
        x = F.relu(self.ln1(self.fc1(x)))

        # LSTM
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)

        lstm_out, hidden = self.lstm(x, hidden)

        # Take last output
        x = lstm_out[:, -1, :]

        # Final layer
        x = F.relu(self.ln2(self.fc2(x)))

        return x, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state."""
        h0 = torch.zeros(1, batch_size, self.lstm_dim, device=device)
        c0 = torch.zeros(1, batch_size, self.lstm_dim, device=device)
        return (h0, c0)


class PolicyNetwork(nn.Module):
    """
    Policy network that outputs action probabilities.
    Handles variable action spaces with action masking.
    """

    def __init__(self, obs_dim: int, max_actions: int, hidden_dim: int = 256,
                 use_lstm: bool = False, lstm_dim: int = 128):
        super().__init__()
        self.max_actions = max_actions
        self.use_lstm = use_lstm

        if use_lstm:
            self.features = LSTMFeatureExtractor(obs_dim, hidden_dim, lstm_dim)
        else:
            self.features = FeatureExtractor(obs_dim, hidden_dim)

        self.policy_head = nn.Linear(hidden_dim, max_actions)

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            obs: Observation tensor
            action_mask: Boolean mask of valid actions (True = valid)
            hidden: LSTM hidden state (only used if use_lstm=True)

        Returns:
            action_probs: Probability distribution over actions
            hidden: New LSTM hidden state (or None)
        """
        if self.use_lstm:
            features, hidden = self.features(obs, hidden)
        else:
            features = self.features(obs)
            hidden = None

        # Raw logits
        logits = self.policy_head(features)

        # Apply action mask (set invalid actions to large negative)
        masked_logits = logits.masked_fill(~action_mask, float('-inf'))

        # Softmax to get probabilities
        action_probs = F.softmax(masked_logits, dim=-1)

        return action_probs, hidden

    def get_action(self, obs: torch.Tensor, action_mask: torch.Tensor,
                   hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                   deterministic: bool = False
                   ) -> Tuple[int, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Sample an action from the policy.

        Returns:
            action: Selected action index
            log_prob: Log probability of the action
            hidden: New LSTM hidden state
        """
        with torch.no_grad():
            probs, hidden = self.forward(obs, action_mask, hidden)

            if deterministic:
                action = probs.argmax(dim=-1).item()
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()

            log_prob = torch.log(probs[0, action] + 1e-10)

        return action, log_prob, hidden


class ValueNetwork(nn.Module):
    """
    Value network that estimates state value V(s).
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 256,
                 use_lstm: bool = False, lstm_dim: int = 128):
        super().__init__()
        self.use_lstm = use_lstm

        if use_lstm:
            self.features = LSTMFeatureExtractor(obs_dim, hidden_dim, lstm_dim)
        else:
            self.features = FeatureExtractor(obs_dim, hidden_dim)

        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            obs: Observation tensor

        Returns:
            value: Estimated state value
            hidden: New LSTM hidden state (or None)
        """
        if self.use_lstm:
            features, hidden = self.features(obs, hidden)
        else:
            features = self.features(obs)
            hidden = None

        value = self.value_head(features)
        return value, hidden


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network with shared feature extraction.
    """

    def __init__(self, obs_dim: int, max_actions: int, hidden_dim: int = 256,
                 use_lstm: bool = False, lstm_dim: int = 128):
        super().__init__()
        self.use_lstm = use_lstm
        self.max_actions = max_actions

        # Shared feature extractor
        if use_lstm:
            self.features = LSTMFeatureExtractor(obs_dim, hidden_dim, lstm_dim)
        else:
            self.features = FeatureExtractor(obs_dim, hidden_dim)

        # Separate heads
        self.policy_head = nn.Linear(hidden_dim, max_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through both policy and value heads.

        Returns:
            action_probs: Action probability distribution
            value: State value estimate
            hidden: New LSTM hidden state
        """
        if self.use_lstm:
            features, hidden = self.features(obs, hidden)
        else:
            features = self.features(obs)
            hidden = None

        # Policy
        logits = self.policy_head(features)
        masked_logits = logits.masked_fill(~action_mask, float('-inf'))
        action_probs = F.softmax(masked_logits, dim=-1)

        # Handle NaN case (when all actions are masked)
        action_probs = torch.nan_to_num(action_probs, nan=1.0 / action_mask.size(-1))

        # Value
        value = self.value_head(features)

        return action_probs, value, hidden

    def get_action_and_value(self, obs: torch.Tensor, action_mask: torch.Tensor,
                              hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                              deterministic: bool = False
                              ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """
        Get action, log probability, entropy, and value in one pass.

        Returns:
            action: Selected action
            log_prob: Log probability of action
            entropy: Policy entropy
            value: State value
            hidden: New LSTM hidden state
        """
        probs, value, hidden = self.forward(obs, action_mask, hidden)

        dist = torch.distributions.Categorical(probs)

        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action.item(), log_prob, entropy, value.squeeze(-1), hidden

    def evaluate_actions(self, obs: torch.Tensor, action_mask: torch.Tensor,
                         actions: torch.Tensor,
                         hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and values for given observations and actions.
        Used during PPO update.

        Returns:
            log_probs: Log probabilities of the actions
            values: State values
            entropy: Policy entropy
        """
        probs, values, _ = self.forward(obs, action_mask, hidden)

        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values.squeeze(-1), entropy


class ArchetypeActorCritic(ActorCritic):
    """
    Actor-Critic with archetype-specific bias initialization.
    """

    def __init__(self, obs_dim: int, max_actions: int, archetype: str,
                 hidden_dim: int = 256, use_lstm: bool = False, lstm_dim: int = 128):
        super().__init__(obs_dim, max_actions, hidden_dim, use_lstm, lstm_dim)
        self.archetype = archetype

        # Apply archetype-specific initialization
        self._initialize_archetype_bias()

    def _initialize_archetype_bias(self):
        """
        Initialize policy head with archetype-specific biases.
        These create initial behavioral preferences that learning can modify.
        """
        with torch.no_grad():
            # Reset to small values
            self.policy_head.bias.fill_(0.0)

            # Action type ranges (approximate, actual mapping depends on action encoding)
            # These biases encourage certain action types initially

            if self.archetype == "architect":
                # Prefers drawing cards and destinations
                # Will be refined during training
                pass

            elif self.archetype == "instant_gratification":
                # Prefers claiming routes immediately
                pass

            elif self.archetype == "hoarder":
                # Strong preference for drawing cards
                pass

            elif self.archetype == "blocker":
                # Neutral - will learn to block from opponent modeling
                pass

            elif self.archetype == "wildcard":
                # Slightly prefer blind draws
                pass


def create_network(obs_dim: int, max_actions: int, archetype: str = None,
                   hidden_dim: int = 256, use_lstm: bool = False, lstm_dim: int = 128
                   ) -> ActorCritic:
    """
    Factory function to create appropriate network.
    """
    if archetype:
        return ArchetypeActorCritic(
            obs_dim, max_actions, archetype, hidden_dim, use_lstm, lstm_dim
        )
    return ActorCritic(obs_dim, max_actions, hidden_dim, use_lstm, lstm_dim)
