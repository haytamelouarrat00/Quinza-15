import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
import random
from typing import List, Dict, Optional


class CardGameNet(nn.Module):
    def __init__(self, config):
        super(CardGameNet, self).__init__()

        # Get network dimensions from config
        self.state_size = config.dqn.state_size
        self.action_size = config.dqn.action_size
        self.hidden_size_1 = config.dqn.hidden_size_1
        self.hidden_size_2 = config.dqn.hidden_size_2

        # Feature layers
        self.features = nn.Sequential(
            nn.Linear(self.state_size, self.hidden_size_1),
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.Linear(self.hidden_size_1, self.hidden_size_2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Value stream - estimates state value
        self.value_stream = nn.Sequential(
            nn.Linear(self.hidden_size_2, self.hidden_size_2 // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size_2 // 2, 1)
        )

        # Advantage stream - estimates advantages of each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.hidden_size_2, self.hidden_size_2 // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size_2 // 2, self.action_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)

        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Combine value and advantages using dueling architecture
        qvalues = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvalues


class DQNAgent:
    def __init__(self, config):
        """Initialize DQN Agent with configuration."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize networks
        self.policy_net = CardGameNet(config).to(self.device)
        self.target_net = CardGameNet(config).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Training parameters
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=config.dqn.learning_rate,
            weight_decay=1e-5  # L2 regularization
        )

        # Experience replay buffer
        self.memory = deque(maxlen=config.dqn.buffer_size)
        self.Experience = namedtuple('Experience',
                                     ['state', 'action', 'reward', 'next_state', 'done'])

        # Training state
        self.epsilon = config.dqn.epsilon_start
        self.training_step = 0

    def encode_state(self, game_state: Dict) -> np.ndarray:
        """Convert game state dictionary into a flat vector."""
        try:
            # Encode player's hand
            hand = np.zeros(40)
            for card in game_state['player_hands'][game_state['current_player']]:
                suit_idx = ['Oros', 'Copas', 'Espadas', 'Bastos'].index(card[0])
                value_idx = card[1] - 3 if card[1] > 7 else card[1] - 1
                card_idx = suit_idx * 10 + value_idx
                if 0 <= card_idx < 40:
                    hand[card_idx] = 1

            # Encode table cards
            table = np.zeros(40)
            for card in game_state['table_cards']:
                suit_idx = ['Oros', 'Copas', 'Espadas', 'Bastos'].index(card[0])
                value_idx = card[1] - 3 if card[1] > 7 else card[1] - 1
                card_idx = suit_idx * 10 + value_idx
                if 0 <= card_idx < 40:
                    table[card_idx] = 1

            # Additional state information
            other_info = np.array([
                game_state['cards_in_deck'] / 40,
                game_state['current_player'] / (len(game_state['player_hands']) - 1),
                len(game_state['collected_cards'][game_state['current_player']]) / 40,
            ])

            return np.concatenate([hand, table, other_info])

        except Exception as e:
            raise ValueError(f"Error encoding game state: {str(e)}")

    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool) -> None:
        """Store experience in replay memory."""
        self.memory.append(self.Experience(state, action, reward, next_state, done))

    def act(self, state: np.ndarray, legal_actions: List[int]) -> int:
        """Choose an action using epsilon-greedy policy."""
        if not legal_actions:
            raise ValueError("No legal actions available")

        if random.random() < self.epsilon:
            return random.choice(legal_actions)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_values = self.policy_net(state_tensor)

        # Mask illegal actions
        mask = torch.ones_like(action_values) * float('-inf')
        mask[0, legal_actions] = 0
        action_values += mask

        return action_values.argmax(dim=1).item()

    def replay(self) -> Optional[float]:
        """Train on a batch of experiences using Double DQN."""
        if len(self.memory) < self.config.dqn.batch_size:
            return None

        # Sample batch
        batch = random.sample(self.memory, self.config.dqn.batch_size)

        # Prepare batch tensors
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.FloatTensor([e.done for e in batch]).to(self.device)

        # Double DQN update
        with torch.no_grad():
            # Select actions using policy network
            next_actions = self.policy_net(next_states).argmax(dim=1)
            # Evaluate actions using target network
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.config.dqn.gamma * next_q_values

        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Compute loss and update
        loss = F.smooth_l1_loss(current_q_values, target_q_values)  # Huber loss for stability

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(
            self.config.dqn.epsilon_min,
            self.epsilon * self.config.dqn.epsilon_decay
        )

        self.training_step += 1

        return loss.item()

    def update_target_network(self) -> None:
        """Update target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str) -> None:
        """Save agent state."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, path)

    def load(self, path: str) -> None:
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
