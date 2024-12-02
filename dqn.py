import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random


class CardGameNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size_1, hidden_size_2):
        super(CardGameNet, self).__init__()

        # Neural network for Q-value prediction with configurable layer sizes
        self.fc1 = nn.Linear(state_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size_1=128, hidden_size_2=64):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2

        # Hyperparameters with default values (will be updated by Optuna)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64

        # Experience replay buffer
        self.memory = deque(maxlen=10000)
        self.Experience = namedtuple('Experience',
                                     ['state', 'action', 'reward', 'next_state', 'done'])

        # Initialize networks with specified architecture
        self.policy_net = CardGameNet(state_size, action_size, hidden_size_1, hidden_size_2)
        self.target_net = CardGameNet(state_size, action_size, hidden_size_1, hidden_size_2)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def encode_state(self, game_state):
        """Convert game state dictionary into a flat vector."""
        # Encode player's hand
        hand = np.zeros(40)  # One-hot encoding for each possible card
        for card in game_state['player_hands'][game_state['current_player']]:
            suit_idx = ['Oros', 'Copas', 'Espadas', 'Bastos'].index(card[0])
            # Adjust value to be 0-9 for each suit (10,11,12 map to 7,8,9)
            if card[1] <= 7:
                value_idx = card[1] - 1
            else:  # For 10, 11, 12
                value_idx = card[1] - 3
            card_idx = suit_idx * 10 + value_idx
            if 0 <= card_idx < 40:  # Safety check
                hand[card_idx] = 1

        # Encode table cards
        table = np.zeros(40)
        for card in game_state['table_cards']:
            suit_idx = ['Oros', 'Copas', 'Espadas', 'Bastos'].index(card[0])
            if card[1] <= 7:
                value_idx = card[1] - 1
            else:  # For 10, 11, 12
                value_idx = card[1] - 3
            card_idx = suit_idx * 10 + value_idx
            if 0 <= card_idx < 40:  # Safety check
                table[card_idx] = 1

        # Additional state information
        other_info = np.array([
            game_state['cards_in_deck'] / 40,  # Normalize deck size
            game_state['current_player'] / (len(game_state['player_hands']) - 1),  # Normalize player position
            len(game_state['collected_cards'][game_state['current_player']]) / 40,  # Normalize collected cards
        ])

        return np.concatenate([hand, table, other_info])

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append(self.Experience(state, action, reward, next_state, done))

    def act(self, state, legal_actions):
        """Choose an action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.choice(legal_actions)

        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.policy_net(state)

        # Mask illegal actions with large negative values
        mask = torch.ones(self.action_size) * float('-inf')
        mask[legal_actions] = 0
        action_values += mask

        return action_values.argmax().item()

    def replay(self):
        """Train on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return

        experiences = random.sample(self.memory, self.batch_size)

        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.FloatTensor([e.done for e in experiences])

        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute loss and update
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        """Update target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
