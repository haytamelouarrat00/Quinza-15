import torch
import torch.nn as nn


class CardEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 16),  # Take both value and suit
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)


class QNetwork(nn.Module):
    def __init__(self, action_size):
        super().__init__()

        # Card encoders
        self.hand_encoder = CardEncoder()
        self.table_encoder = CardEncoder()

        # Calculate the exact combined size
        self.hand_size = 3 * 16  # 3 cards, each encoded to 16 features
        self.table_size = 40 * 16  # 40 cards, each encoded to 16 features
        self.other_features = 1 + 1 + 4  # current_player + deck_size + hand_sizes(4 players)

        self.combined_size = self.hand_size + self.table_size + self.other_features

        self.network = nn.Sequential(
            nn.Linear(self.combined_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, state):
        device = next(self.parameters()).device
        batch_size = state['hand'].shape[0] if isinstance(state['hand'], torch.Tensor) else 1

        # Process hand cards
        hand = state['hand']
        if not isinstance(hand, torch.Tensor):
            hand = torch.tensor(hand, dtype=torch.float32)
        hand = hand.to(device)
        if len(hand.shape) == 2:
            hand = hand.unsqueeze(0)

        # Process table cards
        table_cards = state['table_cards']
        if not isinstance(table_cards, torch.Tensor):
            table_cards = torch.tensor(table_cards, dtype=torch.float32)
        table_cards = table_cards.to(device)
        if len(table_cards.shape) == 2:
            table_cards = table_cards.unsqueeze(0)

        # Print shapes for debugging
        print(f"Hand shape: {hand.shape}")
        print(f"Table shape: {table_cards.shape}")

        # Encode cards
        encoded_hand = self.hand_encoder(hand.view(-1, 2))  # Reshape to (batch_size * cards, 2)
        encoded_hand = encoded_hand.view(batch_size, -1)  # Reshape back to (batch_size, cards * 16)

        encoded_table = self.table_encoder(table_cards.view(-1, 2))  # Reshape to (batch_size * cards, 2)
        encoded_table = encoded_table.view(batch_size, -1)  # Reshape back to (batch_size, cards * 16)

        # Process other state information
        current_player = state['current_player']
        if isinstance(current_player, torch.Tensor):
            current_player = current_player.clone().detach().float()
        else:
            current_player = torch.tensor(current_player, dtype=torch.float32)
        current_player = current_player.view(batch_size, 1).to(device)

        deck_size = state['deck_size']
        if isinstance(deck_size, torch.Tensor):
            deck_size = deck_size.clone().detach().float()
        else:
            deck_size = torch.tensor(deck_size, dtype=torch.float32)
        deck_size = deck_size.view(batch_size, 1).to(device)

        hand_sizes = state['player_hand_sizes']
        if isinstance(hand_sizes, torch.Tensor):
            hand_sizes = hand_sizes.clone().detach().float()
        else:
            hand_sizes = torch.tensor(hand_sizes, dtype=torch.float32)
        if len(hand_sizes.shape) == 1:
            hand_sizes = hand_sizes.unsqueeze(0)
        hand_sizes = hand_sizes.to(device)

        # Print shapes for debugging
        print(f"Encoded hand shape: {encoded_hand.shape}")
        print(f"Encoded table shape: {encoded_table.shape}")
        print(f"Current player shape: {current_player.shape}")
        print(f"Deck size shape: {deck_size.shape}")
        print(f"Hand sizes shape: {hand_sizes.shape}")

        # Combine all features
        combined = torch.cat([
            encoded_hand,
            encoded_table,
            current_player,
            deck_size,
            hand_sizes
        ], dim=1)

        print(f"Combined shape: {combined.shape}")
        print(f"Expected shape: {self.combined_size}")

        return self.network(combined)
