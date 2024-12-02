import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from enum import Enum
import random
from typing import List, Tuple, Dict, Optional
import numpy as np


class Suit(Enum):
    Gold = "gold"
    Cups = "cups"
    Spades = "spades"
    Clubs = "clubs"


@dataclass
class Card:
    value: int
    suit: Suit

    def __str__(self):
        return f"{self.value} of {self.suit.value}"

    def to_array(self) -> np.ndarray:
        """Convert card to numerical representation."""
        suit_map = {Suit.Gold: 0, Suit.Cups: 1, Suit.Spades: 2, Suit.Clubs: 3}
        return np.array([self.value, suit_map[self.suit]])


class QuinzaEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    # Precompute the deck once
    _precomputed_deck = [
        Card(value, suit)
        for suit in Suit
        for value in list(range(1, 8)) + list(range(10, 13))
    ]

    def __init__(self, num_players: int = 4, render_mode: Optional[str] = None):
        super().__init__()

        if not 2 <= num_players <= 4:
            raise ValueError("Number of players must be between 2 and 4")

        self.num_players = num_players
        self.render_mode = render_mode

        # Define action space (which card to play from hand, max 3 cards)
        self.action_space = spaces.Discrete(3)

        # Define observation space
        self.observation_space = spaces.Dict({
            # Hand: 3 cards max, each card has value (1-12) and suit (0-3)
            'hand': spaces.Box(low=np.array([[0, 0]] * 3),  # Changed from 1 to 0 to allow for padding
                               high=np.array([[12, 3]] * 3),
                               dtype=np.int32),
            # Table: 40 cards max (whole deck), each card has value and suit
            'table_cards': spaces.Box(low=np.array([[0, 0]] * 40),  # Changed from 1 to 0 to allow for padding
                                      high=np.array([[12, 3]] * 40),
                                      dtype=np.int32),
            # Current player
            'current_player': spaces.Discrete(num_players),
            # Deck size
            'deck_size': spaces.Discrete(41),  # 0 to 40 cards
            # Player hand sizes
            'player_hand_sizes': spaces.Box(low=np.zeros(num_players),
                                            high=np.array([3] * num_players),
                                            dtype=np.int32)
        })

        # Initialize state variables
        self.deck = None
        self.player_hands = None
        self.table_cards = None
        self.collected_cards = None
        self.current_player = None
        self.cards_played_this_round = None

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Copy the precomputed deck
        self.deck = self._precomputed_deck.copy()
        random.shuffle(self.deck)

        # Initialize game state
        self.player_hands = [[] for _ in range(self.num_players)]
        self.table_cards = []
        self.collected_cards = [[] for _ in range(self.num_players)]
        self.current_player = 0
        self.cards_played_this_round = []

        # Deal initial cards
        self._deal_cards()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute action and return new state."""
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is not in action space {self.action_space}")

        current_hand = self.player_hands[self.current_player]
        if not current_hand:
            return self._get_observation(), 0.0, True, False, self._get_info()

        if action >= len(current_hand):
            # Instead of raising an error, treat as an invalid action with negative reward
            return self._get_observation(), -1.0, False, False, self._get_info()

        # Play selected card
        played_card = current_hand.pop(action)
        self.cards_played_this_round.append(played_card)

        # Find and collect combinations
        reward = 0.0
        combinations = self._find_combinations(played_card)

        if combinations:
            # Take first valid combination
            collected = combinations[0]
            for card in collected[1:]:  # Skip the played card
                self.table_cards.remove(card)
            self.collected_cards[self.current_player].extend(collected)
            reward = float(len(collected))
        else:
            self.table_cards.append(played_card)

        # Move to next player
        self.current_player = (self.current_player + 1) % self.num_players

        # Check if round is over and deal new cards if needed
        if all(len(hand) == 0 for hand in self.player_hands):
            if len(self.deck) > 0:
                self._deal_cards()
                self.cards_played_this_round = []

        # Check if game is over
        terminated = len(self.deck) == 0 and all(len(hand) == 0 for hand in self.player_hands)
        truncated = False

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _deal_cards(self):
        """Deal cards to players according to game rules."""
        if not self.deck:
            return

        cards_per_player = min(3, len(self.deck) // self.num_players)
        if cards_per_player > 0:
            remainder = len(self.deck) % self.num_players
        else:
            remainder = len(self.deck)

        # Deal to players
        for i in range(self.num_players):
            for _ in range(cards_per_player):
                if self.deck:  # Additional check to prevent IndexError
                    self.player_hands[i].append(self.deck.pop())

        # Deal remainder to table
        for _ in range(remainder):
            if self.deck:  # Additional check to prevent IndexError
                self.table_cards.append(self.deck.pop())

    def _get_observation(self) -> Dict:
        """Convert current state to observation space format."""
        # Convert current player's hand to numerical representation with padding
        hand = np.zeros((3, 2), dtype=np.int32)
        current_hand = self.player_hands[self.current_player]
        for i, card in enumerate(current_hand):
            if i < 3:  # Ensure we don't exceed the maximum hand size
                hand[i] = card.to_array()

        # Convert table cards to numerical representation with padding
        table = np.zeros((40, 2), dtype=np.int32)
        for i, card in enumerate(self.table_cards):
            if i < 40:  # Ensure we don't exceed the maximum table size
                table[i] = card.to_array()

        return {
            'hand': hand,
            'table_cards': table,
            'current_player': np.array(self.current_player, dtype=np.int32),
            'deck_size': np.array(len(self.deck), dtype=np.int32),
            'player_hand_sizes': np.array([len(hand) for hand in self.player_hands], dtype=np.int32)
        }

    def _get_info(self) -> Dict:
        """Return additional information about the current state."""
        return {
            'collected_cards': [len(cards) for cards in self.collected_cards],
            'cards_played_this_round': len(self.cards_played_this_round),
            'valid_actions': list(range(len(self.player_hands[self.current_player])))
        }

    def render(self):
        """Render the current state of the game."""
        if self.render_mode == "human":
            print(f"\nCurrent Player: {self.current_player}")
            print(f"Player Hand: {[str(card) for card in self.player_hands[self.current_player]]}")
            print(f"Table Cards: {[str(card) for card in self.table_cards]}")
            print(f"Collected Cards: {[len(cards) for cards in self.collected_cards]}")
            print(f"Deck Size: {len(self.deck)}")

    def _find_combinations(self, played_card: Card) -> List[List[Card]]:
        """Find all combinations of cards that sum to 15."""
        combinations = []
        target = 15 - played_card.value

        if not self.table_cards:
            return combinations

        # Single card combinations
        for card in self.table_cards:
            if card.value == target:
                combinations.append([played_card, card])

        if len(self.table_cards) < 2 or target <= 2:
            return combinations

        # Two card combinations
        for i in range(len(self.table_cards)):
            for j in range(i + 1, len(self.table_cards)):
                if self.table_cards[i].value + self.table_cards[j].value == target:
                    combinations.append([played_card, self.table_cards[i], self.table_cards[j]])

        return combinations
