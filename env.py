import random
from typing import List, Tuple, Dict


def _find_combinations(played_card: Tuple[str, int], table_cards: List[Tuple[str, int]]) -> List[List[Tuple[str, int]]]:
    """Find all possible combinations of cards that sum to 15 including the played card."""

    def get_card_value(card: Tuple[str, int]) -> int:
        return min(card[1], 10)  # 10, 11, 12 count as 10

    def find_subsets(cards, target, current=None, start=0):
        if current is None:
            current = []
        if sum(get_card_value(c) for c in current) == target and played_card in current:
            result.append(current[:])
            return

        for i in range(start, len(cards)):
            current_sum = sum(get_card_value(c) for c in current)
            card_value = get_card_value(cards[i])
            if current_sum + card_value <= target:
                current.append(cards[i])
                find_subsets(cards, target, current, i + 1)
                current.pop()

    result = []
    all_cards = [played_card] + table_cards
    find_subsets(all_cards, 15)
    return [combo for combo in result if len(combo) >= 2]


def _create_deck() -> List[Tuple[str, int]]:
    """Create a Spanish deck of 40 cards."""
    suits = ['Oros', 'Copas', 'Espadas', 'Bastos']
    values = list(range(1, 8)) + list(range(10, 13))  # 1-7, 10-12
    deck = [(suit, value) for suit in suits for value in values]
    random.shuffle(deck)
    return deck


class SpanishCardGameEnv:
    def __init__(self, num_players: int = 4):
        if not 2 <= num_players <= 4:
            raise ValueError("Number of players must be between 2 and 4")

        self.num_players = num_players
        self.deck = _create_deck()
        self.player_hands = [[] for _ in range(num_players)]
        self.table_cards = []
        self.collected_cards = [[] for _ in range(num_players)]
        self.current_player = 0
        self.round_num = 0

    def reset(self) -> Dict:
        """Reset the environment to initial state."""
        self.deck = _create_deck()
        self.player_hands = [[] for _ in range(self.num_players)]
        self.table_cards = []
        self.collected_cards = [[] for _ in range(self.num_players)]
        self.current_player = 0
        self.round_num = 0

        # Initial deal
        self._deal_cards()

        return self._get_state()

    def _deal_cards(self) -> None:
        """Deal cards to players based on remaining deck size."""
        cards_left = len(self.deck)

        if cards_left == 0:
            return

        # Calculate cards per player for this round
        cards_per_player = min(3, cards_left // self.num_players)
        remaining_cards = cards_left - (cards_per_player * self.num_players)

        # Deal cards to players
        for player in range(self.num_players):
            for _ in range(cards_per_player):
                self.player_hands[player].append(self.deck.pop())

        # Put remaining cards on table
        for _ in range(remaining_cards):
            self.table_cards.append(self.deck.pop())

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Index of card in current player's hand to play

        Returns:
            observation: Current game state
            reward: Reward for the action
            done: Whether the episode is finished
            info: Additional information
        """
        if action >= len(self.player_hands[self.current_player]):
            raise ValueError("Invalid action: Card index out of bounds")

        # Play card
        played_card = self.player_hands[self.current_player].pop(action)

        # Find combinations
        combinations = _find_combinations(played_card, self.table_cards)

        if combinations:
            # Take the combination that collects the most cards
            best_combination = max(combinations, key=len)
            # Remove collected cards from table
            for card in best_combination:
                if card != played_card:
                    self.table_cards.remove(card)
            # Add to player's collected cards
            self.collected_cards[self.current_player].extend(best_combination)
            reward = len(best_combination)
        else:
            # If no combination, card goes to table
            self.table_cards.append(played_card)
            reward = 0

        # Move to next player
        self.current_player = (self.current_player + 1) % self.num_players

        # Check if round is over
        if all(len(hand) == 0 for hand in self.player_hands):
            self.round_num += 1
            if len(self.deck) > 0:
                self._deal_cards()

        # Check if game is over
        done = len(self.deck) == 0 and all(len(hand) == 0 for hand in self.player_hands)

        return self._get_state(), reward, done, {"round": self.round_num}

    def _get_state(self) -> Dict:
        """Return current game state."""
        return {
            "current_player": self.current_player,
            "player_hands": self.player_hands,
            "table_cards": self.table_cards,
            "collected_cards": self.collected_cards,
            "cards_in_deck": len(self.deck),
            "round": self.round_num
        }

    def get_legal_actions(self) -> List[int]:
        """Return list of legal actions for current player."""
        return list(range(len(self.player_hands[self.current_player])))

    def render(self) -> str:
        """Return a string representation of the current game state."""
        state = [
            f"Round: {self.round_num}",
            f"Current Player: {self.current_player}",
            f"Cards in deck: {len(self.deck)}",
            f"Table cards: {self.table_cards}",
            "Player hands:"
        ]
        for i, hand in enumerate(self.player_hands):
            state.append(f"  Player {i}: {hand}")
        state.append("Collected cards:")
        for i, collected in enumerate(self.collected_cards):
            state.append(f"  Player {i}: {len(collected)} cards")
        return "\n".join(state)
