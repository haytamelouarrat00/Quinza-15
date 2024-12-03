from typing import List, Tuple, Dict, Optional, Union
import random
from dataclasses import dataclass, asdict
import json
from pathlib import Path


class IllegalActionError(Exception):
    """Raised when an illegal action is attempted."""
    pass


@dataclass
class Card:
    """Representation of a Spanish playing card."""
    suit: str
    value: int

    def __iter__(self):
        return iter((self.suit, self.value))

    def __str__(self):
        return f"{self.suit}-{self.value}"

    @property
    def score_value(self) -> int:
        """Get the scoring value of the card (10, 11, 12 count as 10)."""
        return min(self.value, 10)


class SpanishCardGameEnv:
    """Spanish card game environment with proper space definitions and improved state management."""

    SUITS = ['Oros', 'Copas', 'Espadas', 'Bastos']
    VALUES = list(range(1, 8)) + list(range(10, 13))  # 1-7, 10-12

    def __init__(self, config):
        """Initialize environment with configuration."""
        self.history = None
        self.round_num = None
        self.collected_cards = None
        self.table_cards = None
        self.player_hands = None
        self.current_player = None
        self.config = config
        self.num_players = config.game.num_players
        self.target_sum = config.game.target_sum

        # Define observation and action spaces
        self.observation_space = {
            'hand': (40,),  # One-hot encoding of cards in hand
            'table': (40,),  # One-hot encoding of cards on table
            'state_info': (3,)  # [cards_in_deck/40, current_player/num_players, collected_cards/40]
        }

        self.action_space = config.game.cards_per_deal

        self.reset()

    def reset(self) -> Dict:
        """Reset the environment to initial state."""
        # Create and shuffle deck
        self.deck = [Card(suit, value) for suit in self.SUITS for value in self.VALUES]
        random.shuffle(self.deck)

        # Initialize game state
        self.player_hands = [[] for _ in range(self.num_players)]
        self.table_cards = []
        self.collected_cards = [[] for _ in range(self.num_players)]
        self.current_player = 0
        self.round_num = 0
        self.history = []  # Track game history

        # Initial deal
        self._deal_cards()

        return self._get_state()

    def _validate_action(self, action: int) -> None:
        """Validate if an action is legal."""
        if not isinstance(action, int):
            raise TypeError(f"Action must be an integer, got {type(action)}")

        if action < 0 or action >= len(self.player_hands[self.current_player]):
            raise IllegalActionError(f"Invalid action index: {action}")

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """Execute one step in the environment with improved reward shaping."""
        try:
            self._validate_action(action)

            # Play card
            played_card = self.player_hands[self.current_player].pop(action)

            # Find combinations
            combinations = self._find_combinations(played_card)
            reward = 0

            if combinations:
                # Take the combination that collects the most cards
                best_combination = max(combinations, key=len)

                # Remove collected cards from table
                for card in best_combination:
                    if card != played_card:
                        self.table_cards.remove(card)

                # Add to player's collected cards
                self.collected_cards[self.current_player].extend(best_combination)

                # Reward shaping
                base_reward = len(best_combination)
                sum_value = sum(c.score_value for c in best_combination)

                # Bonus for collecting more cards
                size_bonus = 0.2 * (len(best_combination) - 2)  # Bonus for combinations larger than 2

                # Bonus for collecting high-value cards
                value_bonus = 0.1 * (sum_value - len(best_combination))  # Bonus for cards worth more than 1

                reward = base_reward + size_bonus + value_bonus

            else:
                # Card goes to table
                self.table_cards.append(played_card)
                # Small negative reward for not making a combination
                reward = -0.1

            # Track move in history
            self.history.append({
                'player': self.current_player,
                'action': str(played_card),
                'combination': [str(c) for c in combinations[0]] if combinations else None,
                'reward': reward
            })

            # Move to next player
            self.current_player = (self.current_player + 1) % self.num_players

            # Check if round is over
            round_over = all(len(hand) == 0 for hand in self.player_hands)
            if round_over:
                self.round_num += 1
                if len(self.deck) > 0:
                    self._deal_cards()

            # Check if game is over
            done = len(self.deck) == 0 and round_over

            # Add end-game rewards
            if done:
                end_game_reward = self._calculate_end_game_reward()
                reward += end_game_reward

            return self._get_state(), reward, done, {
                'round': self.round_num,
                'round_over': round_over,
                'cards_collected': len(self.collected_cards[self.current_player])
            }

        except IllegalActionError as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Error executing step: {str(e)}")

    def _calculate_end_game_reward(self) -> float:
        """Calculate end-game rewards based on final state."""
        player_cards = [len(cards) for cards in self.collected_cards]
        max_cards = max(player_cards)

        if player_cards[0] == max_cards:  # If current player has most cards
            if player_cards.count(max_cards) == 1:  # Clear winner
                return 5.0
            else:  # Tie for most cards
                return 2.0
        return 0.0

    def _find_combinations(self, played_card: Card) -> List[List[Card]]:
        """Find all possible combinations of cards that sum to target including the played card."""

        def find_subsets(cards: List[Card], target: int, current: Optional[List[Card]] = None, start: int = 0) -> None:
            if current is None:
                current = []

            current_sum = sum(c.score_value for c in current)
            if current_sum == target and played_card in current:
                result.append(current[:])
                return

            for i in range(start, len(cards)):
                if current_sum + cards[i].score_value <= target:
                    current.append(cards[i])
                    find_subsets(cards, target, current, i + 1)
                    current.pop()

        result = []
        all_cards = [played_card] + self.table_cards
        find_subsets(all_cards, self.target_sum)
        return [combo for combo in result if len(combo) >= 2]

    def _deal_cards(self) -> None:
        """Deal cards to players based on remaining deck size."""
        if not self.deck:
            return

        cards_per_player = min(self.config.game.cards_per_deal,
                               len(self.deck) // self.num_players)
        remaining_cards = len(self.deck) - (cards_per_player * self.num_players)

        # Deal to players
        for player in range(self.num_players):
            for _ in range(cards_per_player):
                self.player_hands[player].append(self.deck.pop())

        # Put remaining cards on table
        for _ in range(remaining_cards):
            self.table_cards.append(self.deck.pop())

    def get_legal_actions(self) -> List[int]:
        """Return list of legal actions for current player."""
        return list(range(len(self.player_hands[self.current_player])))

    def render(self, mode='human') -> Dict:
        """Return a game state representation suitable for GUI rendering."""
        state = {
            'round': self.round_num,
            'current_player': self.current_player,
            'cards_in_deck': len(self.deck),
            'table_cards': [(card.suit, card.value) for card in self.table_cards],
            'player_hands': [
                [(card.suit, card.value) for card in hand]
                for hand in self.player_hands
            ],
            'collected_cards': [
                [(card.suit, card.value) for card in cards]
                for cards in self.collected_cards
            ],
            'total_collected': [len(cards) for cards in self.collected_cards]
        }

        if mode == 'human':
            # Return string representation for console display
            output = [
                f"Round: {state['round']}",
                f"Current Player: {state['current_player']}",
                f"Cards in deck: {state['cards_in_deck']}",
                f"Table cards: {state['table_cards']}",
                "Player hands:"
            ]
            for i, hand in enumerate(state['player_hands']):
                output.append(f"  Player {i}: {hand}")
            output.append("Collected cards:")
            for i, count in enumerate(state['total_collected']):
                output.append(f"  Player {i}: {count} cards")
            return "\n".join(output)

        return state  # Return dictionary for GUI rendering

    def _get_state(self) -> Dict:
        """Return current game state with proper encoding."""
        return {
            'current_player': self.current_player,
            'player_hands': [[tuple(card) for card in hand] for hand in self.player_hands],
            'table_cards': [tuple(card) for card in self.table_cards],
            'collected_cards': [[tuple(card) for card in cards] for cards in self.collected_cards],
            'cards_in_deck': len(self.deck),
            'round': self.round_num
        }

    def save_game(self, filepath: Union[str, Path]) -> None:
        """Save game state and history to file."""
        game_data = {
            'config': asdict(self.config),  # Convert config dataclass to dict
            'state': self._get_state(),
            'history': self.history
        }

        # Convert filepath to Path object
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Use explicit encoding for text file
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(game_data, f, indent=2, ensure_ascii=False)

    def load_game(self, filepath: Union[str, Path]) -> None:
        """Load game state from file."""
        path = Path(filepath)

        with open(path, 'r', encoding='utf-8') as f:
            game_data = json.load(f)

        # Reconstruct game state
        state = game_data['state']
        self.current_player = state['current_player']
        self.round_num = state['round']
        self.history = game_data['history']

        # Reconstruct cards
        self.player_hands = [[Card(s, v) for s, v in hand] for hand in state['player_hands']]
        self.table_cards = [Card(s, v) for s, v in state['table_cards']]
        self.collected_cards = [[Card(s, v) for s, v in cards] for cards in state['collected_cards']]
