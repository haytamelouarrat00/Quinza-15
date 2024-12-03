from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QFrame, QGridLayout)
from PyQt5.QtCore import Qt
import sys


class CardWidget(QFrame):
    """Widget to display a single card."""

    def __init__(self, suit, value):
        super().__init__()
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.setLineWidth(2)
        self.setMinimumSize(60, 90)
        self.setMaximumSize(60, 90)

        layout = QVBoxLayout()

        # Convert value to display format
        if value == 10:
            display_value = "Sota"
        elif value == 11:
            display_value = "Caballo"
        elif value == 12:
            display_value = "Rey"
        else:
            display_value = str(value)

        # Create labels
        value_label = QLabel(display_value)
        suit_label = QLabel(suit)

        # Center align labels
        value_label.setAlignment(Qt.AlignCenter)
        suit_label.setAlignment(Qt.AlignCenter)

        # Add to layout
        layout.addWidget(value_label)
        layout.addWidget(suit_label)

        self.setLayout(layout)


class PlayerHandWidget(QFrame):
    """Widget to display a player's hand."""

    def __init__(self, player_num, is_current=False):
        super().__init__()
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.player_num = player_num
        self.is_current = is_current

        self.layout = QVBoxLayout()

        # Player label
        self.player_label = QLabel(f"Player {player_num}")
        self.player_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.player_label)

        # Cards layout
        self.cards_layout = QHBoxLayout()
        self.layout.addLayout(self.cards_layout)

        # Collected cards label
        self.collected_label = QLabel("Collected: 0")
        self.collected_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.collected_label)

        self.setLayout(self.layout)
        self.update_current_player(is_current)

    def update_current_player(self, is_current):
        """Update visual indication of current player."""
        self.is_current = is_current
        if is_current:
            self.setStyleSheet("background-color: lightgreen;")
            self.player_label.setText(f"Player {self.player_num} (Current)")
        else:
            self.setStyleSheet("")
            self.player_label.setText(f"Player {self.player_num}")

    def update_hand(self, cards, collected_count):
        """Update the displayed cards and collected count."""
        # Clear current cards
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add new cards
        for card in cards:
            card_widget = CardWidget(card[0], card[1])
            self.cards_layout.addWidget(card_widget)

        # Update collected cards count
        self.collected_label.setText(f"Collected: {collected_count}")


class TableCardsWidget(QFrame):
    """Widget to display cards on the table."""

    def __init__(self):
        super().__init__()
        self.setFrameStyle(QFrame.Box | QFrame.Raised)

        self.layout = QVBoxLayout()

        # Table label
        table_label = QLabel("Table Cards")
        table_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(table_label)

        # Cards layout
        self.cards_layout = QHBoxLayout()
        self.layout.addLayout(self.cards_layout)

        self.setLayout(self.layout)

    def update_cards(self, cards):
        """Update the displayed table cards."""
        # Clear current cards
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add new cards
        for card in cards:
            card_widget = CardWidget(card[0], card[1])
            self.cards_layout.addWidget(card_widget)


class CardGameGUI(QMainWindow):
    def __init__(self, num_players=4):
        super().__init__()
        self.num_players = num_players
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle('Spanish Card Game Visualization')
        self.setGeometry(100, 100, 1200, 800)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Game info section
        info_layout = QHBoxLayout()
        self.deck_label = QLabel("Cards in deck: 40")
        self.round_label = QLabel("Round: 0")
        info_layout.addWidget(self.deck_label)
        info_layout.addWidget(self.round_label)
        main_layout.addLayout(info_layout)

        # Table cards
        self.table_widget = TableCardsWidget()
        main_layout.addWidget(self.table_widget)

        # Player hands
        self.player_widgets = []
        players_layout = QGridLayout()

        # Arrange players in a circle around the table
        positions = [
            (2, 1),  # Bottom (Player 0)
            (1, 2),  # Right (Player 1)
            (0, 1),  # Top (Player 2)
            (1, 0)  # Left (Player 3)
        ]

        for i in range(self.num_players):
            player_widget = PlayerHandWidget(i, i == 0)
            self.player_widgets.append(player_widget)
            row, col = positions[i]
            players_layout.addWidget(player_widget, row, col)

        main_layout.addLayout(players_layout)

    def update_state(self, state):
        # Update round and deck info
        self.deck_label.setText(f"Cards in deck: {state['cards_in_deck']}")
        self.round_label.setText(f"Round: {state['round']}")

        # Update table cards
        self.table_widget.update_cards(state['table_cards'])

        # Update player hands
        current_player = state['current_player']
        for i, widget in enumerate(self.player_widgets):
            widget.update_current_player(i == current_player)
            widget.update_hand(
                state['player_hands'][i],
                state['total_collected'][i]
            )


def create_visualization(num_players=4):
    """Create and show the visualization window."""
    app = QApplication(sys.argv)
    window = CardGameGUI(num_players)
    window.show()
    return app, window


if __name__ == "__main__":
    # Example usage with mock game state
    mock_state = {
        'current_player': 0,
        'player_hands': [
            [('Oros', 1), ('Copas', 7)],
            [('Espadas', 10), ('Bastos', 11)],
            [('Oros', 12), ('Copas', 2)],
            [('Espadas', 3), ('Bastos', 4)]
        ],
        'table_cards': [('Oros', 5), ('Copas', 6)],
        'collected_cards': [
            [('Bastos', 1), ('Copas', 3)],
            [('Oros', 2)],
            [('Espadas', 4)],
            []
        ],
        'cards_in_deck': 30,
        'round': 2
    }

    app, window = create_visualization()
    window.update_state(mock_state)
    sys.exit(app.exec_())
