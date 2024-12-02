import pygame
import numpy as np
from src import QuinzaEnv

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
FPS = 30
CARD_WIDTH, CARD_HEIGHT = 50, 75
FONT_SIZE = 24

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Initialize display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("QuinzaEnv Renderer")

# Initialize font
font = pygame.font.Font(None, FONT_SIZE)


def draw_card(card, x, y):
    """Draw a single card."""
    value, suit = card
    pygame.draw.rect(screen, WHITE, (x, y, CARD_WIDTH, CARD_HEIGHT), border_radius=5)
    pygame.draw.rect(screen, BLACK, (x, y, CARD_WIDTH, CARD_HEIGHT), 2, border_radius=5)
    text = font.render(f"{value}", True, BLACK)
    suit_text = font.render(f"{suit}", True, BLACK)
    screen.blit(text, (x + 5, y + 5))
    screen.blit(suit_text, (x + 5, y + 25))


def render_state(state, current_player):
    """Render the current state of the environment."""
    screen.fill(GREEN)

    # Draw current player
    player_text = font.render(f"Current Player: {current_player}", True, BLACK)
    screen.blit(player_text, (20, 20))

    # Draw player hands
    for i, hand in enumerate(state["player_hand_sizes"]):
        x_start = 20 + i * (CARD_WIDTH + 20)
        y_start = SCREEN_HEIGHT - CARD_HEIGHT - 50
        player_label = font.render(f"Player {i} Hand", True, BLUE)
        screen.blit(player_label, (x_start, y_start - 30))
        for j, card in enumerate(state["hand"]):
            draw_card(card, x_start + j * (CARD_WIDTH + 10), y_start)

    # Draw table cards
    table_start_x = 20
    table_start_y = 100
    table_label = font.render("Table Cards", True, RED)
    screen.blit(table_label, (table_start_x, table_start_y - 30))
    for i, card in enumerate(state["table_cards"]):
        draw_card(card, table_start_x + i * (CARD_WIDTH + 10), table_start_y)


def main():
    """Main loop for rendering the environment."""
    env = QuinzaEnv(num_players=4)
    state, info = env.reset()

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        render_state(state, env.current_player)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
