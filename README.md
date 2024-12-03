# Spanish Card Game AI with Deep Q-Learning

A deep reinforcement learning implementation for playing the traditional Spanish card game "Quinze" (Fifteen). The AI
agent learns to play the game using Deep Q-Learning (DQN) with experience replay.

## Game Rules

The game is played with a Spanish deck of 40 cards. Players take turns playing cards and try to form combinations that
sum to 15. Key rules:

- Each player receives 3 cards per round
- Cards 1-7 retain their face value
- Cards 10-12 (Sota, Caballo, Rey) count as 10
- Players can collect combinations of cards that sum to 15
- The player who collects the most cards wins

## Project Structure

```
├── src/
│   ├── dqn.py         # DQN agent implementation
│   ├── env.py         # Game environment
│   ├── trainer.py     # Training utilities
│   ├── hptune.py     # Hyperparameter optimization
│   ├── gui_game.py   # PyGame visualization
│   └── main.py       # Main training script
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Pygame (for visualization)
- Optuna (for hyperparameter tuning)
- Matplotlib
- tqdm

Install dependencies:

```bash
pip install torch numpy pygame optuna matplotlib tqdm
```

## Usage

### Training the Agent

```python
python
main.py
```

This will:

1. Initialize the game environment and DQN agent
2. Train the agent for 1000 episodes
3. Display training metrics including rewards, cards collected, and win rate

### Hyperparameter Optimization

```python
python
hptune.py
```

Uses Optuna to find optimal hyperparameters for:

- Learning rate
- Discount factor (gamma)
- Epsilon decay rate
- Network architecture
- Batch size

### Visualization

```python
python
gui_game.py
```

Launches a PyGame window showing:

- Current game state
- Player hands
- Table cards
- Game progress

## Model Architecture

The DQN implementation features:

- Two-layer neural network for Q-value prediction
- Experience replay buffer
- Target network for stable learning
- Epsilon-greedy exploration strategy

State representation includes:

- One-hot encoding of cards in hand (40 dimensions)
- One-hot encoding of table cards (40 dimensions)
- Additional game state information (3 dimensions)

## Performance

The agent typically achieves:

- ~60-70% win rate against random opponents after 1000 episodes
- Stable learning curve after ~500 episodes
- Effective card combination formation

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Implementation inspired by the DQN paper "Playing Atari with Deep Reinforcement Learning"
- Spanish card game rules adapted from traditional Quinze variants

## Future Improvements

- Implement multi-agent training
- Add priority experience replay
- Enhance state representation
- Add support for different Spanish card games
- Improve GUI visualization