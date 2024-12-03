from dataclasses import dataclass, asdict, field
from typing import Dict, Any
import yaml
from pathlib import Path


@dataclass
class DQNConfig:
    # Network architecture
    state_size: int = 83
    action_size: int = 3
    hidden_size_1: int = 128
    hidden_size_2: int = 64

    # Training hyperparameters
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    learning_rate: float = 0.001
    batch_size: int = 64
    buffer_size: int = 10000
    target_update_frequency: int = 10

    # Prioritized replay parameters
    use_priority_replay: bool = False
    priority_alpha: float = 0.6
    priority_beta_start: float = 0.4
    priority_beta_end: float = 1.0
    priority_beta_steps: int = 100000

    def update(self, params: Dict[str, Any]) -> None:
        """Update config with new parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)


@dataclass
class GameConfig:
    num_players: int = 4  # Maximum 4 players
    cards_per_deal: int = 3  # 3 cards per deal
    target_sum: int = 15  # Target sum for combinations
    points_per_card: int = 1  # Points for each card collected
    points_for_most_cards: int = 2  # Bonus points for most cards

    def validate(self):
        """Validate game configuration."""
        if self.num_players < 2 or self.num_players > 4:
            raise ValueError("Number of players must be between 2 and 4")
        if self.cards_per_deal != 3:
            raise ValueError("Cards per deal must be 3 for this game")
        if self.target_sum != 15:
            raise ValueError("Target sum must be 15 for this game")

    def update(self, params: Dict[str, Any]) -> None:
        """Update config with new parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.validate()


@dataclass
class TrainingConfig:
    num_episodes: int = 1000
    eval_frequency: int = 100
    num_eval_episodes: int = 50
    save_frequency: int = 100
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    def update(self, params: Dict[str, Any]) -> None:
        """Update config with new parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)


@dataclass
class Config:
    dqn: DQNConfig = field(default_factory=DQNConfig)
    game: GameConfig = field(default_factory=GameConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def load(cls, config_path: str) -> 'Config':
        """Load configuration from a YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        config = cls()
        if 'dqn' in config_dict:
            config.dqn.update(config_dict['dqn'])
        if 'game' in config_dict:
            config.game.update(config_dict['game'])
        if 'training' in config_dict:
            config.training.update(config_dict['training'])
        return config

    def save(self, config_path: str) -> None:
        """Save configuration to a YAML file."""
        config_dict = {
            'dqn': asdict(self.dqn),
            'game': asdict(self.game),
            'training': asdict(self.training)
        }

        # Create directory if it doesn't exist
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def update(self, params: Dict[str, Any]) -> None:
        """Update configuration with new parameters."""
        if 'dqn' in params:
            self.dqn.update(params['dqn'])
        if 'game' in params:
            self.game.update(params['game'])
        if 'training' in params:
            self.training.update(params['training'])


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


# Example usage
if __name__ == "__main__":
    # Create default config
    config = get_default_config()

    # Example of updating parameters
    new_params = {
        'dqn': {
            'hidden_size_1': 256,
            'learning_rate': 0.0001
        },
        'training': {
            'num_episodes': 2000
        }
    }

    # Update config
    config.update(new_params)

    # Save to file
    config.save('config/default_config.yaml')

    # Load from file
    loaded_config = Config.load('config/default_config.yaml')

    # Verify updates
    print(f"DQN hidden size 1: {loaded_config.dqn.hidden_size_1}")
    print(f"Learning rate: {loaded_config.dqn.learning_rate}")
    print(f"Number of episodes: {loaded_config.training.num_episodes}")
