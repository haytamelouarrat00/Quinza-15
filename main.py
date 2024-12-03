import argparse
from pathlib import Path
from dqn_config import Config, get_default_config
from env import SpanishCardGameEnv
from dqn import DQNAgent
import trainer
from hptune import HyperparameterOptimizer
from gui_game import create_visualization
import logging
import sys
import threading
import time
from PyQt5.QtCore import QThread, pyqtSignal


def setup_logging(config):
    """Setup logging configuration."""
    log_dir = Path(config.training.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "training.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def create_env_and_agent(config, checkpoint_path=None):
    """Create and initialize environment and agent."""
    env = SpanishCardGameEnv(config)
    agent = DQNAgent(config)

    if checkpoint_path:
        agent.load(checkpoint_path)
        logging.info(f"Loaded checkpoint from {checkpoint_path}")

    return env, agent


def run_optimization(config, n_trials):
    """Run hyperparameter optimization mode."""
    logger = setup_logging(config)
    logger.info("Starting hyperparameter optimization mode")

    def env_creator():
        return SpanishCardGameEnv(config)

    optimizer = HyperparameterOptimizer(config, env_creator)
    best_params = optimizer.optimize(n_trials=n_trials)

    # Create agent with best parameters and run final training
    config.dqn.update(best_params)
    env, agent = create_env_and_agent(config)

    logger.info("Training agent with best parameters...")
    trainer.train_agent(env=env, agent=agent, config=config)


class TrainingThread(QThread):
    """Thread for running the training process."""
    # Define the signal at class level
    update_signal = pyqtSignal(dict)

    def __init__(self, config, parent=None):
        """Initialize the training thread."""
        super(TrainingThread, self).__init__(parent)
        self.config = config
        self.stop_flag = False

    def run(self):
        """Run training process."""
        env = SpanishCardGameEnv(self.config)
        agent = DQNAgent(self.config)

        for episode in range(self.config.training.num_episodes):
            if self.stop_flag:
                break

            state = env.reset()
            done = False

            while not done and not self.stop_flag:
                # Update GUI with current state
                state = env.render(mode='gui')
                self.update_signal.emit(state)

                # Get action from agent
                encoded_state = agent.encode_state(state)
                legal_actions = env.get_legal_actions()
                action = agent.act(encoded_state, legal_actions)

                # Take step
                next_state, reward, done, _ = env.step(action)

                # Store experience and train
                agent.remember(encoded_state, action, reward,
                               agent.encode_state(next_state), done)
                agent.replay()

                state = next_state

                # Slow down visualization
                time.sleep(0.5)  # Add delay to make it easier to follow

            # Update target network periodically
            if episode % self.config.dqn.target_update_frequency == 0:
                agent.update_target_network()

            # Save checkpoint periodically
            if episode % self.config.training.save_frequency == 0:
                checkpoint_dir = Path(self.config.training.checkpoint_dir)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_dir / f"checkpoint_episode_{episode}.pt"
                agent.save(str(checkpoint_path))

    def stop(self):
        """Stop the training process."""
        self.stop_flag = True


def run_training_with_gui(config):
    """Run training mode with GUI visualization."""
    # Create and show visualization
    app, window = create_visualization(config.game.num_players)

    # Create training thread with window as parent
    training_thread = TrainingThread(config, parent=window)

    # Connect signal to slot
    training_thread.update_signal.connect(window.update_state)

    # Start training in separate thread
    training_thread.start()

    # Show window and run GUI
    window.show()

    try:
        sys.exit(app.exec_())
    finally:
        # Ensure training thread is stopped when application closes
        training_thread.stop()
        training_thread.wait()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Spanish Card Game AI Training')
    parser.add_argument('--mode', type=str, choices=['train', 'optimize'],
                        default='train', help='Mode to run in')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--trials', type=int, default=50,
                        help='Number of trials for optimization')
    args = parser.parse_args()

    # Load configuration
    config = Config.load(args.config) if Path(args.config).exists() else get_default_config()

    if args.checkpoint_dir:
        config.training.checkpoint_dir = args.checkpoint_dir

    logger = setup_logging(config)

    if args.mode == 'optimize':
        run_optimization(config, args.trials)
    else:
        run_training_with_gui(config)


if __name__ == "__main__":
    main()
