import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from collections import deque
import logging
from pathlib import Path
import json


def save_metrics(log_dir: Path, metrics: dict, current_epsilon: float) -> None:
    """Save training metrics for GUI visualization."""
    metrics_data = {
        'rewards_history': metrics['rewards_history'],
        'cards_collected_history': metrics['cards_collected_history'],
        'win_rate_history': metrics['win_rate_history'],
        'eval_episodes': metrics['eval_episodes'],
        'current_epsilon': current_epsilon,
        'episode_lengths': metrics['episode_lengths']
    }

    with open(log_dir / 'training_metrics.json', 'w') as f:
        json.dump(metrics_data, f)


def evaluate_agent(env, agent, num_games=50):
    """Evaluate agent performance without exploration."""
    wins = 0
    original_epsilon = agent.epsilon
    agent.epsilon = 0

    for _ in range(num_games):
        state = env.reset()
        state = agent.encode_state(state)
        done = False

        while not done:
            legal_actions = env.get_legal_actions()
            action = agent.act(state, legal_actions)
            next_state_dict, _, done, _ = env.step(action)
            state = agent.encode_state(next_state_dict)

        player_cards = [len(cards) for cards in env.collected_cards]
        if player_cards[0] == max(player_cards):
            wins += 1

    agent.epsilon = original_epsilon
    return wins / num_games


def plot_training_progress(rewards, cards_collected, win_rates, eval_episodes, log_dir: Path):
    """Plot training metrics and save to log directory."""
    plt.figure(figsize=(15, 5))

    # Plot rewards
    plt.subplot(131)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    # Plot cards collected
    plt.subplot(132)
    plt.plot(cards_collected)
    plt.title('Cards Collected')
    plt.xlabel('Episode')
    plt.ylabel('Number of Cards')

    # Plot win rate
    plt.subplot(133)
    plt.plot(eval_episodes, win_rates)
    plt.title('Win Rate')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')

    plt.tight_layout()
    plt.savefig(log_dir / 'training_progress.png')
    plt.close()


def save_agent(agent, filepath: Path) -> None:
    """Save agent's policy network and training state."""
    torch.save({
        'policy_net_state_dict': agent.policy_net.state_dict(),
        'target_net_state_dict': agent.target_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon
    }, filepath)


def load_agent(agent, filepath: Path) -> None:
    """Load agent's saved state."""
    checkpoint = torch.load(filepath)
    agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.epsilon = checkpoint['epsilon']


def train_agent(env, agent, config):
    """
    Train the DQN agent and plot the results.

    Args:
        env: Game environment
        agent: DQN agent
        config: Configuration object containing training parameters
    """
    # Setup logging
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
    logger = logging.getLogger(__name__)

    # Initialize metrics dictionary
    metrics = {
        'rewards_history': [],
        'cards_collected_history': [],
        'win_rate_history': [],
        'eval_episodes': [],
        'episode_lengths': []
    }
    running_rewards = deque(maxlen=100)

    # Create checkpoint directory
    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    for episode in tqdm(range(config.training.num_episodes)):
        state = env.reset()
        state = agent.encode_state(state)
        episode_reward = 0
        steps = 0

        while True:
            legal_actions = env.get_legal_actions()
            action = agent.act(state, legal_actions)

            next_state_dict, reward, done, info = env.step(action)
            next_state = agent.encode_state(next_state_dict)

            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            state = next_state
            episode_reward += reward
            steps += 1

            if done:
                break

        if episode % config.dqn.target_update_frequency == 0:
            agent.update_target_network()

        # Update metrics
        running_rewards.append(episode_reward)
        metrics['rewards_history'].append(episode_reward)
        metrics['episode_lengths'].append(steps)

        cards_collected = sum(len(cards) for cards in env.collected_cards)
        metrics['cards_collected_history'].append(cards_collected)

        # Save metrics every episode for smoother GUI updates
        save_metrics(log_dir, metrics, agent.epsilon)

        # Evaluation phase
        if episode % config.training.eval_frequency == 0:
            win_rate = evaluate_agent(env, agent, config.training.num_eval_episodes)
            metrics['win_rate_history'].append(win_rate)
            metrics['eval_episodes'].append(episode)

            logger.info(f"\nEpisode {episode}")
            logger.info(f"Average Reward (last 100): {np.mean(running_rewards):.2f}")
            logger.info(f"Win Rate: {win_rate:.2f}")
            logger.info(f"Epsilon: {agent.epsilon:.3f}")

            # Save checkpoint
            checkpoint_path = checkpoint_dir / f"checkpoint_episode_{episode}.pt"
            save_agent(agent, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

            plot_training_progress(
                metrics['rewards_history'],
                metrics['cards_collected_history'],
                metrics['win_rate_history'],
                metrics['eval_episodes'],
                log_dir
            )

    # Save final model
    final_path = checkpoint_dir / "final_model.pt"
    save_agent(agent, final_path)
    logger.info(f"Saved final model to {final_path}")

    return metrics
