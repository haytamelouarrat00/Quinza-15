import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from collections import deque


def train_agent(env, agent, num_episodes=1000, target_update=10, eval_frequency=100):
    """
    Train the DQN agent and plot the results.
    """
    # Metrics tracking
    rewards_history = []
    cards_collected_history = []
    win_rate_history = []
    episode_lengths = []
    running_rewards = deque(maxlen=100)
    eval_episodes = []  # Track episodes where evaluation happened

    for episode in tqdm(range(num_episodes)):
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

        if episode % target_update == 0:
            agent.update_target_network()

        running_rewards.append(episode_reward)
        rewards_history.append(episode_reward)
        episode_lengths.append(steps)

        cards_collected = sum(len(cards) for cards in env.collected_cards)
        cards_collected_history.append(cards_collected)

        if episode % eval_frequency == 0:
            win_rate = evaluate_agent(env, agent, num_games=50)
            win_rate_history.append(win_rate)
            eval_episodes.append(episode)  # Store the episode number

            print(f"\nEpisode {episode}")
            print(f"Average Reward (last 100): {np.mean(running_rewards):.2f}")
            print(f"Win Rate: {win_rate:.2f}")
            print(f"Epsilon: {agent.epsilon:.3f}")

            plot_training_progress(rewards_history, cards_collected_history,
                                   win_rate_history, eval_episodes)


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


def plot_training_progress(rewards, cards_collected, win_rates, eval_episodes):
    """Plot training metrics."""
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
    plt.plot(eval_episodes, win_rates)  # Use eval_episodes for x-axis
    plt.title('Win Rate')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')

    plt.tight_layout()
    plt.show()


def save_agent(agent, filepath):
    """Save agent's policy network."""
    torch.save({
        'policy_net_state_dict': agent.policy_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon
    }, filepath)


def load_agent(agent, filepath):
    """Load agent's saved state."""
    checkpoint = torch.load(filepath)
    agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    agent.target_net.load_state_dict(checkpoint['policy_net_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.epsilon = checkpoint['epsilon']
