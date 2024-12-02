import optuna
import numpy as np
from datetime import datetime
import json
import torch
import logging
import os
from env import SpanishCardGameEnv
from dqn import DQNAgent
import trainer


class OptunaOptimizer:
    def __init__(self, env_creator, study_name="card_game_optimization"):
        self.env_creator = env_creator
        self.study_name = study_name
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            storage=f"sqlite:///{study_name}.db",
            load_if_exists=True
        )

    def objective(self, trial):
        params = {
            'gamma': trial.suggest_float('gamma', 0.8, 0.999),
            'epsilon_decay': trial.suggest_float('epsilon_decay', 0.99, 0.9999),
            'epsilon_min': trial.suggest_float('epsilon_min', 0.01, 0.1),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'hidden_size_1': trial.suggest_categorical('hidden_size_1', [64, 128, 256]),
            'hidden_size_2': trial.suggest_categorical('hidden_size_2', [32, 64, 128]),
        }

        try:
            env = self.env_creator()
            agent = DQNAgent(
                state_size=83,
                action_size=3,
                hidden_size_1=params['hidden_size_1'],
                hidden_size_2=params['hidden_size_2']
            )

            # Update other hyperparameters
            agent.gamma = params['gamma']
            agent.epsilon_decay = params['epsilon_decay']
            agent.epsilon_min = params['epsilon_min']
            agent.learning_rate = params['learning_rate']
            agent.batch_size = params['batch_size']
            agent.optimizer = torch.optim.Adam(
                agent.policy_net.parameters(),
                lr=params['learning_rate']
            )

            metrics = self.train_and_evaluate(env, agent, params)
            trial.report(metrics['win_rate'], step=100)

            if trial.should_prune():
                raise optuna.TrialPruned()

            return metrics['win_rate']

        except Exception as e:
            self.logger.error(f"Error in trial: {e}")
            raise optuna.TrialPruned()

    def train_and_evaluate(self, env, agent, params, n_episodes=500, eval_episodes=50):
        # Training phase
        for episode in range(n_episodes):
            state = env.reset()
            state = agent.encode_state(state)
            done = False

            while not done:
                legal_actions = env.get_legal_actions()
                action = agent.act(state, legal_actions)
                next_state_dict, reward, done, _ = env.step(action)
                next_state = agent.encode_state(next_state_dict)
                agent.remember(state, action, reward, next_state, done)
                agent.replay()
                state = next_state

        return self.evaluate_agent(env, agent, eval_episodes)

    def evaluate_agent(self, env, agent, n_episodes):
        total_reward = 0
        wins = 0
        original_epsilon = agent.epsilon
        agent.epsilon = agent.epsilon_min

        for _ in range(n_episodes):
            state = env.reset()
            state = agent.encode_state(state)
            episode_reward = 0
            done = False

            while not done:
                legal_actions = env.get_legal_actions()
                action = agent.act(state, legal_actions)
                next_state_dict, reward, done, _ = env.step(action)
                next_state = agent.encode_state(next_state_dict)
                episode_reward += reward
                state = next_state

            total_reward += episode_reward
            player_cards = [len(cards) for cards in env.collected_cards]
            if player_cards[0] == max(player_cards):
                wins += 1

        agent.epsilon = original_epsilon
        return {
            'average_reward': total_reward / n_episodes,
            'win_rate': wins / n_episodes
        }

    def optimize(self, n_trials=100):
        self.logger.info(f"Starting optimization with {n_trials} trials")
        self.study.optimize(self.objective, n_trials=n_trials, callbacks=[self.log_callback])

        best_params = self.study.best_params
        best_value = self.study.best_value

        self.logger.info(f"Best params found: {best_params}")
        self.logger.info(f"Best value achieved: {best_value}")
        self.save_results()
        return best_params

    def log_callback(self, study, trial):
        if trial.value is not None:
            self.logger.info(f"Trial {trial.number}")
            self.logger.info(f"Value: {trial.value}")
            self.logger.info(f"Params: {trial.params}")

    def save_results(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results = {
            'best_params': self.study.best_params,
            'best_value': self.study.best_value,
            'n_trials': len(self.study.trials),
            'trials': [
                {
                    'number': t.number,
                    'params': t.params,
                    'value': t.value,
                }
                for t in self.study.trials if t.value is not None
            ]
        }

        filename = f'optuna_results_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)


def main():
    def env_creator():
        return SpanishCardGameEnv(num_players=4)

    optimizer = OptunaOptimizer(env_creator)
    best_params = optimizer.optimize(n_trials=50)

    print("\nOptimization completed!")
    print("Best parameters found:")
    print(json.dumps(best_params, indent=2))

    # Create agent with best parameters
    env = env_creator()
    agent = DQNAgent(
        state_size=83,
        action_size=3,
        hidden_size_1=best_params['hidden_size_1'],
        hidden_size_2=best_params['hidden_size_2']
    )

    # Update other hyperparameters
    agent.gamma = best_params['gamma']
    agent.epsilon_decay = best_params['epsilon_decay']
    agent.epsilon_min = best_params['epsilon_min']
    agent.learning_rate = best_params['learning_rate']
    agent.batch_size = best_params['batch_size']
    agent.optimizer = torch.optim.Adam(
        agent.policy_net.parameters(),
        lr=best_params['learning_rate']
    )

    # Train the agent with best parameters
    trainer.train_agent(env, agent, num_episodes=1000)


if __name__ == "__main__":
    main()
