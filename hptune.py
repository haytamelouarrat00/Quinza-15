import optuna
from optuna.trial import TrialState
import numpy as np
from datetime import datetime
import json
import logging
from pathlib import Path
import mlflow
from typing import Dict, Any, Callable
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial


class HyperparameterOptimizer:
    def __init__(self, config, env_creator: Callable, study_name: str = "card_game_optimization"):
        """
        Initialize the hyperparameter optimizer.

        Args:
            config: Configuration object
            env_creator: Function that creates the environment
            study_name: Name for the optimization study
        """
        self.config = config
        self.env_creator = env_creator
        self.study_name = study_name

        # Setup logging
        self.log_dir = Path("logs") / study_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / "optimization.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Initialize MLflow
        mlflow.set_experiment(study_name)

        # Create Optuna study with pruning
        self.study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=20,
                interval_steps=10
            ),
            storage=f"sqlite:///{study_name}.db",
            load_if_exists=True
        )

    def suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for the trial."""
        params = {
            # Network architecture
            'hidden_size_1': trial.suggest_categorical('hidden_size_1', [64, 128, 256, 512]),
            'hidden_size_2': trial.suggest_categorical('hidden_size_2', [32, 64, 128, 256]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),

            # Training parameters
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256]),
            'gamma': trial.suggest_float('gamma', 0.9, 0.999),
            'target_update_freq': trial.suggest_int('target_update_freq', 5, 20),

            # Exploration parameters
            'epsilon_decay': trial.suggest_float('epsilon_decay', 0.99, 0.999),
            'epsilon_min': trial.suggest_float('epsilon_min', 0.01, 0.1),

            # Advanced features
            'use_double_dqn': trial.suggest_categorical('use_double_dqn', [True, False]),
            'use_dueling_network': trial.suggest_categorical('use_dueling_network', [True, False]),
            'use_priority_replay': trial.suggest_categorical('use_priority_replay', [True, False])
        }

        if params['use_priority_replay']:
            params.update({
                'priority_alpha': trial.suggest_float('priority_alpha', 0.4, 0.8),
                'priority_beta_start': trial.suggest_float('priority_beta_start', 0.3, 0.7)
            })

        return params

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for optimization."""
        # Get hyperparameters for this trial
        params = self.suggest_parameters(trial)

        # Start MLflow run
        with mlflow.start_run(nested=True):
            # Log parameters
            mlflow.log_params(params)

            try:
                # Create environment and agent
                env = self.env_creator()
                agent = self._create_agent(params)

                # Train and evaluate
                metrics = self._train_and_evaluate(env, agent, params, trial)

                # Log metrics
                mlflow.log_metrics(metrics)

                # Report intermediate value for pruning
                trial.report(metrics['win_rate'], step=100)

                if trial.should_prune():
                    raise optuna.TrialPruned()

                return metrics['win_rate']

            except Exception as e:
                self.logger.error(f"Error in trial: {str(e)}")
                raise optuna.TrialPruned()

    def _train_and_evaluate(self, env, agent, params: Dict[str, Any], trial: optuna.Trial) -> Dict[str, float]:
        """Train and evaluate the agent with given parameters."""
        n_episodes = self.config.training.num_episodes
        eval_episodes = self.config.training.num_eval_episodes

        # Training phase
        train_rewards = []
        eval_metrics = []

        for episode in range(n_episodes):
            # Training episode
            episode_reward = self._run_episode(env, agent, training=True)
            train_rewards.append(episode_reward)

            # Periodic evaluation
            if episode % self.config.training.eval_frequency == 0:
                eval_metrics.append(self._evaluate_agent(env, agent, eval_episodes))

                # Report intermediate values
                trial.report(eval_metrics[-1]['win_rate'], step=episode)

                if trial.should_prune():
                    raise optuna.TrialPruned()

        # Final evaluation
        final_metrics = self._evaluate_agent(env, agent, eval_episodes)

        return {
            'win_rate': final_metrics['win_rate'],
            'average_reward': final_metrics['average_reward'],
            'training_stability': np.std(train_rewards),
            'final_epsilon': agent.epsilon
        }

    def optimize(self, n_trials: int = 100) -> Dict[str, Any]:
        """Run the optimization process."""
        self.logger.info(f"Starting optimization with {n_trials} trials")

        # Create study callback for live plotting
        plot_callback = partial(self._plot_optimization_history, log_dir=self.log_dir)

        try:
            self.study.optimize(
                self.objective,
                n_trials=n_trials,
                callbacks=[self._log_callback, plot_callback]
            )

            # Save and analyze results
            best_params = self.study.best_params
            self._save_study_results()
            self._analyze_study()

            return best_params

        except KeyboardInterrupt:
            self.logger.info("Optimization interrupted by user")
            self._save_study_results()
            return self.study.best_params

    def _analyze_study(self) -> None:
        """Analyze and visualize study results."""
        # Create importance plot
        fig = optuna.visualization.plot_param_importances(self.study)
        fig.write_image(str(self.log_dir / "param_importance.png"))

        # Create correlation plot
        fig = optuna.visualization.plot_parallel_coordinate(self.study)
        fig.write_image(str(self.log_dir / "param_correlation.png"))

        # Analyze parameter distributions
        param_history = pd.DataFrame([
            {**t.params, 'value': t.value}
            for t in self.study.trials if t.state == TrialState.COMPLETE
        ])

        self._plot_parameter_distributions(param_history)

    def _plot_parameter_distributions(self, param_history: pd.DataFrame) -> None:
        """Plot distributions of parameters for completed trials."""
        numeric_params = param_history.select_dtypes(include=[np.number]).columns

        fig, axes = plt.subplots(
            nrows=len(numeric_params),
            ncols=1,
            figsize=(10, 4 * len(numeric_params))
        )

        for ax, param in zip(axes, numeric_params):
            sns.histplot(data=param_history, x=param, ax=ax)
            ax.set_title(f'Distribution of {param}')

        plt.tight_layout()
        plt.savefig(self.log_dir / "parameter_distributions.png")
        plt.close()

    def _save_study_results(self) -> None:
        """Save study results to file."""
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
                    'state': str(t.state)
                }
                for t in self.study.trials
            ]
        }

        results_file = self.log_dir / f'results_{timestamp}.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

    @staticmethod
    def _log_callback(study: optuna.Study, trial: optuna.Trial) -> None:
        """Callback for logging trial results."""
        if trial.value is not None:
            logging.info(f"Trial {trial.number}")
            logging.info(f"Value: {trial.value}")
            logging.info(f"Params: {trial.params}")

    @staticmethod
    def _plot_optimization_history(study: optuna.Study, trial: optuna.Trial, log_dir: Path) -> None:
        """Plot optimization history after each trial."""
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(str(log_dir / "optimization_history.png"))


def main():
    """Main function for running hyperparameter optimization."""
    from config import get_default_config

    config = get_default_config()

    def env_creator():
        from env import SpanishCardGameEnv
        return SpanishCardGameEnv(config)

    optimizer = HyperparameterOptimizer(config, env_creator)
    best_params = optimizer.optimize(n_trials=50)

    print("\nOptimization completed!")
    print("Best parameters found:")
    print(json.dumps(best_params, indent=2))


if __name__ == "__main__":
    main()
