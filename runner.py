import itertools
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dqn_config import Config, get_default_config
from env import SpanishCardGameEnv
from dqn import DQNAgent
import trainer
import logging
from datetime import datetime


class ExperimentRunner:
    def __init__(self, base_config: Config, experiment_dir: str = "experiments"):
        self.base_config = base_config
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.logger = self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.experiment_dir / "experiments.log"),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

    def run_experiment_grid(self, param_grid: dict, runs_per_config: int = 3):
        """Run experiments for all combinations in parameter grid."""
        # Generate all combinations
        keys = param_grid.keys()
        values = param_grid.values()
        configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

        results = []
        for config_idx, param_config in enumerate(configs):
            self.logger.info(f"Testing configuration {config_idx + 1}/{len(configs)}")
            self.logger.info(f"Parameters: {param_config}")

            # Run multiple times for statistical significance
            config_results = []
            for run in range(runs_per_config):
                # Create new config with these parameters
                config = get_default_config()
                config.dqn.update(param_config)

                # Run training
                env = SpanishCardGameEnv(config)
                agent = DQNAgent(config)
                metrics = trainer.train_agent(env, agent, config)

                config_results.append(metrics)

            # Average results
            avg_results = {
                'params': param_config,
                'avg_win_rate': np.mean([r['win_rate'] for r in config_results]),
                'std_win_rate': np.std([r['win_rate'] for r in config_results]),
                'avg_reward': np.mean([r['average_reward'] for r in config_results]),
                'std_reward': np.std([r['average_reward'] for r in config_results])
            }
            results.append(avg_results)

        self._save_and_plot_results(results)
        return results

    def _save_and_plot_results(self, results):
        """Save and visualize experiment results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save raw results
        with open(self.experiment_dir / f'results_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=2)

        # Convert to DataFrame for analysis
        df = pd.DataFrame(results)

        # Create visualizations
        self._plot_parameter_effects(df, timestamp)
        self._plot_correlation_matrix(df, timestamp)

    def _plot_parameter_effects(self, df, timestamp):
        """Plot the effect of each parameter on performance."""
        param_cols = [col for col in df.columns if col.startswith('params.')]

        fig, axes = plt.subplots(
            nrows=len(param_cols),
            ncols=1,
            figsize=(10, 5 * len(param_cols))
        )

        for ax, param in zip(axes, param_cols):
            sns.boxplot(data=df, x=param, y='avg_win_rate', ax=ax)
            ax.set_title(f'Effect of {param} on Win Rate')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(self.experiment_dir / f'parameter_effects_{timestamp}.png')
        plt.close()

    def _plot_correlation_matrix(self, df, timestamp):
        """Plot correlation matrix of parameters and metrics."""
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()

        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Parameter and Metric Correlations')
        plt.tight_layout()
        plt.savefig(self.experiment_dir / f'correlation_matrix_{timestamp}.png')
        plt.close()


def main():
    # Example parameter grid
    param_grid = {
        'hidden_size_1': [64, 128, 256],
        'learning_rate': [1e-4, 1e-3],
        'batch_size': [32, 64],
        'use_double_dqn': [True, False]
    }

    config = get_default_config()
    runner = ExperimentRunner(config)
    results = runner.run_experiment_grid(param_grid, runs_per_config=3)

    print("Experiment completed! Check the experiments directory for results.")


if __name__ == "__main__":
    main()
