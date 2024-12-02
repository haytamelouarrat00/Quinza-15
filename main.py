from env import SpanishCardGameEnv
from dqn import DQNAgent
import trainer

# Initialize environment and agent
env = SpanishCardGameEnv(num_players=4)
state_size = 83  # 40 (hand) + 40 (table) + 3 (other info)
action_size = 3  # Maximum cards in hand
agent = DQNAgent(state_size, action_size)

# Train the agent
trainer.train_agent(env, agent, num_episodes=1000)
