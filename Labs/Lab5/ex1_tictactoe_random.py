import random
import time

import numpy as np
from pettingzoo.classic import tictactoe_v3

# Auditoriski kodovi za q_learning nekompatibilni so ovaa igra
# Globals
Q_VALUES = {}
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.98
MIN_EPSILON = 0.01
NUM_EPISODES_TRAIN = 999
NUM_EPISODES_TEST = 50
RENDER_EPISODE_DELTA = 500


def get_random_action(action_mask):
	return random.choice(np.flatnonzero(action_mask))


def get_best_action(state, action_mask):
	available_actions = np.flatnonzero(action_mask)
	if state in Q_VALUES:
		q_values_state = Q_VALUES[state]
		best_action = available_actions[np.argmax([q_values_state.get(action, 0)
		                                           for action in available_actions])]
		return best_action
	return random.choice(available_actions)


def state_to_tuple(state):
	return tuple(state.flatten())


def train(episodes=NUM_EPISODES_TRAIN):
	global EPSILON
	for episode in range(episodes):
		print(f"Episode: {episode + 1}, epsilon: {EPSILON}")
		env = tictactoe_v3.env(
			render_mode='human' if episode % RENDER_EPISODE_DELTA == 0 else None
		)  # Only render every Nth ep
		
		env.reset()
		
		done = False
		prev_data = {}
		while not done:
			for agent in env.agent_iter():
				state, reward, done, truncated, _ = env.last()
				
				if reward != 0: print(f"{agent} received reward: {reward}")
				
				if done or truncated: break
				
				state_tuple = state_to_tuple(state["observation"])
				action_mask = state["action_mask"]
				
				if agent in prev_data:
					prev_state, prev_action = prev_data[agent]
					max_future_q = max(Q_VALUES.get(state_tuple, {}).values(), default=0)
					current_q = Q_VALUES.setdefault(prev_state, {}).get(prev_action, 0)
					Q_VALUES[prev_state][prev_action] = current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR *
					                                                                 max_future_q - current_q)
				
				if random.random() < EPSILON:
					action = get_random_action(action_mask)
				else:
					action = get_best_action(state_tuple, action_mask)
				
				prev_data[agent] = (state_tuple, action)
				env.step(action)
		
		EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)


def test(iterations=NUM_EPISODES_TEST):
	total_rewards = {'player_1': 0, "player_2": 0}
	victories = {'player_1': 0, "player_2": 0}
	
	for iter in range(iterations):
		env = tictactoe_v3.env(render_mode=None)
		env.reset()
		
		done = False
		while not done:
			for agent in env.agent_iter():
				state, reward, done, truncated, _ = env.last()
				if done or truncated:
					total_rewards[agent] += reward
					
					if reward == -1:  # reward of -1 means loss
						if agent == 'player_1':
							victories['player_2'] += 1
						else:
							victories['player_1'] += 1
					break
				
				state_tuple = state_to_tuple(state['observation'])
				action_mask = state['action_mask']
				
				if agent == 'player_2':
					action = get_random_action(action_mask)
				else:
					action = get_best_action(state_tuple, action_mask)
				
				env.step(action)
				total_rewards[agent] += reward
	
	avg_rewards = {agent: total / iterations for agent, total in total_rewards.items()}
	return avg_rewards, victories


# TRAIN AND TEST
train()

averages, wins = test(NUM_EPISODES_TEST)
print(f"Average rewards: {averages}, Wins: {wins}")