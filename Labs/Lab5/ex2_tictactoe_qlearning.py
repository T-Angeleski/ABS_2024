import random
import time

import numpy as np
from pettingzoo.classic import tictactoe_v3

# Globals
Q_VALUES_1 = {}
Q_VALUES_2 = {}
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.97
MIN_EPSILON = 0.3  # Namerno malce pogolemo, da ostane toj random faktor
NUM_EPISODES_TRAIN = 999
NUM_EPISODES_TEST = 50
RENDER_EPISODE_DELTA = 500


def state_to_tuple(state):
	return tuple(state.flatten())


def get_random_action(action_mask):
	return random.choice(np.flatnonzero(action_mask))


def get_best_action(q_table, state, action_mask):
	best_q = -float('inf')
	best_action = None
	for action, valid in enumerate(action_mask):
		if valid:
			q_value = q_table.get((state, action), 0.0)
			if q_value > best_q:
				best_q = q_value
				best_action = action
	return best_action


def train_agents(episodes=NUM_EPISODES_TRAIN):
	global EPSILON
	for episode in range(episodes):
		print(f"Episode: {episode + 1}, epsilon: {EPSILON}")
		env = tictactoe_v3.env(
			render_mode='human' if episode % RENDER_EPISODE_DELTA == 0 else None
		)  # Only render every Nth ep
		env.reset()
		
		prev_data = {}
		for agent in env.agent_iter():
			state, reward, done, truncated, _ = env.last()
			
			if agent in prev_data:
				action_mask = state['action_mask']
				available_actions = [action for action, valid in enumerate(action_mask) if valid]
				
				max_future_q = 0
				if available_actions:
					if agent == 'player_1':
						max_future_q = max(
							Q_VALUES_1.get((state_to_tuple(state['observation']), action), 0.0) for action in
							available_actions)
					else:
						max_future_q = max(
							Q_VALUES_2.get((state_to_tuple(state['observation']), action), 0.0) for action in
							available_actions)
				
				prev_state, prev_action = prev_data[agent]
				if agent == "player_1":
					current_q = Q_VALUES_1.get((prev_state, prev_action), 0.0)
					Q_VALUES_1[(prev_state, prev_action)] = current_q + LEARNING_RATE * (
							reward + DISCOUNT_FACTOR * max_future_q - current_q)
				else:
					current_q = Q_VALUES_2.get((prev_state, prev_action), 0.0)
					Q_VALUES_2[(prev_state, prev_action)] = current_q + LEARNING_RATE * (
							reward + DISCOUNT_FACTOR * max_future_q - current_q)
			
			if done or truncated:
				selected_action = None
			else:
				action_mask = state['action_mask']
				if random.random() < EPSILON:
					selected_action = get_random_action(action_mask)
				else:
					if agent == "player_1":
						selected_action = get_best_action(Q_VALUES_1, state_to_tuple(state['observation']), action_mask)
					else:
						selected_action = get_best_action(Q_VALUES_2, state_to_tuple(state['observation']), action_mask)
			
			prev_data[agent] = (state_to_tuple(state['observation']), selected_action)
			env.step(selected_action)
		EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)


def test(iterations=NUM_EPISODES_TEST):
	total_rewards = {'player_1': 0, "player_2": 0}
	victories = {'player_1': 0, "player_2": 0}
	print("Testing...")
	for iter in range(iterations):
		env = tictactoe_v3.env(render_mode=None if iter < 47 else 'human')
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
				
				action_mask = state['action_mask']
				
				if agent == 'player_2':
					action = get_best_action(Q_VALUES_2, state_to_tuple(state['observation']), action_mask)
				else:
					action = get_best_action(Q_VALUES_1, state_to_tuple(state['observation']), action_mask)
				env.step(action)
				total_rewards[agent] += reward
		print(f"Game {iter + 1} done, total rewards: {total_rewards}, victories: {victories}")
	
	avg_rewards = {agent: total / iterations for agent, total in total_rewards.items()}
	return avg_rewards, victories


# TRAIN AND TEST
train_agents(NUM_EPISODES_TRAIN)

averages, wins = test(NUM_EPISODES_TEST)
print(f"Average rewards: {averages}, Wins: {wins}")

print(f"Najcesto zavrsuva igrata vo Neresheno")