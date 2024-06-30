from AudCodes.mdp import *
import numpy as np
import gymnasium as gym

factors = [0.5, 0.7, 0.9]
iterations = [50, 100]
best_factor = None
best_iter_count = None
best_avg_reward = float("-inf")
# Premnogu vreme trebase so policy iteration za Taxi, zatoa smeniv na FrozenLake

# open("results_value.txt", "w")
with open("results_policy.txt", "w") as file:
	for discount_factor in factors:
		for iter in iterations:
			env = gym.make("FrozenLake-v1",
			               render_mode="none")  # Off during training
			
			steps_per_iteration = []
			reward_per_iteration = []
			
			for i in range(iter):
				state, _ = env.reset()
				done = False
				steps, total_reward = 0, 0
				policy, value = policy_iteration(env, env.action_space.n,
				                                env.observation_space.n,
				                                discount_factor=discount_factor)
				
				while not done:
					action = np.argmax(policy[state])
					state, reward, done, _, _ = env.step(action)
					steps += 1
					total_reward += reward
				
				steps_per_iteration.append(steps)
				reward_per_iteration.append(total_reward)
			
			avg_steps = np.mean(steps_per_iteration)
			avg_reward = np.mean(reward_per_iteration)
			
			file.write(
				f"Steps: {steps_per_iteration}, Rewards: "
				f"{reward_per_iteration}\n"
			)
			file.write(
				f"Avg steps for {iter} episodes, discount {discount_factor}: {avg_steps}\n")
			file.write(
				f"Avg reward for {iter} episodes, discount "
				f"{discount_factor}: {avg_reward}\n")
			
			if avg_reward > best_avg_reward:
				best_avg_reward = avg_reward
				best_factor = discount_factor
				best_iter_count = iter
	
	file.write(f"Best combination - Discount factor: {best_factor}, Episode "
	           f"amount: "
	           f"{best_iter_count}, "
	           f"Avg reward: {best_avg_reward}\n")

# Renderiraj najdobra
env = gym.make("FrozenLake-v1", render_mode="human")
state, _ = env.reset()
done = False

policy, _ = policy_iteration(env, env.action_space.n,
                            env.observation_space.n,
                            discount_factor=best_factor)
while not done:
	action = np.argmax(policy[state])
	state, _, done, _, _ = env.step(action)
	env.render()
env.close()