import gymnasium as gym

from AudCodes.q_learning import *

discount_factors = [0.5, 0.9]
learning_rates = [0.1, 0.01]
episode_amounts = [15, 25, 50]
steps_per_episode = [50, 100]  # iteracii
epsilon = 0.5
env = gym.make("Taxi-v3", render_mode="ansi")
num_states = env.observation_space.n
num_actions = env.action_space.n
steps_list = []
rewards_list = []
# Initialize q table
q_table = random_q_table(-2, 0,
                         (num_states, num_actions))
# change file name zavisno dali se korisni epsilon greedy
with open("taxi_best_action_with_epsilon.txt", "w") as file:
	for amount in episode_amounts:
		for rate in learning_rates:
			for factor in discount_factors:
				for steps in steps_per_episode:
					
					file.write(
						f"Episode amount: {amount}, learning rate: {rate}, "
						f"discount factor: {factor}, steps per episode: "
						f"{steps}\n")
					total_steps = 0
					total_reward = 0
					for episode in range(amount):
						
						state, _ = env.reset()
						episode_steps = 0
						episode_reward = 0
						for s in range(steps):
							# Uncomment taa akcija po potreba

							# Sekoj cekor najdobra akcija
							# action = get_best_action(q_table, state)
							# Epsilon greedy
							action = get_action(env, q_table, state, epsilon)
							# Slucaen izbor akcija
							# action = get_random_action(env)

							new_state, reward, done, _, _ = env.step(action)
							episode_reward += reward
							episode_steps += 1
							if done:
								break
							env.render()
							new_q = calculate_new_q_value(q_table, state,
							                              new_state,
							                              action, reward, rate,
							                              factor)
							q_table[state, action] = new_q
							state = new_state
						
						total_steps += episode_steps / amount
						total_reward += episode_reward
					# file.write(f"Episode ended in {episode_steps} steps, "
					#       f"total reward is {episode_reward}")
					file.write(
						f"Avg steps: {round(total_steps, 2)}, total reward:"
						f" {total_reward}\n")
					steps_list.append(round(total_steps, 2))
					rewards_list.append(total_reward)
	env.close()
	avg_steps = np.mean(steps_list)
	avg_reward = np.mean(rewards_list)
	file.write(
		f"Average steps across all episodes: {avg_steps}, average reward:"
		f" {avg_reward}\n")
	file.write(f"All steps: {steps_list}\n")
	file.write(f"All rewards: {rewards_list}\n")
print("Finished, wrote in text file")
# Renderiraj najdobra
env = gym.make("Taxi-v3", render_mode="human")
state, _ = env.reset()
for step in range(50):
	action = get_best_action(q_table, state)
	new_state, reward, done, _, _ = env.step(action)
	if done:
		break
	env.render()
env.close()