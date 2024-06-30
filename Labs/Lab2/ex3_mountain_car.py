import gymnasium as gym
import numpy as np
from AudCodes.q_learning import *


def get_discrete_state(state, low_value, window_size):
    new_state = (state - low_value) / window_size
    return tuple(new_state.astype(int))


if __name__ == '__main__':
    env = gym.make("MountainCar-v0", render_mode="human")

    # Initializing
    NUM_ACTIONS = env.action_space.n
    observation_space_size = [7, 7]  # 7 seems to converge fast
    observation_space_low_value = env.observation_space.low
    observation_space_high_value = env.observation_space.high
    observation_window_size = (observation_space_high_value -
                               observation_space_low_value) / observation_space_size
    q_table = random_q_table(-1, 0, (observation_space_size + [NUM_ACTIONS]))

    epsilon = 0.1
    EPSILON_MIN = 0.05
    DECAY = 0.025
    EPISODES = [3]  # Mnogu sporo trenira, ne staviv povekje epizodi
    LEARNING_RATE = 0.1
    DISCOUNT_FACTOR = 0.9  # 0.9 0.1
    info_to_write = ""  # Za rezultati file text
    steps_episode, rewards_episode = [], []
    for eps in EPISODES:
        for episode in range(eps):
            state, _ = env.reset()
            discrete_state = get_discrete_state(state,
                                                observation_space_low_value,
                                                observation_window_size)
            done = False
            episode_steps, total_reward = 0, 0
            while not done:
                action = get_action(env, q_table, discrete_state, epsilon)
                new_state, reward, done, _, _ = env.step(action)
                new_discrete_state = get_discrete_state(new_state,
                                                        observation_space_low_value,
                                                        observation_window_size)

                new_q = calculate_new_q_value(q_table, discrete_state,
                                              new_discrete_state, action,
                                              reward, LEARNING_RATE,
                                              DISCOUNT_FACTOR)
                q_table[discrete_state, action] = new_q
                discrete_state = new_discrete_state
                episode_steps += 1
                total_reward += reward
            print("Episode done")
            info_to_write += f"Episode {episode + 1} finished, steps: " \
                             f"{episode_steps}, reward: {total_reward}, " \
                             f"epsilon: {epsilon}\n"
            steps_episode.append(episode_steps)
            rewards_episode.append(total_reward)
            if epsilon > EPSILON_MIN:
                epsilon -= DECAY
# mountain_car_epsilon.txt, mountain_car_no_decay.txt
with open("mountain_car_epsilon.txt", "w") as f:
    f.write(info_to_write)
    f.write(f"Average steps per episode: {np.mean(steps_episode)}\n")
    f.write(f"Average reward: {np.mean(rewards_episode)}")
print("done")
