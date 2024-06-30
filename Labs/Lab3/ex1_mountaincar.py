import gymnasium as gym
import numpy as np

from AudCodes.deep_q_learning import DQN
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adam


def build_model(state_space_shape, num_actions):
    model = Sequential([Input(shape=state_space_shape),
                        Dense(32, activation="relu"),
                        Dense(32, activation="relu"),
                        Dense(num_actions, activation="linear")])

    model.compile(Adam(learning_rate=0.001), mean_squared_error)

    return model


if __name__ == '__main__':
    env = gym.make("MountainCar-v0")
    env.reset()

    STATE_SPACE_SHAPE = env.observation_space.shape
    NUM_ACTIONS = env.action_space.n
    NUM_EPISODES = 2
    MAX_STEPS = 150
    BATCH_SIZE = 128
    MEMORY_SIZE = 1024

    # ----------------- TRUE/FALSE WHETHER TO TRAIN OR NOT  -------------------------
    isTraining = str.lower(input("Training or testing? (y/n): ")) == 'y'

    model = build_model(STATE_SPACE_SHAPE, NUM_ACTIONS)
    target_model = build_model(STATE_SPACE_SHAPE, NUM_ACTIONS)

    agent = DQN(STATE_SPACE_SHAPE, NUM_ACTIONS, model, target_model, batch_size=BATCH_SIZE, memory_size=MEMORY_SIZE)
    total_rewards = []

    if isTraining:
        for episode in range(NUM_EPISODES):
            state, _ = env.reset()
            rewards, steps = 0, 0
            done = False

            while not done:
                action = agent.get_action(state, 0.1)
                new_state, reward, done, _, _ = env.step(action)

                # env.render()

                agent.update_memory(state, action, reward, new_state, done)
                state = new_state
                rewards += reward
                steps += 1

                if steps >= MAX_STEPS:
                    # print("Max steps reached, did not reach goal on episode ", episode + 1)
                    break

            print(f"Episode {episode + 1} done")
            print(f"Episode reward is {rewards}")
            total_rewards.append(rewards)

            agent.train()
            agent.update_target_model()

        agent.save("mountain_car", f"{NUM_EPISODES}.weights")
        print("Training done. Total rewards:")
        print(total_rewards)
        print(f"Average reward across {NUM_EPISODES} is {np.mean(total_rewards)}")
    else:
        agent.load("mountain_car", f"{NUM_EPISODES}.weights")
        env = gym.make("MountainCar-v0", render_mode="human")
        done = False
        state, _ = env.reset()
        env.render()
        steps = 0
        while not done:
            action = agent.get_action(state, 0)
            new_state, reward, done, _, _ = env.step(action)

            env.render()
            state = new_state
            steps += 1
            if steps >= 500:
                env.close()
                break
