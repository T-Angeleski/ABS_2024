import gymnasium as gym
from keras import Sequential, Input
from keras.src.layers import Dense
from keras.src.losses import mean_squared_error
from keras.src.optimizers import Adam

from AudCodes.deep_q_learning import DDQN


def build_model(state_space_shape, num_actions):
    model = Sequential([Input(shape=state_space_shape),
                        Dense(8, activation="relu"),
                        Dense(16, activation="relu"),
                        Dense(32, activation="relu"),
                        Dense(16, activation="relu"),
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
    BATCH_SIZE = 32
    MEMORY_SIZE = 512

    # ----------------- TRUE/FALSE WHETHER TO TRAIN OR NOT  -------------------------
    isTraining = str.lower(input("Training or testing? (y/n): ")) == 'y'

    model = build_model(STATE_SPACE_SHAPE, NUM_ACTIONS)
    target_model = build_model(STATE_SPACE_SHAPE, NUM_ACTIONS)

    agent = DDQN(STATE_SPACE_SHAPE, NUM_ACTIONS, model, target_model, batch_size=BATCH_SIZE, memory_size=MEMORY_SIZE)
    total_rewards = []

    # TODO Future: Try out epsilon decay, using start and end values and multiplying (* 0.995)

    if isTraining:
        for episode in range(NUM_EPISODES):
            state, _ = env.reset()
            rewards, steps = 0, 0
            done = False

            while not done:
                action = agent.get_action(state, 0)
                new_state, reward, done, _, _ = env.step(action)

                # env.render()

                agent.update_memory(state, action, reward, new_state, done)
                state = new_state
                rewards += reward
                steps += 1

                if steps >= MAX_STEPS:
                    break

            print("Episode: ", episode + 1, " Total Reward: ", rewards)
            total_rewards.append(rewards)

            agent.train()
            if episode % 5 == 0:
                agent.update_target_model()

        print("Training done")
        agent.save("mountain_car_ex2", f"{NUM_EPISODES}.weights")
        print(f"Rewards: {total_rewards}, average reward: {sum(total_rewards) / len(total_rewards)}")
    else:
        agent.load("mountain_car_ex2", f"{NUM_EPISODES}.weights")
        env = gym.make("MountainCar-v0", render_mode="human")
        done = False
        state, _ = env.reset()
        env.render()

        steps = 0
        while not done:
            action = agent.get_action(state, 0)
            new_state, _, done, _, _ = env.step(action)
            state = new_state
            env.render()

            steps += 1
            if steps >= 500:
                env.close()
                break
