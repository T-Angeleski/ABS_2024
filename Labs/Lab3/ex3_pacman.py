import gymnasium as gym
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D
from tensorflow.keras.optimizers import Adam
from PIL import Image

from AudCodes.deep_q_learning import DQN, DuelingDQN

import matplotlib.pyplot as plt


def build_model(state_space_shape, num_actions, learning_rate):
    model = Sequential([
        Conv2D(16, (8, 8), strides=(4, 4), activation='relu', input_shape=state_space_shape),
        Conv2D(32, (4, 4), strides=(2, 2), activation='relu'),
        Conv2D(32, (3, 3), strides=(1, 1), activation='relu'),
        Conv2D(64, (2, 2), strides=(1, 1), activation='relu'),
        Flatten(),
        Dense(8, activation='relu'),
        Dense(16, activation='relu'),
        Dense(num_actions, activation='linear')
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mean_squared_error")

    return model


def preprocess_frame(state):
    img = Image.fromarray(state).convert("L")
    grayscale = np.array(img, dtype=np.float32)
    grayscale /= 255
    return grayscale


def clip_reward(reward):
    return np.clip(reward, -500., 500.)


def process_output(state, reward):
    new_state = preprocess_frame(state)
    new_reward = clip_reward(reward)
    return new_state, new_reward


if __name__ == '__main__':
    is_training = str.lower(input("Training or testing? (y/n): "))
    env = gym.make('ALE/MsPacman-v5')

    STATE_SPACE_SHAPE = env.observation_space.shape[:-1] + (1,)  # Slice the color value and add 1
    NUM_ACTIONS = env.action_space.n
    NUM_EPISODES = 100
    MAX_STEPS = 5000
    LEARNING_RATE = 0.005
    DISCOUNT_FACTOR = 1.0
    epsilon, EPSILON_DECAY, MIN_EPSILON = 0.99, 0.995, 0.1
    BATCH_SIZE = 32
    MEMORY_SIZE = 2048

    model = build_model(STATE_SPACE_SHAPE, NUM_ACTIONS, LEARNING_RATE)
    target_model = build_model(STATE_SPACE_SHAPE, NUM_ACTIONS, LEARNING_RATE)

    agent_type = str.lower(input("Which agent do you want to use? (DQN - y / DuelingDQN - n): "))
    if agent_type == 'y':
        agent = DQN(STATE_SPACE_SHAPE, NUM_ACTIONS, model, target_model,
                    batch_size=BATCH_SIZE, memory_size=MEMORY_SIZE)
    else:
        layers = [
            Conv2D(16, (8, 8), strides=(4, 4), activation='relu'),
            Conv2D(32, (4, 4), strides=(2, 2), activation='relu'),
            Conv2D(32, (3, 3), strides=(1, 1), activation='relu'),
            Conv2D(64, (2, 2), strides=(1, 1), activation='relu'),
            Flatten()
        ]
        agent = DuelingDQN(STATE_SPACE_SHAPE, NUM_ACTIONS,
                           LEARNING_RATE, DISCOUNT_FACTOR, BATCH_SIZE, MEMORY_SIZE)
        agent.build_model(layers)

    # -------------------------------------------------------
    if is_training == 'y':
        all_rewards, epsilon_over_time, actions_taken = [], [], []

        # show_img = str.lower(input("Do you want to show grayscale images? (y/n): "))
        for episode in range(NUM_EPISODES):
            state, _ = env.reset()
            state = preprocess_frame(state)
            done = False
            episode_reward = 0
            steps = 0
            while not done:
                action = agent.get_action(state, epsilon)

                new_state, reward, done, _, _ = env.step(action)
                new_state, reward = process_output(new_state, reward)

                # Testing showing how the agent sees the grayscale image
                # if steps % 100 == 0 and show_img == 'y':
                #     plt.imshow(new_state, cmap='gray')
                #     plt.show()

                agent.update_memory(state, action, reward, new_state, done)

                state = new_state
                episode_reward += reward
                steps += 1
                if steps >= MAX_STEPS:
                    print(f"Max steps reached for episode: {episode + 1}")
                    break

                actions_taken.append(action)

            agent.train()
            print(f"Reward for episode {episode + 1}: {episode_reward}")
            all_rewards.append(episode_reward)

            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
            epsilon_over_time.append(epsilon)
            if episode % 10 == 0:
                agent.update_target_model()

        agent.save("pacman", f"{NUM_EPISODES}.weights")
        print("Training done")
        print(f"All rewards per episode: {all_rewards}\n")
        print(f"Average rewards are: {sum(all_rewards) / len(all_rewards)}")

        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        axs[0, 0].plot(all_rewards)
        axs[0, 0].set_title('Rewards over episodes')
        axs[0, 0].set_xlabel('Episode')
        axs[0, 0].set_ylabel('Reward')

        episode_numbers = list(range(1, len(epsilon_over_time) + 1))
        axs[0, 1].plot(episode_numbers, epsilon_over_time)
        axs[0, 1].set_title('Epsilon values over episodes')
        axs[0, 1].set_xlabel('Episode')
        axs[0, 1].set_ylabel('Epsilon')

        axs[1, 0].hist(actions_taken, bins=range(NUM_ACTIONS + 1), align='left', rwidth=0.8)
        axs[1, 0].set_title('Actions taken')
        axs[1, 0].set_xlabel('Action')
        axs[1, 0].set_ylabel('Frequency')

        # Remove the unused subplot
        fig.delaxes(axs[1, 1])

        plt.tight_layout()
        plt.show()
    else:
        agent.load("pacman", f"{NUM_EPISODES}.weights")
        env = gym.make('ALE/MsPacman-v5', render_mode="human")
        state, _ = env.reset()
        state = preprocess_frame(state)
        rewards = 0
        done = False
        while not done:
            action = agent.get_action(state, MIN_EPSILON)
            state, reward, done, _, _ = env.step(action)
            state, reward = process_output(state, reward)
            env.render()
            rewards += reward

        print(f"Total rewards: {rewards}")
