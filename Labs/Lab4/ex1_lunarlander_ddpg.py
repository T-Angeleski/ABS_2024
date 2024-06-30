from AudCodes.deep_q_learning import DDPG, OrnsteinUhlenbeckActionNoise
import gymnasium as gym
import numpy as np

def clip_reward(reward):
    return np.clip(reward, -10., 10.)


if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2')
    env.reset()

    STATE_SPACE_SHAPE = env.observation_space.shape[0]
    ACTION_SPACE_SHAPE = env.action_space.shape[0]

    NUM_EPISODES = 2
    LEARNING_RATE_ACTOR = 0.01
    LEARNING_RATE_CRITIC = 0.02
    DISCOUNT_FACTOR = 0.99
    BATCH_SIZE = 128
    MEMORY_SIZE = 2048

    noise = OrnsteinUhlenbeckActionNoise(ACTION_SPACE_SHAPE)
    agent = DDPG(STATE_SPACE_SHAPE, ACTION_SPACE_SHAPE, LEARNING_RATE_ACTOR, LEARNING_RATE_CRITIC, DISCOUNT_FACTOR,
                 BATCH_SIZE, MEMORY_SIZE)

    agent.build_model()

    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        done = False
        env.render()

        while not done:
            action = agent.get_action(state, discrete=False) + noise()
            new_state, reward, done, _, _ = env.step(action)
            env.render()
            clipped_reward = clip_reward(reward)
            print(f"Pre: {reward}, Post: {clipped_reward}")

            numeric_done = ...
            if done:
                numeric_done = 1
            else:
                numeric_done = 0

            agent.update_memory(state, action, clipped_reward, new_state, numeric_done)
            state = new_state

        agent.train()

        if episode % 3 == 0:
            agent.update_target_model()

    agent.save("lunarlander", NUM_EPISODES)