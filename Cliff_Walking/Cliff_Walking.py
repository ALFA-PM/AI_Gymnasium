import gymnasium as gym
import numpy as np
import random

def q_learning_epsilon_constant(it_max, epsilon, learning_rate, discount_factor, render=False):
    env = gym.make('CliffWalking-v0', render_mode="human" if render else None)
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    q_table = np.zeros((num_states, num_actions))
    rewards_table = np.zeros(it_max)
    optimal_episode = None

    for episode in range(it_max):
        terminated = False
        truncated = False
        total_rewards = 0
        observation, info = env.reset()

        while not terminated and not truncated:
            if render:
                env.render()

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[observation])
            
            next_observation, reward, terminated, truncated, info = env.step(action)
            total_rewards += reward

            # Q-learning update
            diff = reward + discount_factor * np.max(q_table[next_observation]) - q_table[observation, action]
            q_table[observation][action] += learning_rate * diff

            observation = next_observation

        rewards_table[episode] = total_rewards

        if total_rewards == -13 and optimal_episode is None:
            optimal_episode = episode

        if render:
            env.render()

    env.close()
    return q_table, rewards_table, optimal_episode

# Define the parameters
it_max = 300
epsilon = 0.1
learning_rate = 0.9
discount_factor = 0.9

# Perform Q-learning with rendering
q_table, rewards_table, optimal_episode = q_learning_epsilon_constant(it_max, epsilon, learning_rate, discount_factor, render=True)

# To show the `percent` of the rewards and success we can write `"rgb_array"` instead of the `"human"` in `render_mode`!
