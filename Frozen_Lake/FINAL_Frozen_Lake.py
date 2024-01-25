import numpy as np
import gym
from tqdm import tqdm


def epsilon_greedy_policy(Q_table, state, epsilon):
    """
    Selects an action using epsilon-greedy policy

    Args:
    Q_table (numpy.ndarray): Q-value table
    state (int): Current state
    epsilon (float): Epsilon value for exploration-exploitation trade-off

    Returns:
    int: Selected action
    """
    # Determine the number of actions available in the current state
    num_actions = len(Q_table[state])

    # Decide whether to explore or exploit
    if np.random.uniform(0, 1) < epsilon:
        # Explore: choose a random action
        return np.random.choice(num_actions)
    else:
        # Exploit: choose the best action based on current Q-values
        return np.argmax(Q_table[state])
        

def q_learning(environment, episodes, alpha=0.2, gamma=0.99, epsilon=0.1):
    Q_table = np.zeros((environment.observation_space.n, environment.action_space.n))
    pbar = tqdm(total=episodes, dynamic_ncols=True)

    for episode in range(episodes):
        state, _ = environment.reset()
        done = False
        episode_reward = 0

        while not done:
            action = epsilon_greedy_policy(Q_table, state, epsilon)
            next_state, reward, done, _, _ = environment.step(action)
            next_action = np.argmax(Q_table[next_state, :])

            td_target = reward + gamma * Q_table[next_state, next_action]
            Q_table[state, action] += alpha * (td_target - Q_table[state, action])

            state = next_state
            episode_reward += reward

        pbar.update(1)

        # Evaluate policy less frequently to save computation
        if episode % 1000 == 0 or episode == episodes - 1:
            avg_reward = evaluate_policy(environment, Q_table, 100)
            pbar.set_description(f"Episode: {episode} - Average Reward: {avg_reward:.2f}")

    pbar.close()
    return Q_table


def evaluate_policy(environment, Q_table, episodes):
    """
    Evaluate the performance of a given policy over a certain number of episodes

    Args:
    environment: The environment to test the policy in
    Q_table (numpy.ndarray): The Q-table representing the policy
    episodes (int): Number of episodes to run the evaluation for

    Returns:
    float: The average total reward per episode
    """
    total_reward = 0
    # Extract the policy from the Q-table
    optimal_policy = np.argmax(Q_table, axis=1)

    for _ in range(episodes):
        state, _ = environment.reset()
        done = False
        episode_reward = 0

        while not done:
            action = optimal_policy[state]
            state, reward, done, _, _ = environment.step(action)
            episode_reward += reward

        total_reward += episode_reward

    # Calculate the average reward per episode
    average_reward = total_reward / episodes
    return average_reward


def demo_agent(environment, Q_table, episodes=1, render=True):
    """
    Demonstrates the behavior of an agent in a given environment using the policy derived from the Q-table

    Args:
    environment: The environment in which to demonstrate the agent
    Q_table (numpy.ndarray): The Q-table used to derive the policy
    episodes (int): Number of episodes to demonstrate
    render (bool): If True, the environment will be rendered during the demonstration
    """
    optimal_policy = np.argmax(Q_table, axis=1)

    for episode in range(episodes):
        state, _ = environment.reset()
        done = False
        print("\nEpisode:", episode + 1)

        while not done:
            if render:
                environment.render()
            action = optimal_policy[state]
            state, _, done, _, _ = environment.step(action)

        if render:
            environment.render()


def main(episodes=10000, demo_episodes=5):
    """
    Main function to run Q-Learning on the FrozenLake environment

    Args:
    episodes (int): Number of episodes to run Q-Learning
    demo_episodes (int): Number of episodes to demonstrate the learned policy
    """
    try:
        # Create and run Q-learning on the environment
        environment = gym.make("FrozenLake-v1")
        Q_table = q_learning(environment, episodes)

        # Evaluate the learned policy
        avg_reward = evaluate_policy(environment, Q_table, episodes)
        print(f"Average reward after q-learning: {avg_reward}")

        # Demonstrate the learned policy
        visual_env = gym.make('FrozenLake-v1', render_mode='human')
        demo_agent(visual_env, Q_table, demo_episodes)

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Clean up and close the environment
        environment.close()
        visual_env.close()

# Ensure the functions q_learning, evaluate_policy, and demo_agent are defined as previously discussed.

if __name__ == "__main__":
    main()

