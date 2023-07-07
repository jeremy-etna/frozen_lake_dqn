import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle


def run(episodes, is_training=True, render=False):
    game = gym.make(
        "FrozenLake-v1",
        map_name="8x8",
        is_slippery=True,
        render_mode="human" if render else None,
    )

    if is_training:
        q = np.zeros((game.observation_space.n, game.action_space.n))
    else:
        f = open("frozen_lake8x8.pkl", "rb")
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.0001  # alpha or learning rate
    discount_factor_g = 0.99  # gamma or discount factor.
    epsilon = 1  # 1 = 100% random actions
    epsilon_decay_rate = 0.005  # epsilon decay rate. 1/0.0001 = 10,000
    random_number_generator = np.random.default_rng()
    rewards_per_episode = np.zeros(episodes)
    # print("reward per episode:", rewards_per_episode)

    for episode in range(episodes):
        current_state = game.reset()[0]
        terminated = False  # True when fall in hole or reached goal
        truncated = False  # True when actions > 200

        while not terminated and not truncated:
            if is_training and random_number_generator.random() < epsilon:
                action = game.action_space.sample()
            else:
                action = np.argmax(q[current_state, :])

            new_state, reward, terminated, truncated, _ = game.step(action)

            if is_training:
                q[current_state, action] = q[
                    current_state, action
                ] + learning_rate_a * (
                    reward
                    + discount_factor_g * np.max(q[new_state, :])
                    - q[current_state, action]
                )

            current_state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if epsilon == 0:
            learning_rate_a = 0.0001

        if reward == 1:
            rewards_per_episode[episode] = 1

    game.close()

    sum_rewards = np.zeros(episodes)
    for step in range(episodes):
        sum_rewards[step] = np.sum(rewards_per_episode[max(0, step - 100) : (step + 1)])
    plt.plot(sum_rewards)
    plt.savefig("frozen_lake8x8.png")

    if is_training:
        f = open("frozen_lake8x8.pkl", "wb")
        pickle.dump(q, f)
        f.close()


if __name__ == "__main__":
    # run(100)

    run(10000, is_training=True, render=False)
