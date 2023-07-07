import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Paramètres
gamma: float = 0.99
epsilon: float = 1.0
learning_rate: float = 0.01
batch_size: int = 32
max_episodes: int = 500
max_timesteps: int = 100


class DQN(nn.Module):
    def __init__(self, obs_space, action_space, embedding_dim=64):
        super(DQN, self).__init__()

        self.embedding_layer = nn.Embedding(obs_space, embedding_dim)
        self.fully_connected = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_space),
        )

    def forward(self, x):
        x = self.embedding_layer(x.long())
        return self.fully_connected(x)


# Environnement
game = gym.make("FrozenLake-v1")
# game = gym.make("FrozenLake-v1", render_mode="human")


# Modèle
model = DQN(game.observation_space.n, game.action_space.n)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def choose_action(current_state: int, epsilon: float) -> int:
    if np.random.random() < epsilon:
        action: int = game.action_space.sample()
    else:
        state: torch.tensor = torch.Tensor([current_state])
        q_values: torch.tensor = model(state)
        # print("q_values:", q_values)
        action: int = q_values.argmax().item()
    return action


def learn(batch: list[tuple[int, int, int, int, bool]]) -> None:
    # Dézipper le batch
    current_states, actions, rewards, next_states, dones = batch
    # Convertir les listes en tensors
    current_states = torch.tensor(current_states)
    actions = torch.LongTensor([actions])
    rewards = torch.Tensor([rewards])
    next_states = torch.tensor([next_states])
    dones = torch.BoolTensor([dones])
    # Utilisation du modèle pour obtenir les Q-values actuels
    current_q_values = model(current_states).gather(0, actions)
    # Utilisation du modèle pour obtenir les Q-values du prochain état
    next_q_values = model(next_states).max(1)[0]
    # Calcul des Q-values cibles
    target_q_values = rewards + gamma * next_q_values * (1 - dones.float())
    # Calcul de la perte entre les Q-values actuels et les Q-values cibles
    loss = nn.MSELoss()(current_q_values, target_q_values.detach())
    # Remise à zéro des gradients
    optimizer.zero_grad()
    # Calcul des gradients
    loss.backward()
    # Mise à jour des poids du modèle
    optimizer.step()


total_rewards = []
# Apprentissage
for episode in range(max_episodes):
    current_state: int = game.reset()[0]
    episode_total_reward: int = 0

    for step in range(max_timesteps):
        action: int = choose_action(current_state, epsilon)

        #  int      float      bool       bool     dict
        next_state, reward, terminated, truncated, info = game.step(action)

        if terminated:
            if step < max_timesteps - 1:
                reward = -1
            else:
                reward = 1
        else:
            reward = 0

        #           int         int      int     int         bool
        batch = [current_state, action, reward, next_state, terminated]

        learn(batch)

        current_state = next_state
        episode_total_reward += reward

        if terminated or step == max_timesteps - 1:
            total_rewards.append(episode_total_reward)
            # print(
            #     f"EPISODE {episode + 1}/{max_episodes}, Reward: {episode_total_reward}"
            # )
            break

    # Décroissance de epsilon
    epsilon = max(0.01, epsilon * 0.995)

game.close()

plt.plot(total_rewards)
plt.ylabel("Total Reward")
plt.xlabel("Episode")
plt.show()
