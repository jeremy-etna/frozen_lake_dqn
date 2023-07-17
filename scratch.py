import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import os
import pandas as pd


# Définir la structure du réseau de neurones
class Network(nn.Module):
    def __init__(self, input_states: int, output_actions: int):
        super(Network, self).__init__()
        self.input_states = input_states
        self.output_actions = output_actions
        self.fc1 = nn.Linear(self.input_states, 32)
        self.fc2 = nn.Linear(32, output_actions)

    # La méthode de passage en avant du réseau
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values


# Définir la structure de la mémoire de replay
class replayMemory(object):
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    # Ajouter une transition à la mémoire
    def push(
        self, transition: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    # Récupérer un échantillon de transitions de la mémoire
    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        samples = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states = zip(*samples)
        return (
            torch.stack(states),
            torch.stack(actions),
            torch.stack(rewards),
            torch.stack(next_states),
        )

    def __len__(self):
        return len(self.memory)


# Définir la structure de l'agent DQN
class DQN:
    def __init__(self, input_states: int, output_actions: int, gamma: float):
        self.model = Network(input_states, output_actions)
        self.gamma = gamma
        self.epsilon = 1.0
        self.reward_window = []
        self.memory = replayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.current_state = torch.Tensor(input_states).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        self.reward_history = []

    # Sélectionner une action à prendre
    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(self.model(state) * temperature, dim=0)
        print("Action probabilities:", probs)
        action = probs.multinomial(num_samples=1)
        return action.data[0, 0]

    # Apprendre à partir d'un batch de transitions
    def learn(
        self,
        batch_state: torch.Tensor,
        batch_next_state: torch.Tensor,
        batch_reward: torch.Tensor,
        batch_action: torch.Tensor,
    ):
        outputs = (
            self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        )
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        print("Outputs:", outputs)
        print("Next outputs:", next_outputs)
        targets = self.gamma * next_outputs + batch_reward
        print("Targets:", targets)
        td_loss = F.smooth_l1_loss(outputs, targets)
        print("TD loss:", td_loss)
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()

    # Mettre à jour l'agent avec une nouvelle transition
    def update(self, reward: float, new_signal: list[float]) -> int:
        new_state = torch.Tensor(new_signal).float()
        print("New state:", new_state)
        # ...
        action = self.select_action(new_state)
        print("Selected action:", action)
        if len(self.memory.memory) > batch_size:
            (
                batch_state,
                batch_next_state,
                batch_action,
                batch_reward,
            ) = self.memory.sample(batch_size)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_reward = reward
        self.current_state = new_state
        print("Current state:", self.current_state)
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        self.reward_history.append(reward)
        return action

    # Calculer le score moyen de l'agent sur les derniers épisodes
    def score(self) -> float:
        return sum(self.reward_window) / (len(self.reward_window) + 1.0)

    # Sauvegarder le modèle de l'agent
    def save(self):
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            "last_brain.pth",
        )

    # Charger le modèle de l'agent
    def load(self):
        if os.path.isfile("last_brain.pth"):
            print("=> loading checkpoint...")
            checkpoint = torch.load("last_brain.pth")
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            print("terminated !")
        else:
            print("no checkpoint found...")

    # Méthode pour sauvegarder l'historique des récompenses
    def save_reward_history(self, filename: str):
        df = pd.DataFrame(self.reward_history, columns=["Reward"])
        df.to_csv(filename, index=False)


# Fonction pour entraîner l'agent DQN sur un certain nombre d'épisodes
def train_dqn(episodes: int):
    score = []
    score_average = []

    for episode in range(episodes):
        print("---------------------------------------------")
        state = env.reset()[0]
        print("Initial state:", state)
        state = np.identity(16)[state : state + 1]
        state = state.reshape(1, -1)

        total_reward = 0
        terminated = False
        steps = 0

        while not terminated:
            action = brain.select_action(torch.Tensor(state))
            print("Selected action:", action)
            next_state, reward, terminated, truncated, info = env.step(action.item())
            print(
                "Next state, reward, termination status:",
                next_state,
                reward,
                terminated,
            )
            next_state = np.identity(16)[next_state]
            next_state = next_state.reshape(1, -1)
            brain.update(reward, state.tolist())

            total_reward += reward
            state = next_state
            steps += 1

        score.append(total_reward)
        score_average.append(sum(score[-100:]) / 100)

    brain.save()
    plt.plot(score, label="Reward")
    plt.plot(score_average, label="Average Reward")
    plt.title("Average Reward and reward over Time")
    plt.xlabel("Episode")
    plt.plot(score_average)
    plt.grid(True)
    plt.show()

    print("Final Average Score:", score_average[-1])
    brain.save_reward_history("reward_history.csv")


# Point d'entrée du programme
if __name__ == "__main__":
    env = gym.make("FrozenLake-v1")
    temperature = 100  # Temperature for softmax
    gamma = 0.5  # Discount factor 0=greedy, 1=long term
    learning_rate = 0.001  # Learning rate
    batch_size = (
        128  # Batch size for replay memory low values = faster training but less stable
    )

    brain = DQN(16, 4, gamma)
    train_dqn(1000)
