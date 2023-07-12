import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import os


class Network(nn.Module):
    def __init__(self, input_states, output_actions):
        super(Network, self).__init__()
        self.input_states = input_states
        self.output_actions = output_actions
        self.fc1 = nn.Linear(self.input_states, 32)
        self.fc2 = nn.Linear(32, output_actions)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

class replayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: torch.cat(x, 0), samples)

    def __len__(self):
        return len(self.memory)
    
class DQN():
    def __init__(self,input_states, output_actions, gamma):
        self.model = Network(input_states, output_actions)
        self.gamma = gamma
        self.reward_window = []
        self.memory = replayMemory(10000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.current_state = torch.Tensor(input_states).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state):
        # probs = F.softmax(self.model(state) * temperature, dim=1)
        probs = F.softmax(self.model(state) * temperature, dim=0)  # Changez dim=1 à dim=0
        action = probs.multinomial(num_samples=1)
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        targets = self.gamma * next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, targets)
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()

    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float()
        self.memory.push((self.current_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_reward = reward
        self.current_state = new_state
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1.)
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, 'last_brain.pth')
        
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint...")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("terminated !")
        else:
            print("no checkpoint found...")



def train_dqn(episodes):
    loss = []
    score = []
    score_average = []

    for episode in range(episodes):
        state = env.reset()[0]
        state = np.identity(16)[state:state + 1]
        state = state.reshape(1, -1)  # Ajoutez cette ligne

        total_reward = 0
        terminated = False
        steps = 0

        while not terminated:
            action = brain.update(0, state.tolist())
            next_state, reward, terminated, truncated, info = env.step(action.item())
            next_state = np.identity(16)[next_state]
            next_state = next_state.reshape(1, -1)  # Ajoutez cette ligne
            total_reward += reward
            state = next_state
            steps += 1

        score.append(total_reward)
        score_average.append(sum(score[-100:]) / 100)
        # print('Episode:', episode, ' Score:', total_reward, ' Average score:', score_average[-1])

    brain.save()
    plt.plot(score)
    plt.plot(score_average)
    plt.show()

    print('Final Average Score:', score_average[-1])

if __name__ == "__main__":
    env = gym.make('FrozenLake-v1')
    temperature = 0
    gamma = 0.99
    brain = DQN(16, 4, gamma)
    train_dqn(1000)
    
