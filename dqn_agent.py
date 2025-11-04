import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQNNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNNet(state_dim, action_dim).to(self.device)
        self.target = DQNNet(state_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 0.9
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.batch_size = 32

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.model.fc[-1].out_features - 1)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.model(state)
        return q.argmax().item()

    def store(self, s, a, r, s_):
        self.memory.append((s, a, r, s_))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s_ = zip(*batch)
        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        s_ = torch.FloatTensor(s_).to(self.device)
        q = self.model(s).gather(1, a.unsqueeze(1)).squeeze()
        q_ = self.target(s_).max(1)[0].detach()
        target = r + self.gamma * q_
        loss = nn.MSELoss()(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay