import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch_geometric as tg

# class PPOActor(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(state_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, action_dim),
#             nn.Softmax(dim=-1)
#         )
#     def forward(self, x):
#         return self.fc(x)

class PPOActor(nn.Module):
    def __init__(self, state_dim, num_dests, num_next_hops):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_dests * num_next_hops * 2)  # 2: 主用+备用
        )
        self.num_dests = num_dests
        self.num_next_hops = num_next_hops
        
        
    def forward(self, x):
        out = self.fc(x)
        # reshape为 [batch, num_dests, 2, num_next_hops]
        return out.view(-1, self.num_dests, 2, self.num_next_hops)

class PPOCritic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.fc(x)
    
# PPOAgent类实现PPO算法
# class PPOAgent:
#     def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, eps_clip=0.2):
#         self.actor = PPOActor(state_dim, action_dim)
#         self.critic = PPOCritic(state_dim)
#         self.optimizerA = optim.Adam(self.actor.parameters(), lr=lr)
#         self.optimizerC = optim.Adam(self.critic.parameters(), lr=lr)
#         self.gamma = gamma
#         self.eps_clip = eps_clip
#         self.memory = []

#     def select_action(self, state):
#         state = torch.FloatTensor(state).unsqueeze(0)
#         probs = self.actor(state)
#         dist = torch.distributions.Categorical(probs)
#         action = dist.sample()
#         return action.item(), dist.log_prob(action).item()

#     def store(self, transition):
#         self.memory.append(transition)

#     def learn(self):
#         if len(self.memory) == 0:
#             return
#         states, actions, rewards, next_states, old_logprobs, dones = zip(*self.memory)
#         states = torch.FloatTensor(states)
#         actions = torch.LongTensor(actions)
#         rewards = torch.FloatTensor(rewards)
#         next_states = torch.FloatTensor(next_states)
#         old_logprobs = torch.FloatTensor(old_logprobs)
#         dones = torch.FloatTensor(dones)

#         # 计算优势
#         values = self.critic(states).squeeze()
#         next_values = self.critic(next_states).squeeze()
#         td_target = rewards + self.gamma * next_values * (1 - dones)
#         advantage = td_target - values

#         # 更新Actor
#         probs = self.actor(states)
#         dist = torch.distributions.Categorical(probs)
#         logprobs = dist.log_prob(actions)
#         ratio = torch.exp(logprobs - old_logprobs)
#         surr1 = ratio * advantage.detach()
#         surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage.detach()
#         actor_loss = -torch.min(surr1, surr2).mean()

#         # 更新Critic
#         critic_loss = nn.functional.mse_loss(values, td_target.detach())

#         self.optimizerA.zero_grad()
#         actor_loss.backward()
#         self.optimizerA.step()

#         self.optimizerC.zero_grad()
#         critic_loss.backward()
#         self.optimizerC.step()

#         self.memory = []

#多头输出PPOAgent 1为主用路径，0为备用路径
class PPOAgent:
    def __init__(self, state_dim, num_dests, num_next_hops, lr=1e-3, gamma=0.99, eps_clip=0.2):
        self.actor = PPOActor(state_dim, num_dests, num_next_hops)
        self.critic = PPOCritic(state_dim)
        self.optimizerA = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizerC = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.memory = []
        self.next_hops_list = []
        self.num_dests = num_dests
        self.num_next_hops = num_next_hops

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)  # [1, state_dim]
        logits = self.actor(state)  # [1, num_dests, 2, num_next_hops]
        actions = []
        logprobs = []
        for dest in range(self.num_dests):
            dest_actions = []
            dest_logprobs = []
            for path_type in range(2):  # 0:主用, 1:备用
                probs = torch.softmax(logits[0, dest, path_type], dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                dest_actions.append(action.item())
                dest_logprobs.append(dist.log_prob(action).item())
            actions.append(dest_actions)      # [主用, 备用]
            logprobs.append(dest_logprobs)
        return actions, logprobs  # actions: [num_dests, 2], logprobs: [num_dests, 2]

    def store(self, transition):
        self.memory.append(transition)

    def learn(self):
        if len(self.memory) == 0:
            return
        states, actions, rewards, next_states, old_logprobs, dones = zip(*self.memory)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)  # shape: [batch, num_dests, 2]
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        old_logprobs = torch.FloatTensor(old_logprobs)  # shape: [batch, num_dests, 2]
        dones = torch.FloatTensor(dones)

        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()
        td_target = rewards + self.gamma * next_values * (1 - dones)
        advantage = td_target - values

        logits = self.actor(states)  # [batch, num_dests, 2, num_next_hops]
        actor_loss = 0
        for dest in range(self.num_dests):
            for path_type in range(2):
                probs = torch.softmax(logits[:, dest, path_type], dim=-1)
                dist = torch.distributions.Categorical(probs)
                logprobs = dist.log_prob(actions[:, dest, path_type])
                ratio = torch.exp(logprobs - old_logprobs[:, dest, path_type])
                surr1 = ratio * advantage.detach()
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage.detach()
                actor_loss += -torch.min(surr1, surr2).mean()
        actor_loss = actor_loss / (self.num_dests * 2)

        critic_loss = nn.functional.mse_loss(values, td_target.detach())

        self.optimizerA.zero_grad()
        actor_loss.backward()
        self.optimizerA.step()

        self.optimizerC.zero_grad()
        critic_loss.backward()
        self.optimizerC.step()

        self.memory = []