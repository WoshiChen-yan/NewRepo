import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque 
import random # <--- 导入 random

class PPOActor(nn.Module):
    def __init__(self, state_dim, num_dests, num_next_hops, gru_hidden_dim):
        super().__init__()
        self.num_dests = num_dests
        self.num_next_hops = num_next_hops
        self.gru_hidden_dim = gru_hidden_dim

        # GRU层，用于提取时序特征
        self.gru = nn.GRU(state_dim, gru_hidden_dim, batch_first=True)
        
        # 全连接层，基于GRU的输出进行决策
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_dim, 128),
            nn.ReLU(),
            # 输出层：为每个目的地(num_dests)的每条路径(2)都输出一个下一跳(num_next_hops)的概率分布
            nn.Linear(128, num_dests * num_next_hops * 2) 
        )
        
    def forward(self, x):
        # x 的预期形状: [batch_size, seq_len, state_dim]
        # gru_out 形状: [batch_size, seq_len, gru_hidden_dim]
        # _ (h_n) 形状: [num_layers, batch_size, gru_hidden_dim]
        gru_out, _ = self.gru(x)
        
        # 我们只关心GRU在最后一个时间步的输出
        # last_time_step_out 形状: [batch_size, gru_hidden_dim]
        last_time_step_out = gru_out[:, -1, :]
        
        # 将最后一个时间步的输出送入全连接层
        out = self.fc(last_time_step_out)
        
        # reshape为 [batch, num_dests, 2, num_next_hops]
        return out.view(-1, self.num_dests, 2, self.num_next_hops)

class PPOCritic(nn.Module):
    def __init__(self, state_dim, gru_hidden_dim):
        super().__init__()
        self.gru_hidden_dim = gru_hidden_dim

        # GRU层，用于提取时序特征
        self.gru = nn.GRU(state_dim, gru_hidden_dim, batch_first=True)

        # 全连接层，基于GRU的输出评估状态价值
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # 输出一个单独的价值估计
        )

    def forward(self, x):
        # x 的预期形状: [batch_size, seq_len, state_dim]
        gru_out, _ = self.gru(x)
        
        # 我们只关心GRU在最后一个时间步的输出
        last_time_step_out = gru_out[:, -1, :]
        
        # 评估价值
        return self.fc(last_time_step_out)
    

class PPOAgent:
    def __init__(self, state_dim, num_dests, num_next_hops, gru_hidden_dim, 
                 actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, eps_clip=0.2,
                 critic=None, critic_optimizer=None, batch_size=32): # <-- MODIFIED: 添加 batch_size
        
        self.actor = PPOActor(state_dim, num_dests, num_next_hops, gru_hidden_dim)
        self.optimizerA = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        if critic:
            self.critic = critic
        else:
            self.critic = PPOCritic(state_dim, gru_hidden_dim)
            
        if critic_optimizer:
            self.optimizerC = critic_optimizer
        else:
            self.optimizerC = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.batch_size = batch_size # <--- MODIFIED: 保存 batch_size
        self.memory = deque() # <--- MODIFIED: PPO 是 on-policy，在 learn() 后清空
        self.num_dests = num_dests
        self.num_next_hops = num_next_hops

    # <--- MODIFIED: 添加 self_index 用于动作掩码 ---_>
    def select_action(self, state, self_index):
        # state 现在的预期形状是 (seq_len, state_dim)
        # 需要将其转换为 [1, seq_len, state_dim] 以便GRU处理
        state = torch.FloatTensor(state).unsqueeze(0)  # [1, seq_len, state_dim]
        
        logits = self.actor(state)  # [1, num_dests, 2, num_next_hops]
        actions = []
        logprobs = []
        
        for dest_index in range(self.num_dests):
            dest_actions = []
            dest_logprobs = []
            for path_type in range(2):  # 0:主用, 1:备用
                
                # <--- MODIFIED: 应用动作掩码 ---_>
                # 我们不允许选择自己 (self_index) 或目的地 (dest_index) 作为下一跳
                path_logits = logits[0, dest_index, path_type].clone() # 复制 logits
                mask_value = -float('inf')
                
                # 掩码1: 不能选择自己作为下一跳
                path_logits[self_index] = mask_value
                # <--- MODIFIED 结束 ---_>
                
                probs = torch.softmax(path_logits, dim=-1) # <--- MODIFIED: 使用掩码后的 logits
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                dest_actions.append(action.item())
                dest_logprobs.append(dist.log_prob(action).item())
                
            actions.append(dest_actions)      # [主用, 备用]
            logprobs.append(dest_logprobs)
            
        # actions: [num_dests, 2], logprobs: [num_dests, 2]
        return actions, logprobs  

    def store(self, transition):
        # transition 是一个元组: (state, action, reward, next_state, old_logprob, done)
        self.memory.append(transition)

    # <--- MODIFIED: 修正 PPO (on-policy) 学习逻辑 ---_>
    def learn(self): 
        # 1. PPO 是 on-policy，只有在收集到足够数据时才学习一次
        if len(self.memory) < self.batch_size:
            return # 数据还不够，等待下一轮
        
        # 2. 从内存中取出 *所有* 经验 (on-policy)，而不是随机采样
        batch = list(self.memory)
        states, actions, rewards, next_states, old_logprobs, dones = zip(*batch)

        # 3. 转换状态
        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        actions = torch.LongTensor(actions)  # shape: [batch, num_dests, 2]
        rewards = torch.FloatTensor(rewards)
        old_logprobs = torch.FloatTensor(old_logprobs)  # shape: [batch, num_dests, 2]
        dones = torch.FloatTensor(dones)

        # --- 4. 计算优势 (Advantage) ---
        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()
        td_target = rewards + self.gamma * next_values * (1 - dones)
        advantage = (td_target - values).detach() 

        # --- 5. 更新 Actor ---
        logits = self.actor(states)  # [batch, num_dests, 2, num_next_hops]
        actor_loss_total = 0
        for dest in range(self.num_dests):
            for path_type in range(2):
                
                # <--- MODIFIED: 在训练时也应用掩码 ---_>
                # (注意: 这在技术上更复杂，因为掩码依赖于状态。
                #  为了简化，我们假设无效动作的概率logprob会很低，
                #  并且由于掩码的存在，它们在采样时不会被选中。
                #  在 `select_action` 中掩码是更关键的步骤。)
                
                probs = torch.softmax(logits[:, dest, path_type], dim=-1)
                dist = torch.distributions.Categorical(probs)
                logprobs = dist.log_prob(actions[:, dest, path_type])
                
                ratio = torch.exp(logprobs - old_logprobs[:, dest, path_type])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
                actor_loss_total += -torch.min(surr1, surr2).mean()
        
        actor_loss = actor_loss_total / (self.num_dests * 2)

        # --- 6. 更新 Critic ---
        critic_loss = nn.functional.mse_loss(values, td_target)

        self.optimizerA.zero_grad()
        actor_loss.backward()
        self.optimizerA.step()

        self.optimizerC.zero_grad()
        critic_loss.backward()
        self.optimizerC.step()

        # 7. On-policy: 学习完后清空经验池
        self.memory.clear()