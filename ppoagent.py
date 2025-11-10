import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# --- MODIFIED (v5): Actor 为所有目的地决策 ---
class PPOActor(nn.Module):
    def __init__(self, state_dim, num_dests, num_next_hops, gru_hidden_dim):
        super().__init__()
        self.num_dests = num_dests
        self.num_next_hops = num_next_hops
        self.gru_hidden_dim = gru_hidden_dim

        self.gru = nn.GRU(state_dim, gru_hidden_dim, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_dim, 128),
            nn.ReLU(),
            # 输出: [所有目的地, 主/备, 所有下一跳]
            nn.Linear(128, num_dests * 2 * num_next_hops) 
        )
        
    def forward(self, x):
        # x 形状: [batch_size, seq_len, state_dim]
        gru_out, _ = self.gru(x)
        last_time_step_out = gru_out[:, -1, :]
        out = self.fc(last_time_step_out)
        
        # reshape 为 [batch, num_dests, 2, num_next_hops]
        return out.view(-1, self.num_dests, 2, self.num_next_hops)

# --- MODIFIED (v5): Critic 接收 状态 + 目标 ---
class PPOCritic(nn.Module):
    def __init__(self, state_dim, num_dests, gru_hidden_dim, dest_embedding_dim=16):
        super().__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.num_dests = num_dests
        self.dest_embedding_dim = dest_embedding_dim
        
        # 目标ID的嵌入层
        self.dest_embed = nn.Embedding(num_dests, dest_embedding_dim)

        # GRU层处理状态
        self.gru = nn.GRU(state_dim, gru_hidden_dim, batch_first=True)
        
        # 全连接层处理 状态(GRU输出) + 目标ID(嵌入)
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_dim + dest_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # 输出一个价值
        )

    def forward(self, x_state, x_dest_idx):
        # x_state 形状: [batch_size, seq_len, state_dim]
        # x_dest_idx 形状: [batch_size] (必须是 Long 类型)
        
        # 1. 处理状态
        gru_out, _ = self.gru(x_state)
        last_state_out = gru_out[:, -1, :] # 形状 [batch_size, gru_hidden_dim]
        
        # 2. 处理目标ID
        # .squeeze(1) 
        dest_embedding = self.dest_embed(x_dest_idx) # 形状 [batch_size, dest_embedding_dim]
        
        # 3. 拼接
        combined = torch.cat([last_state_out, dest_embedding], dim=1)
        
        # 4. 评估价值
        return self.fc(combined)
    

class PPOAgent:
    def __init__(self, state_dim, num_dests, num_next_hops, gru_hidden_dim, dest_embedding_dim,
                 actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, eps_clip=0.2,
                 critic=None, critic_optimizer=None, batch_size=32):
        
        self.actor = PPOActor(state_dim, num_dests, num_next_hops, gru_hidden_dim)
        self.optimizerA = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # --- MODIFIED (v5): 坚持使用共享 Critic ---
        if critic:
            self.critic = critic
        else:
            raise ValueError("Shared Critic 必须被提供")
            
        if critic_optimizer:
            self.optimizerC = critic_optimizer
        else:
            raise ValueError("Shared Critic Optimizer 必须被提供")
        # --- 变化结束 ---

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.batch_size = batch_size
        self.memory = deque() 
        self.num_dests = num_dests
        self.num_next_hops = num_next_hops

    def select_action(self, state, self_index, neighbor_map_indices):
        state = torch.FloatTensor(state).unsqueeze(0)  # [1, seq_len, state_dim]
        
        # logits 形状 [1, num_dests, 2, num_next_hops]
        logits = self.actor(state)
        
        actions = []
        logprobs = []
        
        for dest_index in range(self.num_dests):
            dest_actions = []
            dest_logprobs = []
            for path_type in range(2):
                
                path_logits = logits[0, dest_index, path_type].clone() 
                mask_value = -torch.inf
                
                # --- MODIFIED (v5): 掩码逻辑更新 ---
                # 1. 创建一个全掩码
                mask = torch.full_like(path_logits, mask_value)
                
                # 2. 只允许选择有效的邻居
                if neighbor_map_indices: # 确保邻居列表不为空
                    valid_indices = torch.LongTensor(neighbor_map_indices)
                    mask[valid_indices] = 0.0 # 允许选择邻居
                
                # 3. (双重保险) 确保不能选自己或目的地
                mask[self_index] = mask_value
                mask[dest_index] = mask_value
                
                # 4. 应用掩码
                path_logits = path_logits + mask
                # --- 掩码结束 ---

                probs = torch.softmax(path_logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                
                dest_actions.append(action.item())
                dest_logprobs.append(dist.log_prob(action).item())
                
            actions.append(dest_actions)      # [num_dests, 2]
            logprobs.append(dest_logprobs)    # [num_dests, 2]
            
        return actions, logprobs  

    def store(self, transition):
        # transition: (state, actions, rewards_list, next_state, old_logprobs, done)
        self.memory.append(transition)

    def learn(self): 
        if len(self.memory) < self.batch_size:
            return 
        
        batch = list(self.memory)
        self.memory.clear() 
        
        # rewards_list 是 [batch, num_dests]
        states, actions, rewards_list, next_states, old_logprobs_list, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        actions = torch.LongTensor(actions)               # [batch, num_dests, 2]
        rewards = torch.FloatTensor(rewards_list)         # [batch, num_dests]
        old_logprobs = torch.FloatTensor(old_logprobs_list) # [batch, num_dests, 2]
        dones = torch.FloatTensor(dones).unsqueeze(1)     # [batch, 1]

        # --- MODIFIED (v5): 循环为每个目的地训练 ---
        # Actor 和 Critic 的损失是所有目的地的总和
        
        actor_loss_total = 0
        critic_loss_total = 0
        
        logits = self.actor(states) # [batch, num_dests, 2, num_next_hops]

        for dest_index in range(self.num_dests):
            # 准备 Critic 输入
            dest_idx_tensor = torch.LongTensor([dest_index] * self.batch_size) # [batch_size]
            
            # --- 计算优势 (Advantage) ---
            # V(s, d) 和 V(s', d)
            values = self.critic(states, dest_idx_tensor).squeeze() # [batch_size]
            next_values = self.critic(next_states, dest_idx_tensor).squeeze() # [batch_size]
            
            # 奖励 R(s, d)
            dest_rewards = rewards[:, dest_index] # [batch_size]
            
            td_target = dest_rewards + self.gamma * next_values * (1 - dones.squeeze())
            advantage = (td_target - values).detach() 

            # --- 累加 Critic 损失 ---
            critic_loss_total += nn.functional.mse_loss(values, td_target)

            # --- 累加 Actor 损失 (用于此目的地) ---
            for path_type in range(2):
                probs = torch.softmax(logits[:, dest_index, path_type], dim=-1)
                dist = torch.distributions.Categorical(probs)
                
                logprobs = dist.log_prob(actions[:, dest_index, path_type])
                old_logs = old_logprobs[:, dest_index, path_type]
                
                ratio = torch.exp(logprobs - old_logs)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
                actor_loss_total += -torch.min(surr1, surr2).mean()

        # --- 归一化总损失 ---
        actor_loss = actor_loss_total / (self.num_dests * 2)
        critic_loss = critic_loss_total / self.num_dests

        # --- 执行反向传播 ---
        self.optimizerA.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5) # 梯度裁剪
        self.optimizerA.step()

        self.optimizerC.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5) # 梯度裁剪
        self.optimizerC.step()