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
        dest_embedding = self.dest_embed(x_dest_idx) # 形状 [batch_size, dest_embedding_dim]
        if dest_embedding.dim() == 3: # 处理 [batch_size, 1, dim] 的情况
             dest_embedding = dest_embedding.squeeze(1)
        
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

    # --- MODIFIED (v6): 动作掩码 (Action Masking) ---
    def select_action(self, state, self_index, neighbor_map_indices):
        """
        Args:
            state (np.array): 形状为 [seq_len, state_dim]
            self_index (int): 智能体所在节点的 *全局* 索引
            neighbor_map_indices (list): *有效邻居* 的 *全局* 索引列表
        """
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
                mask_value = -torch.inf # 使用 -infinity
                
                # --- 动作掩码 (Action Masking) ---
                # 1. 创建一个全掩码 (所有动作都被禁止)
                mask = torch.full_like(path_logits, mask_value)
                
                # 2. 只有有效的邻居可以被选中
                if neighbor_map_indices: # 检查列表是否为空
                    valid_indices = torch.LongTensor(neighbor_map_indices)
                    mask[valid_indices] = 0.0 # 解除掩码
                
                # 3. 严格禁止：选择自己
                mask[self_index] = mask_value
                # 4. 严格禁止：选择目的地
                # mask[dest_index] = mask_value
                

                # 应用掩码 (logits + mask)
                path_logits = path_logits + mask
                
                # 检查是否所有动作都被掩盖了 (例如, 孤立节点)
                if torch.all(torch.isinf(path_logits)):
                    # 如果全被掩盖, 强行选择一个 (比如自己，虽然无效但至少不会崩溃)
                    action = torch.tensor(self_index) 
                    logprob = torch.tensor(-1e9) # 极低的 log_prob
                else:
                    probs = torch.softmax(path_logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()
                    logprob = dist.log_prob(action)

                dest_actions.append(action.item())
                dest_logprobs.append(logprob.item())
                
            actions.append(dest_actions)      # [num_dests, 2]
            logprobs.append(dest_logprobs)    # [num_dests, 2]
            
        return actions, logprobs  

    def store(self, transition):
        # transition: (state, actions, rewards_list, next_state, old_logprobs, done)
        self.memory.append(transition)

    # --- MODIFIED (v6): 添加梯度裁剪 ---
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

        current_batch_size = states.shape(0)
        
        actor_loss_total = 0
        critic_loss_total = 0
        
        logits = self.actor(states) # [batch, num_dests, 2, num_next_hops]

        for dest_index in range(self.num_dests):
            # 准备 Critic 输入
            dest_idx_tensor = torch.LongTensor([dest_index] * current_batch_size) # [batch_size]
            
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