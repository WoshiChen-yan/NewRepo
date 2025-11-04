import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
# import torch_geometric as tg # (这个库在您当前的代码中没有使用，暂时注释掉)
from collections import deque # <--- MODIFIED: 导入deque

# <--- MODIFIED: 整个 PPOActor 类被重写以包含 GRU ---_>
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

# <--- MODIFIED: 整个 PPOCritic 类被重写以包含 GRU ---_>
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
    
# <--- MODIFIED: PPOAgent 类被修改以支持共享 Critic 和 GRU ---_>
class PPOAgent:
    def __init__(self, state_dim, num_dests, num_next_hops, gru_hidden_dim, 
                 actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, eps_clip=0.2,
                 critic=None, critic_optimizer=None): # <-- MODIFIED: 添加了新参数
        
        self.actor = PPOActor(state_dim, num_dests, num_next_hops, gru_hidden_dim)
        self.optimizerA = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # <--- MODIFIED: 实现共享 Critic ---_>
        # 如果外部提供了 critic (共享的)，则使用它
        if critic:
            self.critic = critic
        else:
            # 否则 (例如用于测试)，创建自己的 critic
            self.critic = PPOCritic(state_dim, gru_hidden_dim)
            
        # 同样，使用共享的优化器 (如果提供)
        if critic_optimizer:
            self.optimizerC = critic_optimizer
        else:
            self.optimizerC = optim.Adam(self.critic.parameters(), lr=critic_lr)
        # <--- MODIFIED 结束 ---_>

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.memory = deque(maxlen=1000) # <--- MODIFIED: 使用 deque 提高内存效率
        self.num_dests = num_dests
        self.num_next_hops = num_next_hops

    def select_action(self, state):
        # state 现在的预期形状是 (seq_len, state_dim)
        # 需要将其转换为 [1, seq_len, state_dim] 以便GRU处理
        state = torch.FloatTensor(state).unsqueeze(0)  # [1, seq_len, state_dim]
        
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
        # actions: [num_dests, 2], logprobs: [num_dests, 2]
        return actions, logprobs  
    def store(self, transition):
        # transition 是一个元组: (state, action, reward, next_state, old_logprob, done)
        self.memory.append(transition)

    def learn(self, batch_size=32): # <--- MODIFIED: 添加 batch_size
        if len(self.memory) < batch_size: # <--- MODIFIED
            return
        
        # <--- MODIFIED: 从内存中随机采样一个批次 ---_>
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, old_logprobs, dones = zip(*batch)

        # <--- MODIFIED: 转换状态，现在它们是序列 ---_>
        # states 和 next_states 是元组，每个元素是 (seq_len, state_dim)
        # 我们需要将它们堆叠成 (batch_size, seq_len, state_dim)
        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        # <--- MODIFIED 结束 ---_>

        actions = torch.LongTensor(actions)  # shape: [batch, num_dests, 2]
        rewards = torch.FloatTensor(rewards)
        old_logprobs = torch.FloatTensor(old_logprobs)  # shape: [batch, num_dests, 2]
        dones = torch.FloatTensor(dones)

        # --- 计算优势 (Advantage) ---
        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()
        td_target = rewards + self.gamma * next_values * (1 - dones)
        advantage = (td_target - values).detach() # <--- MODIFIED: 立即 detach

        # --- 更新 Actor ---
        logits = self.actor(states)  # [batch, num_dests, 2, num_next_hops]
        actor_loss_total = 0
        for dest in range(self.num_dests):
            for path_type in range(2):
                probs = torch.softmax(logits[:, dest, path_type], dim=-1)
                dist = torch.distributions.Categorical(probs)
                logprobs = dist.log_prob(actions[:, dest, path_type])
                
                ratio = torch.exp(logprobs - old_logprobs[:, dest, path_type])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
                actor_loss_total += -torch.min(surr1, surr2).mean()
        
        actor_loss = actor_loss_total / (self.num_dests * 2)

        # --- 更新 Critic ---
        # Critic 的损失是 TD target 和当前 value 之间的 MSE
        critic_loss = nn.functional.mse_loss(values, td_target)

        self.optimizerA.zero_grad()
        actor_loss.backward()
        self.optimizerA.step()

        self.optimizerC.zero_grad()
        critic_loss.backward()
        self.optimizerC.step()

        # <--- MODIFIED: 不再清空内存，PPO通常会重用数据 ---_>
        # self.memory = [] 
        # (注意：如果使用 PPO 的 on-policy 方式，您应该在 learn() 之后清空 self.memory)
        # (为简单起见，我们暂时不清空，使其更像 off-policy。
        #  如果需要严格的 on-policy, 您应该在 agent.store() 之外收集一个完整的轨迹，
        #  然后调用 learn()，最后清空内存。)
        # 
        #  我们还是改回 on-policy 的标准做法：在 `learn` 被调用后清空内存
        self.memory.clear()