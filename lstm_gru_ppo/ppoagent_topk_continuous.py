import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class PPOActorTopKContinuous(nn.Module):
    def __init__(self, state_dim, num_dests, num_next_hops, gru_hidden_dim, top_k):
        super().__init__()
        self.num_dests = int(num_dests)
        self.num_next_hops = int(num_next_hops)
        self.top_k = int(top_k)

        self.gru = nn.GRU(state_dim, gru_hidden_dim, batch_first=True)
        self.hop_head = nn.Sequential(
            nn.Linear(gru_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_dests * self.num_next_hops),
        )
        self.alloc_head = nn.Sequential(
            nn.Linear(gru_hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_dests * self.top_k),
        )

    def forward(self, x):
        y, _ = self.gru(x)
        last = y[:, -1, :]
        hop_logits = self.hop_head(last).view(-1, self.num_dests, self.num_next_hops)
        alloc_raw = self.alloc_head(last).view(-1, self.num_dests, self.top_k)
        dirichlet_alpha = torch.nn.functional.softplus(alloc_raw) + 1.0
        return hop_logits, dirichlet_alpha


class PPOAgentTopKContinuous:
    def __init__(
        self,
        state_dim,
        num_dests,
        num_next_hops,
        gru_hidden_dim,
        dest_embedding_dim,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        eps_clip=0.2,
        critic=None,
        critic_optimizer=None,
        batch_size=32,
        entropy_coeff=0.01,
        top_k=3,
    ):
        self.num_dests = int(num_dests)
        self.num_next_hops = int(num_next_hops)
        self.gamma = float(gamma)
        self.eps_clip = float(eps_clip)
        self.batch_size = int(batch_size)
        self.entropy_coeff = float(entropy_coeff)
        self.top_k = int(max(2, top_k))

        self.actor = PPOActorTopKContinuous(state_dim, num_dests, num_next_hops, gru_hidden_dim, self.top_k)
        self.optimizerA = optim.Adam(self.actor.parameters(), lr=actor_lr)

        if critic is None or critic_optimizer is None:
            raise ValueError("Shared critic and optimizer are required")
        self.critic = critic
        self.optimizerC = critic_optimizer

        self.memory = deque()

    def _valid_hops(self, self_index, neighbor_map_indices, dest_index):
        valid = set(neighbor_map_indices or [])
        valid.discard(self_index)
        valid.discard(dest_index)
        return sorted(valid)

    def select_action(self, state, self_index, neighbor_map_indices):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        hop_logits, dir_alpha = self.actor(state_t)

        selected_hops = []
        split_allocs = []
        hop_logps = []
        alloc_logps = []

        eps = 1e-6

        for d in range(self.num_dests):
            valid = self._valid_hops(self_index, neighbor_map_indices, d)
            logits = hop_logits[0, d].clone()

            if not valid:
                selected_hops.append([self_index] * self.top_k)
                split_allocs.append([1.0 / self.top_k] * self.top_k)
                hop_logps.append(-1e9)
                alloc_logps.append(-1e9)
                continue

            mask = torch.full_like(logits, -torch.inf)
            mask[torch.LongTensor(valid)] = 0.0
            probs = torch.softmax(logits + mask, dim=-1)

            k_eff = min(self.top_k, len(valid))
            _, top_idx = torch.topk(probs, k=k_eff, dim=-1)
            if k_eff < self.top_k:
                pad = top_idx[-1].repeat(self.top_k - k_eff)
                top_idx = torch.cat([top_idx, pad], dim=0)

            idx_list = [int(x.item()) for x in top_idx]
            selected_hops.append(idx_list)

            hop_lp = float(torch.log(probs.clamp(eps, 1.0)[top_idx]).sum().item())
            hop_logps.append(hop_lp)

            dist = torch.distributions.Dirichlet(dir_alpha[0, d])
            alloc = dist.sample().clamp(eps, 1.0)
            alloc = alloc / alloc.sum()
            split_allocs.append([float(x.item()) for x in alloc])
            alloc_logps.append(float(dist.log_prob(alloc).item()))

        return selected_hops, split_allocs, hop_logps, alloc_logps

    def store(self, transition):
        self.memory.append(transition)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        (
            states,
            selected_hops,
            split_allocs,
            rewards_list,
            next_states,
            old_hop_logps,
            old_alloc_logps,
            dones,
        ) = zip(*batch)

        states_t = torch.FloatTensor(np.array(states))
        next_states_t = torch.FloatTensor(np.array(next_states))
        selected_hops_t = torch.LongTensor(np.array(selected_hops))
        split_allocs_t = torch.FloatTensor(np.array(split_allocs))
        rewards_t = torch.FloatTensor(np.array(rewards_list))
        old_hop_lp_t = torch.FloatTensor(np.array(old_hop_logps))
        old_alloc_lp_t = torch.FloatTensor(np.array(old_alloc_logps))
        dones_t = torch.FloatTensor(np.array(dones)).unsqueeze(1)

        hop_logits, dir_alpha = self.actor(states_t)

        actor_loss_total = 0.0
        critic_loss_total = 0.0
        eps = 1e-6
        batch_n = states_t.shape[0]

        for d in range(self.num_dests):
            dest_idx = torch.LongTensor([d] * batch_n)
            v = self.critic(states_t, dest_idx).squeeze()
            v_next = self.critic(next_states_t, dest_idx).squeeze()

            r = rewards_t[:, d]
            td = r + self.gamma * v_next * (1.0 - dones_t.squeeze())
            adv = (td - v).detach()
            critic_loss_total += nn.functional.mse_loss(v, td)

            probs_d = torch.softmax(hop_logits[:, d, :], dim=-1).clamp(eps, 1.0)
            idx = selected_hops_t[:, d, :]
            lp_new_hop = torch.log(probs_d.gather(1, idx)).sum(dim=1)
            ratio_hop = torch.exp(lp_new_hop - old_hop_lp_t[:, d])
            s1 = ratio_hop * adv
            s2 = torch.clamp(ratio_hop, 1 - self.eps_clip, 1 + self.eps_clip) * adv
            actor_loss_total += -torch.min(s1, s2).mean()

            dist = torch.distributions.Dirichlet(dir_alpha[:, d, :])
            alloc = split_allocs_t[:, d, :].clamp(eps, 1.0)
            alloc = alloc / alloc.sum(dim=1, keepdim=True)
            lp_new_alloc = dist.log_prob(alloc)
            ratio_alloc = torch.exp(lp_new_alloc - old_alloc_lp_t[:, d])
            s1a = ratio_alloc * adv
            s2a = torch.clamp(ratio_alloc, 1 - self.eps_clip, 1 + self.eps_clip) * adv
            actor_loss_total += -torch.min(s1a, s2a).mean()
            actor_loss_total -= self.entropy_coeff * dist.entropy().mean()

        actor_loss = actor_loss_total / (self.num_dests * 2)
        critic_loss = critic_loss_total / self.num_dests

        self.optimizerA.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.optimizerA.step()

        self.optimizerC.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.optimizerC.step()
