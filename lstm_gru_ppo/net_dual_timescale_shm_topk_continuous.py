import multiprocessing as mp
import queue
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from net import Net
from lstm_gru_ppo.ppoagent_topk_continuous import PPOAgentTopKContinuous
from lstm_gru_ppo.trend_shared_memory import TrendSharedMemory


class TrendLSTM(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y, _ = self.lstm(x)
        last = y[:, -1, :]
        return self.head(last).squeeze(-1)


def _normalize_snapshot(snapshot_3d):
    """Map raw [rssi, loss, latency] into [0,1] risk-space features."""
    snap = np.asarray(snapshot_3d, dtype=np.float32)
    out = np.zeros_like(snap, dtype=np.float32)

    rssi = snap[:, :, 0]
    loss = snap[:, :, 1]
    latency = snap[:, :, 2]

    out[:, :, 0] = np.clip((-rssi - 50.0) / 50.0, 0.0, 1.0)
    out[:, :, 1] = np.clip(loss / 100.0, 0.0, 1.0)
    out[:, :, 2] = np.clip(latency / 300.0, 0.0, 1.0)

    out[np.isnan(out)] = 1.0
    return out


def _build_lstm_input(snapshot_history, window_size):
    """Build tensor [N*N, T, 3] from history of snapshots."""
    if not snapshot_history:
        return None, 0

    seq = list(snapshot_history)[-window_size:]
    norm_seq = [_normalize_snapshot(x) for x in seq]
    n = norm_seq[-1].shape[0]
    t = len(norm_seq)

    x = np.zeros((n * n, t, 3), dtype=np.float32)
    idx = 0
    for i in range(n):
        for j in range(n):
            for k in range(t):
                x[idx, k, :] = norm_seq[k][i, j, :]
            idx += 1

    return torch.FloatTensor(x), n


def _compute_trend_from_snapshot(snapshot_history, model, window_size):
    """
    LSTM slow-stream inference interface.
    Input window T in [8,16], output [N,N] risk probability in [0,1].
    """
    x, n = _build_lstm_input(snapshot_history, window_size)
    if x is None:
        return None

    model.eval()
    with torch.no_grad():
        pred = model(x).cpu().numpy().reshape(n, n).astype(np.float32)

    pred = np.clip(pred, 0.0, 1.0)
    np.fill_diagonal(pred, 0.0)
    return pred


def _slow_predictor_worker(shm_name, n_nodes, input_queue, stop_event, lstm_window=12, lstm_ckpt_path=None):
    shm = TrendSharedMemory(name=shm_name, rows=n_nodes, cols=n_nodes, create=False)
    model = TrendLSTM(input_dim=3, hidden_dim=64, num_layers=1)

    if lstm_ckpt_path:
        try:
            ckpt = torch.load(lstm_ckpt_path, map_location="cpu")
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict, strict=False)
            print(f"[SlowStream-LSTM] loaded checkpoint: {lstm_ckpt_path}")
        except Exception as e:
            print(f"[SlowStream-LSTM] checkpoint load failed ({e}), using random initialized model")

    history = deque(maxlen=max(8, min(16, int(lstm_window))))

    try:
        while not stop_event.is_set():
            try:
                item = input_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None:
                continue

            snapshot = item.get("snapshot")
            if snapshot is None:
                continue

            history.append(np.asarray(snapshot, dtype=np.float32))
            trend = _compute_trend_from_snapshot(history, model, window_size=history.maxlen)
            if trend is None:
                continue

            conf = 0.6 + 0.3 * min(1.0, len(history) / float(history.maxlen))
            shm.write_matrix(trend, confidence=float(conf))
    finally:
        shm.close()


class NetDualTimeScaleSHMTopKContinuous(Net):
    def __init__(self, name=1, interval=1, t_slow=1.0, t_fast=0.01, top_k=3,
                 lstm_window=12, lstm_ckpt_path=None):
        super().__init__(name=name, interval=interval)
        self.T_slow = float(t_slow)
        self.T_fast = float(t_fast)
        self.top_k = int(max(2, top_k))
        self.lstm_window = int(max(8, min(16, lstm_window)))
        self.lstm_ckpt_path = lstm_ckpt_path

        self._shm_name = None
        self._trend_shm = None
        self._slow_queue = None
        self._slow_stop_event = None
        self._slow_process = None
        self._last_snapshot_submit_ts = 0.0

    def select_core_nodes(self, num_nodes_rate=0.3):
        super().select_core_nodes(num_nodes_rate=num_nodes_rate)
        if not self.core_nodes:
            return
        for core_node in self.core_nodes:
            name = core_node.name
            self.all_node_agents[name] = PPOAgentTopKContinuous(
                state_dim=self.state_dim,
                num_dests=self.num_dests,
                num_next_hops=self.num_next_hops,
                gru_hidden_dim=self.gru_hidden_dim,
                dest_embedding_dim=self.dest_embedding_dim,
                actor_lr=1e-4,
                critic_lr=1e-3,
                critic=self.global_critic,
                critic_optimizer=self.global_critic_optimizer,
                batch_size=self.batch_size,
                top_k=self.top_k,
            )
            if name not in self.reward_history:
                self.reward_history[name] = {}

    def start_slow_predictor(self):
        n = len(self.nodes)
        if n <= 0:
            raise RuntimeError("No nodes found; add nodes before starting slow predictor.")
        if self._slow_process is not None and self._slow_process.is_alive():
            return

        self._shm_name = f"trend_matrix_{self.name}_{int(time.time())}_{n}"
        self._trend_shm = TrendSharedMemory(name=self._shm_name, rows=n, cols=n, create=True)

        self._slow_queue = mp.Queue(maxsize=2)
        self._slow_stop_event = mp.Event()
        self._slow_process = mp.Process(
            target=_slow_predictor_worker,
            args=(
                self._shm_name,
                n,
                self._slow_queue,
                self._slow_stop_event,
                self.lstm_window,
                self.lstm_ckpt_path,
            ),
            daemon=True,
        )
        self._slow_process.start()
        print(
            f"[SlowStream-LSTM] process started: pid={self._slow_process.pid}, "
            f"T_slow={self.T_slow}s, window={self.lstm_window}"
        )

    def stop_slow_predictor(self):
        if self._slow_stop_event is not None:
            self._slow_stop_event.set()
        if self._slow_process is not None:
            self._slow_process.join(timeout=1.5)
            if self._slow_process.is_alive():
                self._slow_process.terminate()
            self._slow_process = None
        if self._slow_queue is not None:
            self._slow_queue.close()
            self._slow_queue = None
        if self._trend_shm is not None:
            try:
                self._trend_shm.close()
            finally:
                try:
                    self._trend_shm.unlink()
                except FileNotFoundError:
                    pass
            self._trend_shm = None
        print("[SlowStream] process stopped")

    def _build_network_snapshot(self):
        n = len(self.nodes)
        snapshot = np.full((n, n, 3), np.nan, dtype=np.float32)
        for i, src in enumerate(self.nodes):
            for j, dst in enumerate(self.nodes):
                if i == j:
                    snapshot[i, j, :] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    continue
                info = src.get_link_quality_by_mac(dst.mac)
                if not info:
                    continue
                snapshot[i, j, 0] = float(info.get("rssi", -100.0))
                snapshot[i, j, 1] = float(info.get("loss", 100.0))
                snapshot[i, j, 2] = float(info.get("latency", 1000.0))
        return snapshot

    def _submit_snapshot_if_due(self):
        if self._slow_queue is None:
            return
        now = time.time()
        if now - self._last_snapshot_submit_ts < self.T_slow:
            return
        payload = {"timestamp": now, "snapshot": self._build_network_snapshot()}
        try:
            self._slow_queue.put_nowait(payload)
            self._last_snapshot_submit_ts = now
        except queue.Full:
            pass

    def _read_trend(self):
        if self._trend_shm is None:
            return None, 0.0, 0.0, 0
        return self._trend_shm.read_matrix()

    def _compute_trend_alpha(self, now_ts, trend_ts, conf):
        age = now_ts - trend_ts
        freshness = max(0.0, 1.0 - age / max(2.0 * self.T_slow, 1e-6))
        alpha = min(1.0, max(0.0, freshness * conf))
        return alpha, age

    def _blend_alloc(self, alloc, risk, alpha):
        alloc = np.asarray(alloc, dtype=np.float32)
        beta = float(min(1.0, max(0.0, risk * alpha)))
        uni = np.full_like(alloc, 1.0 / len(alloc))
        out = (1.0 - beta) * alloc + beta * uni
        s = float(np.sum(out))
        return out / s if s > 1e-8 else uni

    def update_routing(self):
        self._submit_snapshot_if_due()

        all_actions = {}
        all_transitions = {}

        for node in self.core_nodes:
            agent = self.all_node_agents.get(node.name)
            if not agent:
                continue
            node_index = self.nodes.index(node)
            state_seq, neighbor_map_indices = self.get_state_sequence(node.name)
            hops, allocs, old_hop_lp, old_alloc_lp = agent.select_action(state_seq, node_index, neighbor_map_indices)
            all_transitions[node.name] = (state_seq, hops, allocs, old_hop_lp, old_alloc_lp)
            all_actions[node.name] = (hops, allocs)

        for node_name, (hops, allocs) in all_actions.items():
            self.apply_node_routing(self.node_dict[node_name], hops, allocs)

        for edge_node in self.edge_nodes:
            self.apply_default_routing(edge_node)

        time.sleep(0.1)
        self.test_all_links_concurrent()

        self.global_steps += 1
        current_step_core_rewards = []

        for node in self.core_nodes:
            agent = self.all_node_agents.get(node.name)
            if not agent or node.name not in all_transitions:
                continue

            state_seq, hops, allocs, old_hop_lp, old_alloc_lp = all_transitions[node.name]
            new_state_seq, _ = self.get_state_sequence(node.name)
            done = False

            rewards = []
            for dest_index in range(self.num_dests):
                dest_node = self.nodes[dest_index]
                if dest_node == node:
                    rewards.append(0.0)
                    continue
                info = node.get_link_quality_by_mac(dest_node.mac)
                reward = self._calculate_reward_from_link(info)
                rewards.append(reward)
                if dest_node.name not in self.reward_history[node.name]:
                    self.reward_history[node.name][dest_node.name] = []
                self.reward_history[node.name][dest_node.name].append(reward)

            agent.store((state_seq, hops, allocs, rewards, new_state_seq, old_hop_lp, old_alloc_lp, done))
            current_step_core_rewards.append(float(np.mean(rewards)))

        if self.global_steps > self.batch_size:
            for core_node in self.core_nodes:
                agent = self.all_node_agents.get(core_node.name)
                if agent:
                    agent.learn()

        if current_step_core_rewards:
            avg_reward = float(np.mean(current_step_core_rewards))
            self.avg_core_reward_history.append(avg_reward)

    def apply_node_routing(self, core_node, selected_hops_matrix, split_allocs_matrix=None):
        trend_matrix, trend_ts, conf, epoch = self._read_trend()
        alpha, age = self._compute_trend_alpha(time.time(), trend_ts, conf)

        core_node.cmd("ip route flush all proto 188")
        src_index = self.nodes.index(core_node)

        for dest_index, hops in enumerate(selected_hops_matrix):
            dest_node = self.nodes[dest_index]
            if dest_node == core_node:
                continue

            dest_ip = dest_node.ip.split('/')[0]
            core_ip = core_node.ip.split('/')[0]
            core_iface = core_node.name

            filt_hops = []
            for h in hops:
                if 0 <= h < len(self.nodes):
                    hop_ip = self.nodes[h].ip.split('/')[0]
                    if hop_ip != core_ip:
                        filt_hops.append(h)
            if not filt_hops:
                continue

            first_ip = self.nodes[filt_hops[0]].ip.split('/')[0]
            if first_ip == dest_ip:
                core_node.cmd(f"ip route replace {dest_ip} dev wlan{core_iface} metric 10 proto 188")
                continue

            alloc = np.array(split_allocs_matrix[dest_index] if split_allocs_matrix else [1.0], dtype=np.float32)
            if alloc.size != len(hops):
                alloc = np.full((len(hops),), 1.0 / max(1, len(hops)), dtype=np.float32)

            valid_alloc = []
            for i, h in enumerate(hops):
                if h in filt_hops:
                    valid_alloc.append(float(alloc[i]))
            if not valid_alloc:
                valid_alloc = [1.0]
            valid_alloc = np.asarray(valid_alloc, dtype=np.float32)
            valid_alloc = np.clip(valid_alloc, 1e-6, 1.0)
            valid_alloc = valid_alloc / np.sum(valid_alloc)

            if trend_matrix is not None:
                risk = float(trend_matrix[src_index, dest_index])
                valid_alloc = self._blend_alloc(valid_alloc, risk, alpha)

            cmd_parts = [f"ip route replace {dest_ip} proto 188"]
            for i, h in enumerate(filt_hops):
                hop_ip = self.nodes[h].ip.split('/')[0]
                if hop_ip == dest_ip and i > 0:
                    continue
                w = int(round(float(valid_alloc[i]) * 100.0))
                w = min(100, max(1, w))
                cmd_parts.append(f"nexthop via {hop_ip} weight {w}")

            if len(cmd_parts) <= 1:
                cmd = f"ip route replace {dest_ip} via {first_ip} metric 10 proto 188"
            else:
                cmd = " ".join(cmd_parts)
            core_node.cmd(cmd)

        if trend_matrix is not None and epoch % 10 == 0:
            print(f"[FastStream-TopK] trend epoch={epoch}, age={age:.3f}s, alpha={alpha:.2f}")
