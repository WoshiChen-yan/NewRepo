import multiprocessing as mp
import queue
import time

import numpy as np

from net import Net
from ppoagent import SPLIT_WEIGHTS
from ppoagent_dual_timescale import PPOAgentDualTimeScale
from trend_shared_memory import TrendSharedMemory


def _compute_trend_from_snapshot(snapshot_3d):
    """
    snapshot_3d shape: [N, N, 3] where channels = [rssi, loss, latency]
    return trend matrix shape [N, N] in [0,1]
    """
    n = snapshot_3d.shape[0]
    trend = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            rssi, loss, latency = snapshot_3d[i, j]

            if np.isnan(rssi) or np.isnan(loss) or np.isnan(latency):
                trend[i, j] = 1.0
                continue

            rssi_risk = min(1.0, max(0.0, (-float(rssi) - 50.0) / 50.0))
            loss_risk = min(1.0, max(0.0, float(loss) / 100.0))
            latency_risk = min(1.0, max(0.0, float(latency) / 300.0))
            risk = 0.45 * rssi_risk + 0.35 * loss_risk + 0.20 * latency_risk
            trend[i, j] = float(min(1.0, max(0.0, risk)))

    return trend


def _slow_predictor_worker(shm_name, n_nodes, input_queue, stop_event):
    shm = TrendSharedMemory(name=shm_name, rows=n_nodes, cols=n_nodes, create=False)
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

            trend = _compute_trend_from_snapshot(snapshot)
            shm.write_matrix(trend, confidence=0.70)
    finally:
        shm.close()


class NetDualTimeScaleSHM(Net):
    """
    Multiprocessing + shared memory version.
    - Slow stream: separate process computes trend matrix and writes shared memory.
    - Fast stream: main process runs PPO/UCMP and reads latest trend in O(1).
    """

    def __init__(self, name=1, interval=1, t_slow=1.0, t_fast=0.01):
        super().__init__(name=name, interval=interval)

        self.T_slow = float(t_slow)
        self.T_fast = float(t_fast)

        self._shm_name = None
        self._trend_shm = None
        self._slow_queue = None
        self._slow_stop_event = None
        self._slow_process = None
        self._last_snapshot_submit_ts = 0.0

    def select_core_nodes(self, num_nodes_rate=0.3):
        """
        Reuse base node scoring/election, then replace core-node agents with
        PPOAgentDualTimeScale (shared critic remains unchanged).
        """
        super().select_core_nodes(num_nodes_rate=num_nodes_rate)

        if not self.core_nodes:
            return

        for core_node in self.core_nodes:
            name = core_node.name
            self.all_node_agents[name] = PPOAgentDualTimeScale(
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
            args=(self._shm_name, n, self._slow_queue, self._slow_stop_event),
            daemon=True,
        )
        self._slow_process.start()
        print(f"[SlowStream] process started: pid={self._slow_process.pid}, T_slow={self.T_slow}s")

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

        snapshot = self._build_network_snapshot()
        payload = {"timestamp": now, "snapshot": snapshot}

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

    def update_routing(self):
        self._submit_snapshot_if_due()
        return super().update_routing()

    def apply_node_routing(self, core_node, actions_matrix, split_ratios_matrix=None):
        trend_matrix, trend_ts, conf, epoch = self._read_trend()
        alpha, age = self._compute_trend_alpha(time.time(), trend_ts, conf)

        core_node.cmd("ip route flush all proto 188")
        src_index = self.nodes.index(core_node)

        for dest_index, actions in enumerate(actions_matrix):
            dest_node = self.nodes[dest_index]
            if dest_node == core_node:
                continue

            primary_hop_index = actions[0]
            backup_hop_index = actions[1]

            if primary_hop_index >= len(self.nodes) or backup_hop_index >= len(self.nodes):
                continue

            primary_hop_node = self.nodes[primary_hop_index]
            backup_hop_node = self.nodes[backup_hop_index]

            dest_ip = dest_node.ip.split('/')[0]
            primary_hop_ip = primary_hop_node.ip.split('/')[0]
            backup_hop_ip = backup_hop_node.ip.split('/')[0]
            core_ip = core_node.ip.split('/')[0]
            core_iface = core_node.name

            if primary_hop_ip == core_ip:
                continue

            if primary_hop_ip == dest_ip:
                cmd = f"ip route replace {dest_ip} dev wlan{core_iface} metric 10 proto 188"
                core_node.cmd(cmd)
                continue

            split_level = split_ratios_matrix[dest_index] if split_ratios_matrix else 0
            if trend_matrix is not None:
                risk = float(trend_matrix[src_index, dest_index])
                shift = int(round(alpha * risk * 2.0))
                split_level = int(min(4, max(0, split_level + shift)))

            w_primary, w_backup = SPLIT_WEIGHTS[split_level]

            can_split = (
                w_primary > 0 and w_backup > 0
                and primary_hop_ip != backup_hop_ip
                and backup_hop_ip != core_ip
                and backup_hop_ip != dest_ip
            )

            if can_split:
                cmd = (
                    f"ip route replace {dest_ip} proto 188 "
                    f"nexthop via {primary_hop_ip} weight {w_primary} "
                    f"nexthop via {backup_hop_ip} weight {w_backup}"
                )
            elif w_primary > 0:
                cmd = f"ip route replace {dest_ip} via {primary_hop_ip} metric 10 proto 188"
            else:
                cmd = f"ip route replace {dest_ip} via {backup_hop_ip} metric 10 proto 188"

            core_node.cmd(cmd)

        if trend_matrix is not None and epoch % 10 == 0:
            print(f"[FastStream] trend epoch={epoch}, age={age:.3f}s, alpha={alpha:.2f}")
