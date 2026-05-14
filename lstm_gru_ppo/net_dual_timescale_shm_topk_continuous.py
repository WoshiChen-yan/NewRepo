
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
    """趋势预测LSTM模型 - 用于预测网络链路风险概率"""
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=1):
        super().__init__()
        # LSTM层：输入维度3(RSSI, loss, latency)，隐藏层维度64
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        # 输出头：将LSTM输出映射到[0,1]概率区间
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # 输出风险概率
        )

    def forward(self, x):
        """前向传播：输入形状 [batch, seq_len, 3]，输出形状 [batch]"""
        y, _ = self.lstm(x)  # 获取LSTM所有时间步输出
        last = y[:, -1, :]   # 取最后一个时间步的隐藏状态
        return self.head(last).squeeze(-1)  # 通过输出头得到风险概率


def _normalize_snapshot(snapshot_3d):
    """将原始链路质量数据 [rssi, loss, latency] 归一化到 [0,1] 风险空间"""
    snap = np.asarray(snapshot_3d, dtype=np.float32)
    out = np.zeros_like(snap, dtype=np.float32)

    # 提取三个特征维度
    rssi = snap[:, :, 0]      # 接收信号强度
    loss = snap[:, :, 1]      # 丢包率
    latency = snap[:, :, 2]   # 延迟

    # RSSI归一化：-100dBm→1(最差), -50dBm→0(最好)
    out[:, :, 0] = np.clip((-rssi - 50.0) / 50.0, 0.0, 1.0)
    # 丢包率归一化：0%→0, 100%→1
    out[:, :, 1] = np.clip(loss / 100.0, 0.0, 1.0)
    # 延迟归一化：0ms→0, 300ms→1
    out[:, :, 2] = np.clip(latency / 300.0, 0.0, 1.0)

    # 将NaN值设为1(最高风险)
    out[np.isnan(out)] = 1.0
    return out


def _build_lstm_input(snapshot_history, window_size):
    """从快照历史构建LSTM输入张量 [N*N, T, 3]"""
    if not snapshot_history:
        return None, 0

    # 取最近window_size个快照
    seq = list(snapshot_history)[-window_size:]
    # 对每个快照进行归一化
    norm_seq = [_normalize_snapshot(x) for x in seq]
    n = norm_seq[-1].shape[0]  # 节点数量
    t = len(norm_seq)          # 时间步数量

    # 将[N,N,T,3]结构转换为[N*N,T,3]：每个链路作为独立样本
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
    LSTM慢流推理接口
    输入：时间窗口T ∈ [8,16]，输出：[N,N]风险概率矩阵 ∈ [0,1]
    """
    x, n = _build_lstm_input(snapshot_history, window_size)
    if x is None:
        return None

    # 模型评估模式，禁用梯度计算
    model.eval()
    with torch.no_grad():
        pred = model(x).cpu().numpy().reshape(n, n).astype(np.float32)

    # 裁剪输出到[0,1]范围
    pred = np.clip(pred, 0.0, 1.0)
    # 对角线(自身到自身)风险设为0
    np.fill_diagonal(pred, 0.0)
    return pred


def _slow_predictor_worker(shm_name, n_nodes, input_queue, stop_event, lstm_window=12, lstm_ckpt_path=None):
    """
    LSTM慢流预测器工作进程
    独立进程运行，负责周期性预测网络链路风险趋势
    """
    # 连接到已创建的共享内存
    shm = TrendSharedMemory(name=shm_name, rows=n_nodes, cols=n_nodes, create=False)
    # 初始化趋势预测LSTM模型
    model = TrendLSTM(input_dim=3, hidden_dim=64, num_layers=1)

    # 加载预训练模型权重（如果提供）
    if lstm_ckpt_path:
        try:
            ckpt = torch.load(lstm_ckpt_path, map_location="cpu")
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict, strict=False)
            print(f"[SlowStream-LSTM] 加载预训练模型: {lstm_ckpt_path}")
        except Exception as e:
            print(f"[SlowStream-LSTM] 模型加载失败 ({e}), 使用随机初始化模型")

    # 维护滑动窗口历史记录，窗口大小限制在[8,16]
    history = deque(maxlen=max(8, min(16, int(lstm_window))))

    try:
        # 主循环：持续处理快照直到收到停止信号
        while not stop_event.is_set():
            try:
                # 从队列获取新快照（超时0.2秒）
                item = input_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None:
                continue

            snapshot = item.get("snapshot")
            if snapshot is None:
                continue

            # 添加到历史记录
            history.append(np.asarray(snapshot, dtype=np.float32))
            # 计算趋势矩阵
            trend = _compute_trend_from_snapshot(history, model, window_size=history.maxlen)
            if trend is None:
                continue

            # 计算置信度：随着历史数据增多，置信度从0.6增加到0.9
            conf = 0.6 + 0.3 * min(1.0, len(history) / float(history.maxlen))
            # 将趋势矩阵写入共享内存
            shm.write_matrix(trend, confidence=float(conf))
    finally:
        # 清理共享内存连接
        shm.close()


class NetDualTimeScaleSHMTopKContinuous(Net):
    """
    双时间尺度共享内存Top-K连续路由网络
    结合LSTM慢流趋势预测和PPO快速路由决策的混合架构
    """
    def __init__(self, name=1, interval=1, t_slow=5.0, t_fast=0.5, top_k=3,
                 lstm_window=12, lstm_ckpt_path=None):
        super().__init__(name=name, interval=interval)
        # 慢流时间周期（LSTM趋势更新间隔）
        self.T_slow = float(t_slow)
        # 快流时间周期（路由决策更新间隔）
        self.T_fast = float(t_fast)
        # Top-K策略：每个目的节点选择的下一跳数量
        self.top_k = int(max(2, top_k))
        # LSTM输入窗口大小，限制在[8,16]
        self.lstm_window = int(max(8, min(16, lstm_window)))
        # LSTM预训练模型路径
        self.lstm_ckpt_path = lstm_ckpt_path

        # 共享内存相关成员变量
        self._shm_name = None           # 共享内存名称
        self._trend_shm = None          # 趋势共享内存对象
        self._slow_queue = None         # 慢流队列（传递快照）
        self._slow_stop_event = None    # 慢流进程停止信号
        self._slow_process = None       # 慢流进程对象
        self._last_snapshot_submit_ts = 0.0  # 上次提交快照的时间戳

    def select_core_nodes(self, num_nodes_rate=0.3):
        """选择核心节点并为每个核心节点创建PPO智能体"""
        super().select_core_nodes(num_nodes_rate=num_nodes_rate)
        if not self.core_nodes:
            return
        
        # 为每个核心节点创建Top-K连续动作空间的PPO智能体
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
            # 初始化奖励历史记录
            if name not in self.reward_history:
                self.reward_history[name] = {}

    def start_slow_predictor(self):
        """启动LSTM慢流预测器进程"""
        n = len(self.nodes)
        if n <= 0:
            raise RuntimeError("未找到节点；请在启动慢预测器前添加节点。")
        # 如果慢流进程已在运行，则直接返回
        if self._slow_process is not None and self._slow_process.is_alive():
            return

        # 创建唯一的共享内存名称
        self._shm_name = f"trend_matrix_{self.name}_{int(time.time())}_{n}"
        # 创建趋势共享内存（N×N矩阵）
        self._trend_shm = TrendSharedMemory(name=self._shm_name, rows=n, cols=n, create=True)

        # 创建进程间通信队列
        self._slow_queue = mp.Queue(maxsize=2)
        # 创建停止事件
        self._slow_stop_event = mp.Event()
        # 创建并启动慢流预测进程
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
            f"[SlowStream-LSTM] 进程启动: pid={self._slow_process.pid}, "
            f"T_slow={self.T_slow}s, window={self.lstm_window}"
        )

    def stop_slow_predictor(self):
        """停止并清理慢流预测器进程"""
        # 设置停止事件
        if self._slow_stop_event is not None:
            self._slow_stop_event.set()
        # 等待进程结束
        if self._slow_process is not None:
            self._slow_process.join(timeout=1.5)
            # 如果超时未结束则强制终止
            if self._slow_process.is_alive():
                self._slow_process.terminate()
            self._slow_process = None
        # 关闭队列
        if self._slow_queue is not None:
            self._slow_queue.close()
            self._slow_queue = None
        # 关闭并删除共享内存
        if self._trend_shm is not None:
            try:
                self._trend_shm.close()
            finally:
                try:
                    self._trend_shm.unlink()
                except FileNotFoundError:
                    pass
            self._trend_shm = None
        print("[SlowStream] 进程已停止")

    def _build_network_snapshot(self):
        """构建网络链路质量快照 [N, N, 3]"""
        n = len(self.nodes)
        # 初始化快照矩阵，默认值为NaN
        snapshot = np.full((n, n, 3), np.nan, dtype=np.float32)
        
        # 遍历所有源-目的节点对
        for i, src in enumerate(self.nodes):
            for j, dst in enumerate(self.nodes):
                # 自身到自身的链路设为理想状态
                if i == j:
                    snapshot[i, j, :] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    continue
                # 获取链路质量信息
                info = src.get_link_quality_by_mac(dst.mac)
                if not info:
                    continue
                # 填充RSSI、丢包率、延迟
                snapshot[i, j, 0] = float(info.get("rssi", -100.0))
                snapshot[i, j, 1] = float(info.get("loss", 100.0))
                snapshot[i, j, 2] = float(info.get("latency", 1000.0))
        return snapshot

    def _submit_snapshot_if_due(self):
        """如果达到慢流周期，则提交网络快照到慢流预测器"""
        if self._slow_queue is None:
            return
        now = time.time()
        # 检查是否达到慢流时间周期
        if now - self._last_snapshot_submit_ts < self.T_slow:
            return
        # 构建快照并提交到队列
        payload = {"timestamp": now, "snapshot": self._build_network_snapshot()}
        try:
            self._slow_queue.put_nowait(payload)
            self._last_snapshot_submit_ts = now
        except queue.Full:
            # 队列满时丢弃当前快照
            pass

    def _read_trend(self):
        """从共享内存读取趋势矩阵"""
        if self._trend_shm is None:
            return None, 0.0, 0.0, 0
        return self._trend_shm.read_matrix()

    def _compute_trend_alpha(self, now_ts, trend_ts, conf):
        """计算趋势权重因子alpha
        alpha = 新鲜度 × 置信度
        新鲜度随时间衰减，在2*T_slow时间后衰减至0
        """
        age = now_ts - trend_ts  # 趋势矩阵的年龄
        # 新鲜度：随着时间推移从1衰减到0
        freshness = max(0.0, 1.0 - age / max(2.0 * self.T_slow, 1e-6))
        # 最终权重 = 新鲜度 × 置信度，裁剪到[0,1]
        alpha = min(1.0, max(0.0, freshness * conf))
        return alpha, age

    def _blend_alloc(self, alloc, risk, alpha):
        """根据风险和alpha混合分配策略
        高风险时趋向均匀分配，低风险时保持原分配
        """
        alloc = np.asarray(alloc, dtype=np.float32)
        # beta = 风险 × alpha，控制混合比例
        beta = float(min(1.0, max(0.0, risk * alpha)))
        # 均匀分配向量
        uni = np.full_like(alloc, 1.0 / len(alloc))
        # 混合：(1-beta)*原分配 + beta*均匀分配
        out = (1.0 - beta) * alloc + beta * uni
        # 归一化
        s = float(np.sum(out))
        return out / s if s > 1e-8 else uni

    def update_routing(self):
        """
        主路由更新循环（快流）
        执行步骤：
        1. 检查并提交快照到慢流预测器
        2. 收集所有核心节点的路由决策
        3. 应用路由配置到实际网络
        4. 收集奖励并训练PPO智能体
        """
        # [Future TODO - 设计思路]
        # - 添加路由抖动保护：当链路变化超过阈值时阻止更新
        # - 添加紧急回退策略：当趋势新鲜度过旧时使用默认路由
        # - 添加按目的节点/流量类别的权重调整
        
        # 检查是否需要提交快照到慢流预测器
        self._submit_snapshot_if_due()

        all_actions = {}     # 存储所有节点的路由动作
        all_transitions = {} # 存储所有转换用于训练

        # 第一阶段：收集所有核心节点的动作
        for node in self.core_nodes:
            agent = self.all_node_agents.get(node.name)
            if not agent:
                continue
            node_index = self.nodes.index(node)
            # 获取节点状态序列
            state_seq, neighbor_map_indices = self.get_state_sequence(node.name)
            # PPO智能体选择动作（Top-K下一跳和分配比例）
            hops, allocs, old_hop_lp, old_alloc_lp = agent.select_action(state_seq, node_index, neighbor_map_indices)
            all_transitions[node.name] = (state_seq, hops, allocs, old_hop_lp, old_alloc_lp)
            all_actions[node.name] = (hops, allocs)

        # 第二阶段：应用路由到核心节点
        for node_name, (hops, allocs) in all_actions.items():
            self.apply_node_routing(self.node_dict[node_name], hops, allocs)

        # 应用默认路由到边缘节点
        for edge_node in self.edge_nodes:
            self.apply_default_routing(edge_node)

        # 等待路由生效并测试所有链路
        time.sleep(0.1)
        self.test_all_links_concurrent()

        # 更新全局步数
        self.global_steps += 1
        current_step_core_rewards = []

        # 第三阶段：收集奖励并存储转换
        for node in self.core_nodes:
            agent = self.all_node_agents.get(node.name)
            if not agent or node.name not in all_transitions:
                continue

            state_seq, hops, allocs, old_hop_lp, old_alloc_lp = all_transitions[node.name]
            new_state_seq, _ = self.get_state_sequence(node.name)
            done = False

            rewards = []
            # 计算每个目的节点的奖励
            for dest_index in range(self.num_dests):
                dest_node = self.nodes[dest_index]
                if dest_node == node:
                    rewards.append(0.0)  # 自身无奖励
                    continue
                info = node.get_link_quality_by_mac(dest_node.mac)
                reward = self._calculate_reward_from_link(info)
                rewards.append(reward)
                # 记录奖励历史
                if dest_node.name not in self.reward_history[node.name]:
                    self.reward_history[node.name][dest_node.name] = []
                self.reward_history[node.name][dest_node.name].append(reward)

            # 存储转换到经验回放缓冲区
            agent.store((state_seq, hops, allocs, rewards, new_state_seq, old_hop_lp, old_alloc_lp, done))
            current_step_core_rewards.append(float(np.mean(rewards)))

        # 第四阶段：训练PPO智能体（当收集足够样本后）
        if self.global_steps > self.batch_size:
            for core_node in self.core_nodes:
                agent = self.all_node_agents.get(core_node.name)
                if agent:
                    agent.learn()

        # 记录平均奖励
        if current_step_core_rewards:
            avg_reward = float(np.mean(current_step_core_rewards))
            self.avg_core_reward_history.append(avg_reward)

    def apply_node_routing(self, core_node, selected_hops_matrix, split_allocs_matrix=None):
        """
        将路由决策应用到核心节点
        结合趋势预测结果动态调整路由分配
        """
        # 读取趋势矩阵和相关元数据
        trend_matrix, trend_ts, conf, epoch = self._read_trend()
        # 计算趋势权重因子alpha
        alpha, age = self._compute_trend_alpha(time.time(), trend_ts, conf)

        # 清空该节点所有proto 188的路由
        core_node.cmd("ip route flush all proto 188")
        src_index = self.nodes.index(core_node)

        # 遍历所有目的节点
        for dest_index, hops in enumerate(selected_hops_matrix):
            dest_node = self.nodes[dest_index]
            if dest_node == core_node:
                continue  # 跳过自身

            dest_ip = dest_node.ip.split('/')[0]
            core_ip = core_node.ip.split('/')[0]
            core_iface = core_node.name

            # 过滤有效下一跳（排除自身）
            filt_hops = []
            for h in hops:
                if 0 <= h < len(self.nodes):
                    hop_ip = self.nodes[h].ip.split('/')[0]
                    if hop_ip != core_ip:
                        filt_hops.append(h)
            if not filt_hops:
                continue

            # 如果直接连接到目的节点，使用直连路由
            first_ip = self.nodes[filt_hops[0]].ip.split('/')[0]
            if first_ip == dest_ip:
                core_node.cmd(f"ip route replace {dest_ip} dev wlan{core_iface} metric 10 proto 188")
                continue

            # 获取分配比例
            alloc = np.array(split_allocs_matrix[dest_index] if split_allocs_matrix else [1.0], dtype=np.float32)
            if alloc.size != len(hops):
                alloc = np.full((len(hops),), 1.0 / max(1, len(hops)), dtype=np.float32)

            # 过滤有效分配比例
            valid_alloc = []
            for i, h in enumerate(hops):
                if h in filt_hops:
                    valid_alloc.append(float(alloc[i]))
            if not valid_alloc:
                valid_alloc = [1.0]
            valid_alloc = np.asarray(valid_alloc, dtype=np.float32)
            valid_alloc = np.clip(valid_alloc, 1e-6, 1.0)
            valid_alloc = valid_alloc / np.sum(valid_alloc)  # 归一化

            # 如果有趋势矩阵，根据风险调整分配
            if trend_matrix is not None:
                risk = float(trend_matrix[src_index, dest_index])
                valid_alloc = self._blend_alloc(valid_alloc, risk, alpha)

            # 构建ip route命令
            cmd_parts = [f"ip route replace {dest_ip} proto 188"]
            for i, h in enumerate(filt_hops):
                hop_ip = self.nodes[h].ip.split('/')[0]
                if hop_ip == dest_ip and i > 0:
                    continue  # 避免重复指定目的节点
                w = int(round(float(valid_alloc[i]) * 100.0))
                w = min(100, max(1, w))  # 权重限制在[1,100]
                cmd_parts.append(f"nexthop via {hop_ip} weight {w}")

            # 执行路由命令
            if len(cmd_parts) <= 1:
                cmd = f"ip route replace {dest_ip} via {first_ip} metric 10 proto 188"
            else:
                cmd = " ".join(cmd_parts)
            core_node.cmd(cmd)

        # 定期打印趋势信息
        if trend_matrix is not None and epoch % 10 == 0:
            print(f"[FastStream-TopK] 趋势epoch={epoch}, 时间指数={age:.3f}s, alpha={alpha:.2f}")