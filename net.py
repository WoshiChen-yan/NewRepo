import matplotlib.backends
import networkx as nx
import numpy as np
import matplotlib
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
from node import Node 
import time
import subprocess
from module import Mac80211Hwsim
from wmediumdConnector import w_server
from ppoagent import PPOAgent , PPOCritic # <--- MODIFIED: 导入 PPOAgent 和 PPOCritic
import matplotlib.pyplot as plt
import threading
import torch
import math
from collections import deque # <--- MODIFIED: 导入 deque
import random # <--- MODIFIED: 导入 random 用于 learn

class Net:
    
    _used_cfg_ids=set()
    _next_cfg_id = 1
    _next_node_id = 1

    def __init__(self,name=1,interval=1):
         self.name=name
         self.current_node_id=0
         self.interval=interval #default interval=1
         self.nodes=[]#用于存储节点
         
         
         self.core_nodes=[]#用于存储核心节点
         self.node_agents={}#用于存储核心节点的智能体
         
         # <--- MODIFIED: 为 GRU 添加状态历史管理 ---_>
         self.node_states={} # 存储节点 {name: state_vector} 的最新状态
         self.state_history = {} # 存储 {name: deque([state_vector, ...], maxlen=seq_len)}
         self.seq_len = 5 # <--- MODIFIED: 定义GRU要看的时间序列长度
         self.state_dim = 0 # <--- MODIFIED: 将在 select_core_nodes 中设置
         self.num_dests = 0 # <--- MODIFIED: 将在 select_core_nodes 中设置
         self.num_next_hops = 0 # <--- MODIFIED: 将在 select_core_nodes 中设置
         self.gru_hidden_dim = 64 # <--- MODIFIED: GRU 隐藏层大小
         self.node_dict = {} # <--- MODIFIED: 方便按名称查找节点
         self.reward_history = {} # <--- MODIFIED: 存储奖励历史用于绘图
         # <--- MODIFIED 结束 ---_>

         self.graph=nx.Graph(name=self.name)
         self.dqn_initialized=False
         self.interval=1
         self.global_critic=None
         self.global_critic_optimizer=None
             
    def set_position(self,name,position):
        """设置网络位置"""
        for node in self.nodes:
            if node.name==name:
                node.set_position(position)
                print(f"节点'{node.name}'位置已设置为: {node.position}")
                
                print(f"节点'{node.name}'位置已设置为: {node.position}")
                break
        print(f"网络位置已设置为: {self.position}")
    
    
    def setup_ovs_for_nodes(self):
        base_port=6633
        for node in self.nodes:
            port=base_port + 1
            node.set_ovsbridge(base_port) 
        
    def start_core_controller(self):
        
        self.controllers={}
        base_port=6633
        for i, core_node in enumerate(self.core_nodes):
            port = base_port + i
            cmd=f"ryu manager core_controller.py --ofp-tcp-listen-port {port}"
            proc=subprocess.Popen(cmd.split(),stdout=subprocess.PIPE)
            self.controllers[core_node.name]=proc
            print(f"核心节点 {core_node.name} 的控制器已启动，监听端口: {port}")
        
                       
    def cleanup_interfaces(self):
        """清理所有虚拟无线接口"""
        try:
            # 卸载现有的 mac80211_hwsim 模块
            subprocess.run(['sudo', 'rmmod', 'mac80211_hwsim'], 
                        stderr=subprocess.DEVNULL)
            time.sleep(1)
            
        except Exception as e:
            print(f"Error cleaning up interfaces: {e}")
    
    @classmethod
    def get_next_node_id(cls):
        """Get the next node ID"""
        node_id = cls._next_node_id
        cls._next_node_id += 1
        return node_id
            
    def add_node(self,name,mac,ip,position=(0,0,0),direction=(0,0,0)):
        
     
        if any(node.name==name for node in self.nodes):
            print(f"节点'{name}'已存在，无法添加")
            return
        
        if any(node.mac==mac for node in self.nodes):
            print(f"节点'{mac}'已存在，无法添加")
            return  
        
        if any(node.ip==ip for node in self.nodes):
            print(f"节点'{ip}'已存在，无法添加")
            return
        
        
        node_id=self.get_next_node_id()   
        ##调用类内部方法获取id
        self.current_node_id+=1
        
        node=Node(id=node_id,name=name,mac=mac,ip=ip,position=position,direction=direction)
        self.nodes.append(node)
        self.graph.add_node(
            node.name,
            position=node.position,
            direction=node.direction,
            txpower=node.txpower
            )
        
    @classmethod
    def reset_node_ids(cls):
        """Reset the node ID counter"""
        cls._next_node_id = 1
        
    def   start_network(self):
        for node in self.nodes:
            Mac80211Hwsim(on_the_fly=True,Node=node)
            node.addwlans()  
            # node.set_ovsbridge(6633)
        self.wmediumd_config()
        self.start_wmediumd_inbackground()
        time.sleep(1)
    
    def move_nodes(self):
        for node in self.nodes:
            x,y,z=node.position
            dx,dy,dz=node.direction
            
            new_x=x+dx*self.interval
            new_y=y+dy*self.interval
            new_z=z+dz*self.interval
            
            node.position=(new_x,new_y,new_z)
            self.graph.nodes[node.name]['position']=(new_x,new_y,new_z)
            print(f"节点'{node.name}'位置已更新为: {node.position}")
         
    
    # <--- MODIFIED: 重命名为 _get_current_state_vector，并使用 node.py 的新函数 ---_>
    def _get_current_state_vector(self, node):
        """获取节点 *当前* 的状态向量"""
        state_vector = []
        
        # 1. 添加链路质量信息
        # (我们假设 self.nodes 列表的顺序是固定的，用作目标/下一跳的索引)
        for target_node in self.nodes:
            if target_node == node:
                continue
                
            # <--- MODIFIED: 使用 node.py 中新添加的函数 ---_>
            link_info = node.get_link_quality_by_mac(target_node.mac) 
            
            if link_info:
                state_vector.extend([
                    link_info['distance']/1000.0,
                    link_info['latency']/100.0,
                    link_info['loss']/100.0,
                    link_info['rssi']/100.0,
                    link_info['doppler_shift']/1000.0
                ])
            else:
                # 如果没有数据 (不应该发生，但作为保险)
                state_vector.extend([9.9, 99.9, 100.0, -100.0, 0.0])

        # 2. (TODO) 添加队列状态
        # queue_status = node.get_queue_status() # 您需要
        # state.extend(queue_status)
        
        # 3. 存储最新的状态向量，供将来（例如，其他节点）查询
        self.node_states[node.name] = state_vector
        return state_vector

    # <--- MODIFIED: 新增函数，用于管理和返回 GRU 所需的状态序列 ---_>
    def get_state_sequence(self, node_name):
        """获取并更新节点的状态历史序列，用于GRU"""
        
        # 1. 获取当前时刻的最新状态
        current_state_vector = self._get_current_state_vector(self.node_dict[node_name])
        
        # 2. 查找或创建该节点的状态历史 (deque)
        if node_name not in self.state_history:
            # 创建一个固定长度的队列，并用 "零状态" 填充
            zero_state = [0.0] * self.state_dim
            self.state_history[node_name] = deque([zero_state] * self.seq_len, maxlen=self.seq_len)
            
        # 3. 将最新状态推入队列 (最旧的状态会被自动挤出)
        self.state_history[node_name].append(current_state_vector)
        
        # 4. 返回整个序列
        return np.array(self.state_history[node_name])


    def compute_node_reward(self, node):
        """计算指定节点的奖励"""
        # (这是一个简化的奖励。在实际中，您应该基于流量生成器 (项目3)
        # 测量的真实端到端延迟和吞吐量来计算奖励)
        reward = 0
        
        # 简单地使用本地链路质量的平均值
        link_qualities = []
        
        # <--- MODIFIED: 仅从最新的历史记录中计算奖励 ---_>
        current_time = time.time()
        for record in node.link_quality_history:
            # 只看最近 2*interval 秒内的记录
            if record['time'] > current_time - (self.interval * 2):
                 # 奖励 = 10 - (延迟(ms)/10 * 0.5 + 丢包(%) * 0.5)
                 # 目标: 延迟 < 10ms, 丢包 = 0%
                 q = 10.0 - (min(record['latency'], 100)/10.0 * 0.5 + min(record['loss'], 100)/100.0 * 0.5)
                 link_qualities.append(q)

        if not link_qualities:
            return -10.0 # 如果没有链路信息，给予惩罚
            
        return sum(link_qualities) / len(link_qualities)
    
    # <--- MODIFIED: 重写 update_routing 作为 PPO 的训练步骤 ---_>
    def update_routing(self):
        """
        为所有核心节点执行一个完整的 RL 步骤:
        获取状态 -> 选择动作 -> 执行动作 -> 获取新状态 -> 存储经验 -> 学习
        """
        
        # 1. (可选) 确保所有节点的链路信息都是最新的
        # self.test_all_links_concurrent() # test.py 中的循环已经调用了它

        # 2. 遍历所有核心节点智能体
        for core_node in self.core_nodes:
            agent = self.node_agents.get(core_node.name)
            if not agent:
                print(f"错误: 核心节点 {core_node.name} 没有找到智能体")
                continue

            # 3. 获取状态 (时序)
            state_seq = self.get_state_sequence(core_node.name)
            
            # 4. 智能体选择动作 (主/备路径)
            # actions: [num_dests, 2], logprobs: [num_dests, 2]
            actions, old_logprobs = agent.select_action(state_seq)
            
            # 5. 在环境中执行动作 (下发路由)
            self.apply_node_routing(core_node, actions)
            
            # 6. 获取执行动作后的新状态和奖励
            # (注意：在真实环境中，新状态和奖励应该在动作执行 *之后* 测量)
            # (为简单起见，我们假设移动和链路更新已经发生)
            new_state_seq = self.get_state_sequence(core_node.name)
            reward = self.compute_node_reward(core_node)
            done = False # 在持续任务中，done 始终为 False
            
            # <--- MODIFIED: 记录奖励 ---_>
            if core_node.name not in self.reward_history:
                self.reward_history[core_node.name] = []
            self.reward_history[core_node.name].append(reward)
            # <--- MODIFIED 结束 ---_>


            # 7. 存储经验
            # transition = (state, action, reward, next_state, old_logprob, done)
            agent.store((state_seq, actions, reward, new_state_seq, old_logprobs, done))
            
            # 8. 触发智能体学习
            # (PPO 标准: 收集N步后再学习)
            # 我们在 ppo_agent.py 中设置了 memory.clear()，
            # 所以这里每次调用 learn() 都是 on-policy 的 (使用刚收集到的1个样本)
            # 为了更高效，我们应该收集
            if len(agent.memory) >= 32: # 举例: 收集到32个经验后再学习
                agent.learn(batch_size=32)
        
             
    
    # <--- MODIFIED: 重写 apply_node_routing 以处理复杂的“主/备”动作 ---_>
    def apply_node_routing(self, core_node, actions):
        """
        应用核心节点的路由决策 (主/备路径)
        Args:
            core_node (Node): 做出决策的节点
            actions (list): 形状为 [num_dests, 2] 的动作索引列表
                            actions[i][0] = 目的地i 的 主路径 下一跳索引
                            actions[i][1] = 目的地i 的 备用路径 下一跳索引
        """
        
        # (这是一个简化的路由下发。在您的项目中，这应该调用 FRR 的 VTYSH 或 API)
        
        print(f"--- 正在更新节点 {core_node.name} 的路由表 ---")

        # 确保 self.nodes 列表作为索引是可用的
        # self.nodes 列表就是我们的 "目的地" 和 "下一跳" 的索引
        
        for dest_index in range(self.num_dests):
            dest_node = self.nodes[dest_index]
            
            # 跳过到自己的路由
            if dest_node == core_node:
                continue
                
            primary_hop_index = actions[dest_index][0]
            backup_hop_index = actions[dest_index][1]
            
            # 确保索引在范围内
            if primary_hop_index >= len(self.nodes) or backup_hop_index >= len(self.nodes):
                print(f"  到 {dest_node.name}: 索引越界，跳过")
                continue

            primary_hop_node = self.nodes[primary_hop_index]
            backup_hop_node = self.nodes[backup_hop_index]
            
            dest_ip = dest_node.ip.split('/')[0]
            primary_hop_ip = primary_hop_node.ip.split('/')[0]
            backup_hop_ip = backup_hop_node.ip.split('/')[0]

            # 下发主路由 (metric 10)
            # 使用 `ip route replace` 来原子性地替换或添加路由
            # `proto 188` 是一个示例，用于标记这些是AI生成的路由，方便清除
            cmd_primary = f"ip route replace {dest_ip} via {primary_hop_ip} metric 10 proto 188"
            core_node.cmd(cmd_primary)
            
            # 下发备用路由 (metric 20)
            # 确保主备路径不同
            if primary_hop_ip != backup_hop_ip:
                # `ip route add` 用于添加第二条 (备用) 路径
                cmd_backup = f"ip route add {dest_ip} via {backup_hop_ip} metric 20 proto 188"
                core_node.cmd(cmd_backup)
            
            # 打印调试信息 (只打印一两个作为示例)
            if dest_index == 0 or dest_index == 1:
                print(f"  到 {dest_node.name} ({dest_ip}):")
                print(f"    主路 -> {primary_hop_node.name} ({primary_hop_ip}) [metric 10]")
                if primary_hop_ip != backup_hop_ip:
                    print(f"    备路 -> {backup_hop_node.name} ({backup_hop_ip}) [metric 20]")
        
        # print(f"--- 节点 {core_node.name} 路由更新完毕 ---")
        

    def end_test(self):
        self.cleanup_interfaces()
        # 停止所有节点上的 iperf
        for node in self.nodes:
            node.cmd("pkill iperf")
        
        for node in self.nodes:
            node.remove_node()
        print(f"网络'{self.name}'已经结束测试")
        self.reset_node_ids()
        
        
    def stop_rm_specific(self,name):
        for node in self.nodes:
            if node.name==name:
                node.remove_node()
        
    def generate_nodes(self, num_nodes, pattern="random", base_position=(0,0,10), spacing=3):
        """
        Automatically generate and add nodes to the network with different position patterns
        Args:
            num_nodes: Number of nodes to generate
            pattern: Position pattern ("grid", "circle", "random", "line")
            base_position: Starting position (x,y,z)
            spacing: Space between nodes
        """
        for i in range(num_nodes):
            # Generate basic node properties
            name = f"{i+11}"  # Names start from 11
            mac = f"02:00:00:00:{i+1:02d}:00"  # MAC addresses
            ip = f"10.10.10.{i+1}/24"  # IP addresses
            
            # Calculate position based on pattern
            if pattern == "grid":
                x = base_position[0] + (i % 3) * spacing
                y = base_position[1] + (i // 3) * spacing
                z = base_position[2] +  i
                position = (x, y, z)
                # Direction pointing to center
                center = (x + spacing/2, y + spacing/2, z)
                direction = self._calculate_direction(position, center)
                
            elif pattern == "circle":
                angle = (2 * np.pi * i) / num_nodes
                x = base_position[0] + spacing * np.cos(angle)
                y = base_position[1] + spacing * np.sin(angle)
                z = base_position[2] +  i
                position = (x, y, z)
                # Direction pointing to circle center
                direction = self._calculate_direction(position, base_position)
                
            elif pattern == "line":
                x = base_position[0] + i * spacing
                y = base_position[1] + i * spacing
                z = base_position[2] +  i
                position = (x, y, z)
                # Direction along the line
                direction = (1, 0, 0)
                
            elif pattern == "random":
                x = base_position[0] + np.random.uniform(-spacing, spacing)
                y = base_position[1] + np.random.uniform(-spacing, spacing)
                z = base_position[2] + np.random.uniform(-spacing, spacing)
                position = (x, y, z)
                # Random direction
                direction = self._normalize_vector(np.random.uniform(-1, 1, 3))
            
            # Add node to network
            self.add_node(name, mac, ip, position, direction)
            print(f"Added node {name} at position {position} with direction {direction}")

    def _calculate_direction(self, from_pos, to_pos):
        """Calculate normalized direction vector from one position to another"""
        direction = np.array(to_pos) - np.array(from_pos)
        return tuple(self._normalize_vector(direction))

    def _normalize_vector(self, vector):
        """Normalize a vector to unit length"""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return tuple(np.zeros_like(vector))
        return tuple(vector / norm)
    
    def yawmd_config(self, output_path="test.cfg", append=False):
        # ... (此函数无需修改) ...
        pass
            
            
    def wmediumd_config(self):
        # ... (此函数无需修改) ...
        pass
    
    
    def test_connectivity(self, count=10, timeout=1):
        # ... (此函数无需修改) ...
        pass


    # <--- MODIFIED: 取消注释 select_core_nodes 并实现 PPOAgent 的创建 ---_>
    def select_core_nodes(self, num_nodes_rate=0.3):
        """
        选择核心节点，并初始化共享的 Critic 和所有 PPO 智能体
        """
        print("\n=== 初始化核心节点和 PPO 智能体 ===")
        
        # 1. 选择核心节点 (使用您原来的邻居/RSSI逻辑)
        node_scores = {}
        for node in self.nodes:
            # (确保节点在选择前已经有邻居信息)
            # (为简单起见，我们假设所有节点都是核心节点)
            # neighbors = node.get_neighbor() 
            # ... (您原来的打分逻辑) ...
            node_scores[node.name] = 1 # 简化：所有节点得分相同
            
        num_core = max(1, int(len(self.nodes) * num_nodes_rate))
        # <--- MODIFIED: 简化 - 选择所有节点作为核心节点进行训练 ---_>
        num_core = len(self.nodes) 
        
        core_nodes_sorted = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)[:num_core]
        
        self.core_nodes = [
            node for node in self.nodes
            if node.name in [name for name, _ in core_nodes_sorted]
        ]
        
        # <--- MODIFIED: 创建一个 节点名 -> 节点对象 的映射，方便查找 ---_>
        self.node_dict = {node.name: node for node in self.nodes}

        print(f"已选择 {len(self.core_nodes)} 个核心节点: {[node.name for node in self.core_nodes]}")

        # 2. 定义状态和动作空间的维度
        # 状态维度 = (节点数 - 1) * 5 (distance, latency, loss, rssi, doppler)
        self.state_dim = (len(self.nodes) - 1) * 5 
        # (如果添加了队列状态，这里需要增加维度)
        
        # 动作空间
        self.num_dests = len(self.nodes) # 目标是所有其他节点
        self.num_next_hops = len(self.nodes) # 下一跳可以是任何节点

        if self.state_dim <= 0:
            print("错误：状态维度为0，无法创建智能体。")
            return

        print(f"智能体参数: state_dim={self.state_dim}, num_dests={self.num_dests}, num_next_hops={self.num_next_hops}")

        # 3. 创建 *共享的* Critic 和其优化器
        print("创建共享 Critic...")
        self.global_critic = PPOCritic(self.state_dim, self.gru_hidden_dim)
        self.global_critic_optimizer = torch.optim.Adam(self.global_critic.parameters(), lr=1e-3)

        # 4. 为每个核心节点创建 PPOAgent (独立 Actor)
        for core_node in self.core_nodes:
            name = core_node.name
            if name not in self.node_agents:
                agent = PPOAgent(
                    state_dim=self.state_dim,
                    num_dests=self.num_dests,
                    num_next_hops=self.num_next_hops,
                    gru_hidden_dim=self.gru_hidden_dim,
                    actor_lr=1e-4,
                    critic_lr=1e-3,
                    critic=self.global_critic, # <--- 传入共享 Critic
                    critic_optimizer=self.global_critic_optimizer # <--- 传入共享优化器
                )
                self.node_agents[name] = agent
                print(f"已为节点 {name} 创建 PPOAgent (Actor)")
         
                    
    def validate_agent_performance(self, episodes=3):
        # ... (此函数无需修改, 但请注意它仍在调用旧的 apply_node_routing) ...
        # ... (您可能需要更新此函数以反映新的动作空间) ...
        pass

    def _get_current_path(self, src, dst):
        # ... (此函数无需修改) ...
        pass
        
    def start_wmediumd_inbackground(self, config_path='test.cfg'):
        """
        启动wmediumd服务器
        Args:
            config_path: wmediumd配置文件路径
        """
        try:
            # Start the process
            cmd= f"sudo wmediumd -c {config_path} "
            result = subprocess.Popen(
                cmd.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            print(f"Started wmediumd with PID: {result.pid}")
            time.sleep(3)
            
            if result.poll() is None:
                print("wmediumd is running in the background")
            else:
                print("wmediumd failed to keep running")
                raise Exception("wmediumd failed to keep running")
        except Exception as e:
            print(f"Error starting wmediumd: {e}")
            if 'result' in locals():
                result.kill()
                result.communicate()
            exit(1)
            
        # w_server.connect()
            
    def compute_path_loss(self,distance, freq_ghz=2.4):
        """自由空间路径损耗（dB）"""
        if distance < 1:
            distance = 1
        c = 3e8  # 光速
        freq = freq_ghz * 1e9
        loss = 20 * math.log10(distance) + 20 * math.log10(freq) - 147.55
        return loss


    def compute_rssi(self,tx_power_dbm, path_loss_db):
        """接收信号强度（dBm）"""
        return tx_power_dbm - path_loss_db


    def compute_snr(self,rssi_dbm, noise_dbm=-90):
        """信噪比（dB）"""
        return rssi_dbm - noise_dbm


    def compute_bandwidth(self,src, dst, link_info):
        """
        用Shannon公式估算最大带宽（单位：Mbit）
        """
        distance = link_info.get('distance', 1)
        tx_power = link_info.get('tx_power', 15)  # dBm
        freq = link_info.get('freq', 2.4)         # GHz
        noise = link_info.get('noise', -90)       # dBm
        bandwidth = link_info.get('channel_bw', 20)  # MHz

        path_loss = self.compute_path_loss(distance, freq)
        rssi = self.compute_rssi(tx_power, path_loss)
        snr = self.compute_snr(rssi, noise)
        snr_linear = 10 ** (snr / 10)
        # Shannon capacity: C = B * log2(1 + SNR)
        cap = bandwidth * 1e6 * math.log2(1 + snr_linear)  # 单位：bps
        cap_mbit = cap / 1e6
        return max(1, int(cap_mbit))


    def compute_delay(self,src, dst, link_info):
        """
        延迟 = 传播延迟 + 排队/处理延迟
        传播延迟 = 距离 / 光速（单位：ms）
        """
        distance = link_info.get('distance', 1)
        base_delay = 10  # ms，基础排队/处理延迟
        propagation_delay = distance / 3e8 * 1e3  # ms
        return int(base_delay + propagation_delay)


    def compute_loss(self,src, dst, link_info):
        """
        丢包率与SNR相关，SNR越低丢包越高
        这里用经验映射：SNR<5dB丢包30%，5~15dB线性下降，>15dB丢包1%
        """
        distance = link_info.get('distance', 1)
        tx_power = link_info.get('tx_power', 15)
        freq = link_info.get('freq', 2.4)
        noise = link_info.get('noise', -90)
        path_loss = self.compute_path_loss(distance, freq)
        rssi = self.compute_rssi(tx_power, path_loss)
        snr = self.compute_snr(rssi, noise)
        if snr < 5:
            loss = 30
        elif snr < 15:
            loss = 30 - (snr - 5) * 2.9  # 线性下降
        else:
            loss = 1
        return max(0, min(30, int(loss)))
        
            
    def set_tc_for_all_nodes(self):
        """
        设置所有节点的流量控制
        """
        for src in self.nodes:
            for dst in self.nodes:
                if src == dst:
                    continue
                
                link_info = src.get_link_quality(dst)
                # 由外部函数/智能体/仿真环境动态计算
                bandwidth = self.compute_bandwidth(src, dst, link_info)  # 单位：Mbit
                delay = self.compute_delay(src, dst, link_info)          # 单位：ms
                loss = self.compute_loss(src, dst, link_info) 
                iface=f"wlan{src.name}"
                # 单位：百分比
                # 删除已有规则
                subprocess.call(f"tc qdisc del dev {iface} root", shell=True)
                # 添加新规则
                cmd = (
                    f"tc qdisc add dev {iface} root handle 1: htb default 11; "
                    f"tc class add dev {iface} parent 1: classid 1:1 htb rate {bandwidth}mbit; "
                    f"tc class add dev {iface} parent 1:1 classid 1:11 htb rate {bandwidth}mbit; "
                    f"tc qdisc add dev {iface} parent 1:11 handle 10: netem delay {delay}ms loss {loss}%"
                )
                subprocess.call(cmd, shell=True)
                print(f"[TC] {iface}: 带宽={bandwidth}Mbit 延迟={delay}ms 丢包={loss}% ({src.name}->{dst.name})")
    
    def plot_node_link_quality(self,src_node_name=None,target_node_name=None,save_path=None):
        times = []
        rssis = []
        bitrates = []
        loss=[]
        latency=[]
        exec_times=[]
        
        src_node = self.node_dict.get(src_node_name)
        if not src_node:
            print(f"未找到节点 {src_node_name}")
            return
            
        for record in src_node.link_quality_history:
            if record['target'] == target_node_name:
                times.append(record['time'])
                rssis.append(record['rssi'])
                bitrates.append(record['bitrate'])
                loss.append(record['loss'])
                latency.append(record['latency'])
                
        if not times:
            print(f"节点 {src_node_name} 到 {target_node_name} 无链路数据")
            return
            
        plt.figure(figsize=(16,14))
        plt.subplot(4,1,1)
        plt.plot(times, rssis, marker='o')
        plt.title(f"Node {src_node.name} to Node{target_node_name} RSSI over time")
        plt.ylabel("RSSI (dBm)")
        plt.xlabel("Time")
        
        plt.subplot(4,1,2)
        plt.plot(times, bitrates, marker='o', color='orange')
        plt.title(f"Node {src_node.name} to Node{ target_node_name} Bitrate over time")
        plt.ylabel("Bitrate (Mbps)")
        plt.xlabel("Time")
        
        plt.subplot(4,1,3)
        plt.plot(times, loss, marker='o', color='red')
        plt.title(f"Node {src_node.name} to Node{ target_node_name} Packet Loss over time")
        plt.ylabel("Loss (%)")
        plt.xlabel("Time")
        
        plt.subplot(4,1,4)
        plt.plot(times, latency, marker='o', color='green') 
        plt.title(f"Node {src_node.name} to Node{ target_node_name} Latency over time")
        plt.ylabel("Latency (ms)")
        plt.xlabel("Time")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            # 确保 test 目录存在
            subprocess.run(["mkdir", "-p", "test"])
            plt.savefig(f"test/{src_node_name} -> {target_node_name} link quality.png")
        plt.close() 
            
    def plot_all_nodes(self):
        n=len(self.nodes)
        for i in range(n):
            for j in range(i+1,n):
                src= self.nodes[i]
                target= self.nodes[j]
                self.plot_node_link_quality(src.name, target.name)
            
    # <--- MODIFIED: 新增函数，用于绘制奖励历史 ---_>
    def plot_reward_history(self, save_path="test/reward_history.png"):
        """绘制所有核心节点的奖励收敛曲线"""
        plt.figure(figsize=(12, 8))
        
        for node_name, rewards in self.reward_history.items():
            if rewards:
                # 计算移动平均值，使曲线更平滑
                moving_avg = np.convolve(rewards, np.ones(10)/10, mode='valid')
                plt.plot(moving_avg, label=f"Node {node_name}")
        
        plt.title("PPO Agent Reward History (Moving Average)")
        plt.ylabel("Average Reward")
        plt.xlabel("Training Step")
        plt.legend()
        plt.grid(True)
        
        # 确保 test 目录存在
        subprocess.run(["mkdir", "-p", "test"])
        plt.savefig(save_path)
        print(f"奖励历史图已保存到: {save_path}")
        plt.close()


    def test_all_links_concurrent(self):
        threads = []
        n = len(self.nodes)
        
        # <--- MODIFIED: 使用字典来组织任务，避免重复ping ---_>
        tasks = {} # {node: {'ips': [], 'macs': []}}
        
        for i in range(n):
            src_node = self.nodes[i]
            if src_node.name not in tasks:
                tasks[src_node.name] = {'ips': [], 'macs': []}
                
            for j in range(i + 1, n):
                dst_node = self.nodes[j]
                
                # 双向任务
                tasks[src_node.name]['ips'].append(dst_node.ip.split('/')[0])
                tasks[src_node.name]['macs'].append(dst_node.mac)
                
                if dst_node.name not in tasks:
                    tasks[dst_node.name] = {'ips': [], 'macs': []}
                tasks[dst_node.name]['ips'].append(src_node.ip.split('/')[0])
                tasks[dst_node.name]['macs'].append(src_node.mac)

        def node_task(node, ips, macs):
            """线程任务函数"""
            if ips: # 确保有任务
                node.get_latency(ips)
                node.get_rssi(macs)
        
        for node_name, task in tasks.items():
            node = self.node_dict[node_name]
            t = threading.Thread(target=node_task, args=(node, task['ips'], task['macs']))
            threads.append(t)  
            t.start()
            
        for t in threads:
            t.join()
        
   
