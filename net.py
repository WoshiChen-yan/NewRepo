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
         
          # --- 局部状态和动作空间 ---
         
           # 每个邻居5个特征: distance, latency, loss, rssi, doppler
        
        
        
         # --- 全局参数 ---
         # <--- 为 GRU 添加状态历史管理 ---_>
         self.node_states={} # 存储节点 {name: state_vector} 的最新状态
         self.state_history = {} # 存储 {name: deque([state_vector, ...], maxlen=seq_len)}
         self.seq_len = 5 # <---  定义GRU要看的时间序列长度
         self.state_dim = 0 # <---  将在 select_core_nodes 中设置
         
         self.MAX_NEIGHBORS = 0 # <--- 每个节点考虑的最大邻居数
         self.num_dests =  0# <---  将在 select_core_nodes 中设置
         self.num_next_hops =  0# <---  将在 select_core_nodes 中设置
         self.dest_embedding_dim = 16 
         
         
         self.gru_hidden_dim = 64 # <--- GRU 隐藏层大小
         self.node_dict = {} # <--- 方便按名称查找节点
         
         self.reward_history = {} # <--- 存储奖励历史用于绘图
         self.batch_size = 2  # 收集32个时间步的数据后再学习
         self.global_steps = 0 # 全局时间步计数器
         self.agent_migration_interval = 1 # 每1个时间步迁移一次智能体

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
        self.node_dict[name]=node # <--- MODIFIED: 更新节点映射 ---_>
        
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
            
        total_nodes = len(self.nodes)
        if total_nodes == 0:
            print("  警告：网络中没有节点，无法选择核心节点。")
            return
        
        self.num_dests= total_nodes
        self.num_next_hops= total_nodes
        self.MAX_NEIGHBORS = total_nodes - 1
        self.state_dim = self.MAX_NEIGHBORS * 5
        
        
        
        self.wmediumd_config()
        self.start_wmediumd_inbackground()
        print(f"等待节点进入稳定............")
        time.sleep(8)
        
    
    def move_nodes(self,inter):
        if not inter:
            inter=self.interval
        for node in self.nodes:
            x,y,z=node.position
            dx,dy,dz=node.direction
            
            new_x=x+dx*inter
            new_y=y+dy*inter
            new_z=z+dz*inter
            
            node.position = (new_x, new_y, new_z)
            node.position = (round(new_x, 1), round(new_y, 1), round(new_z, 1))
            self.graph.nodes[node.name]['position'] = (round(new_x, 1), round(new_y, 1), round(new_z, 1))
            print(f"节点'{node.name}'位置已更新为: ({node.position[0]:.1f}, {node.position[1]:.1f}, {node.position[2]:.1f})")
         
    
    
    def _get_current_state_vector(self, node):
        state_vector = []
        
        # 1. 获取所有邻居的链路质量
        all_links = []
        for target_node in self.nodes:
            if target_node == node:
                continue
            link_info = node.get_link_quality_by_mac(target_node.mac)
            if link_info:
                all_links.append((link_info['rssi'], target_node, link_info))
        
        all_links.sort(key=lambda x: x[0], reverse=True)
        neighbors = all_links[:self.MAX_NEIGHBORS]
        
        neighbor_map_indices = []
        
        for rssi, target_node, link_info in neighbors:
            state_vector.extend([
                link_info['distance']/1000.0,
                link_info['latency']/100.0,
                link_info['loss']/100.0,
                link_info['rssi']/100.0,
                link_info['doppler_shift']/1000.0
            ])
            neighbor_map_indices.append(self.nodes.index(target_node))

        # 5. 填充状态向量
        padding_len = (self.MAX_NEIGHBORS * 5) - len(state_vector)
        if padding_len > 0:
            state_vector.extend([0.0] * padding_len)
            
        self.node_states[node.name] = state_vector
        return state_vector, neighbor_map_indices

    # --- (v5) 状态序列 ---
    def get_state_sequence(self, node_name):
        current_state_vector, neighbor_map_indices = self._get_current_state_vector(self.node_dict[node_name])
        
        if node_name not in self.state_history:
            zero_state = [0.0] * self.state_dim
            self.state_history[node_name] = deque([zero_state] * self.seq_len, maxlen=self.seq_len)
            
        self.state_history[node_name].append(current_state_vector)
        
        return np.array(self.state_history[node_name]), neighbor_map_indices


    # --- (v5) 奖励计算 ---
    def _calculate_reward_from_link(self, link_info):
        if not link_info:
            return -10.0 
        
        latency = min(link_info.get('latency', 1000.0), 1000.0) 
        loss = min(link_info.get('loss', 100.0), 100.0)
        
        lat_penalty = (latency / 100.0) * 5.0
        loss_penalty = (loss / 100.0) * 5.0
        
        reward = 10.0 - lat_penalty - loss_penalty
        
        if latency >= 9999.0 or (loss == 100.0 and latency > 1000):
            return -5.0
            
        return reward

    
    # --- MODIFIED (v6): 核心训练逻辑重构 ---
    def update_routing(self):
        """
        (v6)
        执行一个完整的 决策 -> 执行 -> 测量 -> 学习 循环
        以解决“奖励近视”问题
        """
        
        all_actions_for_all_nodes = {} 
        all_transitions = {} # 临时存储 (S, A, logP)
        
        # --- 步骤 1: 决策 (所有核心节点) ---
        # (基于 S_t 决定 A_t)
        for core_node in self.core_nodes: 
            agent = self.node_agents.get(core_node.name)
            if not agent:
                continue 
            
            node_index = self.nodes.index(core_node)
            
            # 1a. 获取局部状态 S_t
            state_seq, neighbor_map_indices = self.get_state_sequence(core_node.name)
        
            # 1b. 智能体选择动作 A_t
            actions, old_logprobs = agent.select_action(state_seq, node_index, neighbor_map_indices)
            
            # 1c. 存储决策，稍后执行
            all_actions_for_all_nodes[core_node.name] = actions
            
            # 1d. 存储经验的前半部分
            all_transitions[core_node.name] = (state_seq, actions, old_logprobs)

            
        # --- 步骤 2: 执行 (所有节点) ---
        # (应用 A_t 到环境中)
        
        # 2a. 应用核心节点的 PPO 路由
        for node_name, node_actions in all_actions_for_all_nodes.items():
            core_node = self.node_dict[node_name]
            self.apply_node_routing(core_node, node_actions) # 下发 ip route

        # 2b. 应用边缘节点的默认路由
        for node in self.nodes:
            if node not in self.core_nodes:
                self.apply_default_routing(node)
        
        # (给路由一个极短的时间生效)
        time.sleep(0.1) 
            
        # --- 步骤 3: 测量 (获取 E2E 结果) ---
        # (运行 fping 来测量 A_t 导致的端到端结果 R_t)
        self.test_all_links_concurrent()
        
        
        # --- 步骤 4: 学习 (所有核心节点) ---
        self.global_steps += 1
        
        for core_node in self.core_nodes:
            agent = self.node_agents.get(core_node.name)
            if not agent:
                continue

            # 4a. 取出经验的前半部分 (S_t, A_t, logP_t)
            (state_seq, actions, old_logprobs) = all_transitions[core_node.name]
            
            # 4b. 获取新状态 S_{t+1}
            new_state_seq, _ = self.get_state_sequence(core_node.name)
            done = False 

            # 4c. 计算 *端到端* 奖励 R_t
            rewards_list = []
            for dest_index in range(self.num_dests):
                dest_node = self.nodes[dest_index]
                if dest_node == core_node:
                    rewards_list.append(0) # 到自己的奖励为0
                    continue
                
                # --- 这是关键修正 ---
                # 直接查找 core_node -> dest_node 的 E2E fping 结果
                e2e_link_info = core_node.get_link_quality_by_mac(dest_node.mac)
                reward = self._calculate_reward_from_link(e2e_link_info)
                # --- 修正结束 ---
                
                rewards_list.append(reward)
                
                # 记录奖励 (用于绘图)
                if dest_node.name not in self.reward_history[core_node.name]:
                    self.reward_history[core_node.name][dest_node.name] = []
                self.reward_history[core_node.name][dest_node.name].append(reward)

            # 4d. 存储完整的经验 (S_t, A_t, R_t, S_{t+1})
            agent.store((state_seq, actions, rewards_list, new_state_seq, old_logprobs, done))
            
        # --- 步骤 5: 触发全局学习 ---
        if self.global_steps > self.batch_size: 
            # (可选) 打印日志，避免刷屏
            if self.global_steps % 10 == 0: # 每 10 步打印一次
                print(f"\n--- [全局学习步骤 {self.global_steps}] ---")

            for agent in self.node_agents.values():
                # agent.learn() 现在会从 replay buffer 采样
                agent.learn() 

            if self.global_steps % 10 == 0:
                print("--- [学习步骤完成] ---\n")
    
        # if self.global_steps % self.batch_size == 0 and self.global_steps > 0:
        #     print(f"\n--- [全局学习步骤 {self.global_steps}] ---")
        #     for agent in self.node_agents.values():
        #         agent.learn()
        #     print("--- [学习步骤完成] ---\n")
            
    
    # --- (v5) 边缘路由逻辑 ---
    def apply_default_routing(self, edge_node):
        """
        为边缘节点（非核心节点）设置简单的默认路由：
        将所有流量路由到“邻居中”的“质量最好的”核心节点。
        """
        best_core_neighbor = None
        best_quality = -float('inf')
        best_rssi = -float('inf')

        if not self.core_nodes:
            return 

        # 1. 寻找 *邻居中* 质量最好的 *核心节点*
        neighbors = []
        for n in self.nodes: # 检查所有节点
             if n == edge_node: continue
             
             link_info = edge_node.get_link_quality_by_mac(n.mac)
             if link_info and link_info['rssi'] > -100: # 必须是可达的邻居
                 neighbors.append((n, link_info))
        
        for neighbor_node, link_info in neighbors:
            if neighbor_node in self.core_nodes: # 如果这个邻居是核心节点
                # quality = self._calculate_reward_from_link(link_info)
                rssi=link_info.get('rssi', -100)
                
                if rssi > best_rssi:
                    best_rssi=rssi
                    best_core_neighbor = neighbor_node
        
        # 2. 下发默认路由
        edge_node.cmd("ip route flush table main") 
        if best_core_neighbor:
            core_ip = best_core_neighbor.ip.split('/')[0]
            edge_node.cmd(f"ip route replace default via {core_ip} ")


    # --- (v5) 核心路由应用 ---
    def apply_node_routing(self, core_node, actions_matrix):
        """
        应用核心节点的路由决策
        """
        
        core_node.cmd("ip route flush all proto 188") 
        
        for dest_index, actions in enumerate(actions_matrix):
            dest_node = self.nodes[dest_index]
            if dest_node == core_node: continue
                
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
            core_iface = core_node.name # 获取 wlan 接口名

            # 1. 检查无效动作: 下一跳是自己
            # (注意: ppoagent 里的掩码 应该已经阻止了_index，
            #  但我们在这里检查 IP 作为双重保险)
            if primary_hop_ip == core_ip:
                continue # 非法动作，不安装路由 (agent 会得到坏奖励)

            # 2. 检查有效动作: 下一跳是目的地 (直连路由)
            elif primary_hop_ip == dest_ip:
                # 这是最优的 1 跳直连路由
                # 我们必须安装一个 'dev' 路由，而不是 'via' 路由
                cmd_direct = f"ip route replace {dest_ip} dev wlan{core_iface} metric 10 proto 188"
                core_node.cmd(cmd_direct)
                
            # 3. 检查有效动作: 下一跳是网关 (2跳路由)
            else:
                # 这是标准的 2 跳路由
                cmd_primary = f"ip route replace {dest_ip} via {primary_hop_ip} metric 10 proto 188"
                core_node.cmd(cmd_primary)

                # 4. (可选) 安装备用路由
                # 确保备用路由也不是自己或目的地
                if primary_hop_ip != backup_hop_ip and backup_hop_ip != core_ip and backup_hop_ip != dest_ip:
                    cmd_backup = f"ip route add {dest_ip} via {backup_hop_ip} metric 20 proto 188"
                    core_node.cmd(cmd_backup)

            # if primary_hop_ip == core_node.ip.split('/')[0] or primary_hop_ip == dest_ip:
            #     continue 

            # cmd_primary = f"ip route replace {dest_ip} via {primary_hop_ip} metric 10 proto 188"
            # core_node.cmd(cmd_primary)
            
            # if primary_hop_ip != backup_hop_ip and backup_hop_ip != core_node.ip.split('/')[0] and backup_hop_ip != dest_ip:
            #     cmd_backup = f"ip route add {dest_ip} via {backup_hop_ip} metric 20 proto 188"
            #     core_node.cmd(cmd_backup)

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
        
    def generate_nodes(self, num_nodes, pattern="random", base_position=(0,0,0), spacing=3):
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
    
            

    def wmediumd_config(self, path='test.cfg'):
        """
        生成 wmediumd 配置文件（test.cfg），格式与示例一致。
        使用 self.nodes 中的 mac, position, direction, txpower 字段。
        """
        # prepare lists
        ids = [f'"{n.mac}"' for n in self.nodes]
        positions = [f"({n.position[0]:.1f}, {n.position[1]:.1f}, {n.position[2]:.1f})" for n in self.nodes]
        # directions 精确到小数点后一位，确保三元组存在
        directions = []
        for n in self.nodes:
            d = n.direction
            if isinstance(d, (list, tuple)):
                if len(d) >= 3:
                    directions.append(f"({d[0]:.1f}, {d[1]:.1f})")
                elif len(d) == 2:
                    directions.append(f"({d[0]:.1f}, {d[1]:.1f}, 0.0)")
                else:
                    directions.append("(0.0, 0.0, 0.0)")
            else:
                directions.append("(0.0, 0.0, 0.0)")
        txpowers = [f"{getattr(n, 'txpower', 30.0):.1f}" for n in self.nodes]

        content = []
        content.append("ifaces :")
        content.append("{")
        content.append(f"    count={len(ids)};")
        content.append("    ids = [")
        content.append("    " + ",\n    ".join(ids))
        content.append("    ];")
        content.append("}")
        content.append("        ")
        content.append("model :")
        content.append("{")
        content.append('    type = "path_loss";')
        content.append("    positions = (")
        content.append("        " + ",\n        ".join(positions))
        content.append("    );")
        content.append("    directions = (")
        content.append("        " + ",\n        ".join(directions))
        content.append("    );")
        content.append("    tx_powers = (" + ", ".join(txpowers) + ");")
        content.append('    model_name = "log_distance";')
        content.append("    path_loss_exp = 3.5;")
        content.append("    xg = 0.0;")
        content.append("};")

        cfg_text = "\n".join(content) + "\n"

        try:
            with open(path, "w") as f:
                f.write(cfg_text)
            print(f"wmediumd 配置已写入: {path}")
        except Exception as e:
            print(f"[ERROR] 写入 wmediumd 配置失败: {e}")

    
    
   

    def select_core_nodes_distributed(self, num_nodes_rate=0.3):
        """
        (v5)
        根据局部邻居质量选择核心节点，并动态迁移智能体。
        """
        print(f"\n--- [分布式核心节点选择 步骤 {self.global_steps}] ---")
        
        node_scores = {}
        
        # 1. 计算每个节点的“局部中心性”得分
        for node in self.nodes:
            score = 0
            try:
                # MODIFIED: 使用 get_link_quality_by_mac 替代 get_neighbor
                for target_node in self.nodes:
                    if target_node == node: continue
                    link_info = node.get_link_quality_by_mac(target_node.mac)
                    # 优质邻居：丢包 < 20% 且 延迟 < 100ms
                    if link_info and link_info['loss'] < 80 and link_info['latency'] < 200:
                        score += 1
            except Exception as e:
                print(f"  获取 {node.name} 邻居失败: {e}")
                score = 0
            node_scores[node.name] = score
        
        # 2. 选择 Top 30% 节点
        num_core = max(1, int(len(self.nodes) * num_nodes_rate))
        sorted_nodes = sorted(node_scores.items(), key=lambda item: item[1], reverse=True)
        new_core_node_names = {name for name, score in sorted_nodes[:num_core]}
        
        old_core_node_names = set(self.node_agents.keys())
        
        print(f"  局部得分排名: {sorted_nodes}")
        print(f"  原核心节点: {old_core_node_names}")
        print(f"  新核心节点 (Top {num_core}): {new_core_node_names}")
        
        # 3. 初始化共享 Critic (如果不存在)
        if not self.global_critic:
            print("  创建共享 Critic (v5)...")
            self.global_critic = PPOCritic(self.state_dim, self.num_dests, self.gru_hidden_dim, self.dest_embedding_dim)
            self.global_critic_optimizer = torch.optim.Adam(self.global_critic.parameters(), lr=1e-3)

        # 4. 智能体迁移
        # 4a. 销毁下线的智能体
        nodes_to_remove = old_core_node_names - new_core_node_names
        for node_name in nodes_to_remove:
            if node_name in self.node_agents:
                del self.node_agents[node_name]
                del self.reward_history[node_name]
                print(f"  [迁移] 销毁节点 {node_name} 的智能体。")

        # 4b. 创建上线的智能体
        nodes_to_add = new_core_node_names - old_core_node_names
        for node_name in nodes_to_add:
            core_node = self.node_dict[node_name]
            self.reward_history[core_node.name] = {}
            
            agent = PPOAgent(
                state_dim=self.state_dim,
                num_dests=self.num_dests,
                num_next_hops=self.num_next_hops,
                gru_hidden_dim=self.gru_hidden_dim,
                dest_embedding_dim=self.dest_embedding_dim,
                critic=self.global_critic, # 传入共享 Critic
                critic_optimizer=self.global_critic_optimizer,
                batch_size=self.batch_size 
            )
            self.node_agents[node_name] = agent
            print(f"  [迁移] 已为新核心节点 {node_name} 创建 PPO 智能体。")
        
        # 5. 更新核心节点列表
        self.core_nodes = [self.node_dict[name] for name in new_core_node_names]
         
                    
    # 取消注释 select_core_nodes 并实现 PPOAgent 的创建 ---_>
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
            print(f"图已保存到: {save_path}")
        else:
            # 确保 test 目录存在
            subprocess.run(["mkdir", "-p", "test"])
            plt.savefig(f"test/{src_node_name} -> {target_node_name} link quality.png")
            output_path=f"test/{src_node_name} -> {target_node_name} link quality.png"
            plt.savefig(output_path)
            print(f"图已保存到: {output_path}")
        plt.close() 
            
    def plot_all_nodes(self):
        n=len(self.nodes)
        for i in range(n):
            for j in range(i+1,n):
                src= self.nodes[i]
                target= self.nodes[j]
                self.plot_node_link_quality(src.name, target.name)
            
    def plot_reward_history(self, save_path="test/reward_history.png"):
        plt.figure(figsize=(15, 8))
        
        all_rewards = [] 
        
        for src_name, dest_dict in self.reward_history.items():
            for dest_name, rewards in dest_dict.items():
                if rewards:
                    plt.plot(rewards, alpha=0.1, color='gray', label=f'_{src_name}-{dest_name}')
                    all_rewards.append(rewards)

        if not all_rewards:
            print("没有收集到奖励数据，无法绘图。")
            plt.close()
            return

        try:
            max_len = max(len(r) for r in all_rewards)
            padded_rewards = [r + [r[-1]] * (max_len - len(r)) if r else [0] * max_len for r in all_rewards]
            
            avg_rewards = np.mean(padded_rewards, axis=0)
            
            window_size = 5
            if len(avg_rewards) > window_size:
                moving_avg = np.convolve(avg_rewards, np.ones(window_size)/window_size, mode='valid')
                plt.plot(np.arange(window_size-1, len(avg_rewards)), moving_avg, color='blue', linewidth=2, label=f'Global Average Reward (Moving Avg w={window_size})')
            else:
                plt.plot(avg_rewards, color='blue', linewidth=2, label='Global Average Reward')

        except Exception as e:
            print(f"计算平均奖励时出错: {e}")


        plt.title(f"PPO Agent Reward History (v6 - E2E Reward + CTDE)")
        plt.ylabel("Average Reward (E2E Per-Destination)")
        plt.xlabel("Training Iteration (Global Step)")
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"奖励历史图已保存到: {save_path}")
        plt.close()

    # def test_all_links_concurrent(self):
    #     threads= []
    #     threads_results = {}
    #     n = len(self.nodes)
        
    #     # --- 总是为所有节点创建条目 ---
    #     # 这样 get_link_quality_by_mac 总能找到
    #     for src in self.nodes:
    #         for dst in self.nodes:
    #             if src == dst: continue
    #             src.link_quality_history.append({
    #                 'target': dst.name,
    #                 'mac': dst.mac,
    #                 'ip': dst.ip.split('/')[0],
    #                 'time': time.time(),
    #                 'rssi': -100 ,
    #                 'bitrate': 0 ,
    #                 'loss': 100 ,
    #                 'latency': 9999 ,
    #             })
    #     # --- 结束 ---

    #     for i in range(n):
    #         ips=[]
    #         macs=[]
    #         for j in range(n): # MODIFIED: fping all nodes, not just i+1
    #             if i == j: continue
    #             src = self.nodes[i]
    #             dst = self.nodes[j]
                
    #             ips.append(dst.ip.split('/')[0])
    #             macs.append(dst.mac)
                
    #         def node_task(node, ips, macs):
    #             """线程任务函数"""
    #             node.get_latency(ips) # 先 fping
    #             node.get_rssi(macs)   # 再 iw (iw 会更新 fping 创建的条目)
    #         t= threading.Thread(target=node_task, args=(self.nodes[i], ips, macs))
    #         threads.append(t)  
    #         t.start()
            
    #     for t in threads:
    #         t.join()
    def test_all_links_concurrent(self):
        """
        (REFACTORED)
        并发测试所有链路，并为每个 src -> dst 对追加一个*单独的*、*合并的*历史条目。
        这可以防止 "锯齿" 图和数据污染。
        """
        threads = []
        # 用于收集所有线程的原始数据
        # 格式: { 'node_name': (latency_results_dict, rssi_results_dict), ... }
        thread_results = {}
        n = len(self.nodes)

        def node_task(node, ips, macs):
            """线程任务函数：获取数据并存储在共享字典中"""
            latency_results = node.get_latency(ips) # 返回 {ip: data}
            rssi_results = node.get_rssi(macs)     # 返回 {mac: data}
            thread_results[node.name] = (latency_results, rssi_results)

        for i in range(n):
            src_node = self.nodes[i]
            ips = []
            macs = []
            for j in range(n):
                if i == j: continue
                dst_node = self.nodes[j]
                ips.append(dst_node.ip.split('/')[0])
                macs.append(dst_node.mac.lower()) # 确保 MAC 是小写
                
            t = threading.Thread(target=node_task, args=(src_node, ips, macs))
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join() # 等待所有 fping/iw 完成

        # --- 核心修改：数据合并与追加 ---
        current_time = time.time()
        for src in self.nodes:
            # 检查节点是否有返回结果 (如果线程出错则跳过)
            if src.name not in thread_results:
                continue    
            latency_data, rssi_data = thread_results[src.name]
            for dst in self.nodes:
                if src == dst:
                    continue
                    
                dst_ip = dst.ip.split('/')[0]
                dst_mac = dst.mac.lower()
                
                # 1. 建立此时间步的默认条目
                final_entry = {
                    'target': dst.name,
                    'mac': dst.mac,
                    'ip': dst_ip,
                    'time': current_time,
                    'rssi': -100.0,    # 默认 (无法通信)
                    'bitrate': 0.0,    # 默认 (无法通信)
                    'loss': 100.0,     # 默认 (无法通信)
                    'latency': 9999.0  # 默认 (无法通信)
                }
                
                # 2. 如果 fping 找到了延迟/丢包数据，则覆盖
                if dst_ip in latency_data:
                    final_entry.update(latency_data[dst_ip])
                    
                # 3. 如果 iw 找到了 RSSI/Bitrate 数据，则覆盖
                if dst_mac in rssi_data:
                    final_entry.update(rssi_data[dst_mac])
                    
                # 4. 追加这一个合并后的、完整的数据点
                src.link_quality_history.append(final_entry)
