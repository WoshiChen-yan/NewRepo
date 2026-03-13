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
from ppoagent import PPOAgent, PPOCritic, SPLIT_WEIGHTS  # SPLIT_WEIGHTS: ip route nexthop weight 整数对
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
         self.all_node_agents={}#用于存储所有节点的智能体
         
         self.core_nodes=[]#用于存储核心节点
         self.node_agents={}#用于存储核心节点的智能体
         self.edge_nodes=[]#用于存储边缘节点
         
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
         self.batch_size = 5  # 收集5个时间步的数据后再学习
         self.global_steps = 0 # 全局时间步计数器
         self.agent_migration_interval = 10 # 每10个时间步迁移一次智能体
         
         # 2. (新) 保留事件触发器
         self.avg_core_reward_history = []
         self.reelection_trigger_threshold = -2.0 # (可调) 当核心平均奖励低于-2.0
         self.reelection_patience = 3          # (可调) 连续 3 次低于阈值
         self.bad_network_health_counter = 0   # 内部计数器

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
        self.state_dim = self.MAX_NEIGHBORS * 6  # 6 features: distance,latency,loss,rssi,doppler,throughput
        
        
        
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
                link_info['doppler_shift']/1000.0,
                link_info.get('utilization', 0.0)  # 链路利用率 [0,1] = throughput/bitrate
            ])
            neighbor_map_indices.append(self.nodes.index(target_node))

        # 5. 填充状态向量 (6 features per neighbor)
        padding_len = (self.MAX_NEIGHBORS * 6) - len(state_vector)
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
        throughput = min(link_info.get('throughput', 0.0), 100.0)  # Mbps, cap at 100

        lat_penalty = (latency / 100.0) * 5.0
        loss_penalty = (loss / 100.0) * 5.0
        # 吞吐量奖励: 链路实际被使用时给正向激励 (最大 +2.0)
        throughput_bonus = min(throughput / 50.0, 2.0)

        reward = 10.0 - lat_penalty - loss_penalty + throughput_bonus
        
        if latency >= 9999.0 or (loss == 100.0 and latency > 1000):
            return -5.0
            
        return reward
    
    def initialize_agents_and_elect(self):
            """
            
            在网络启动时调用一次。
            1. 创建共享的 Critic。
            2. 为 *所有* 节点创建 PPOAgent (Actor) 并永久存储。
            3. 运行一次 `select_core_nodes_distributed` 来选举 *初始* 核心节点。
            """
            print("\n=== (v0.2.0) 初始化所有节点的永久智能体 ===")

            # 1. 确保全局参数已设置
            if self.state_dim <= 0:
                print("错误：状态维度为0，无法创建智能体。")
                return

            # 2. 创建共享 Critic
            if not self.global_critic:
                print("  创建共享 Critic...")
                self.global_critic = PPOCritic(self.state_dim, self.num_dests, self.gru_hidden_dim, self.dest_embedding_dim)
                self.global_critic_optimizer = torch.optim.Adam(self.global_critic.parameters(), lr=1e-3)

            # 3. 为 *每个* 节点创建并存储一个永久的 PPOAgent
            for node in self.nodes:
                if node.name not in self.all_node_agents:
                    self.reward_history[node.name] = {} # 初始化奖励历史
                    
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
                    self.all_node_agents[node.name] = agent
                    print(f"  已为节点 {node.name} 创建永久 PPO 智能体。")
            
            # 4. 运行一次初始选举
            print("  执行初始核心节点选举...")
            self.test_all_links_concurrent() # 选举前需要链路数据
            self.select_core_nodes_distributed(num_nodes_rate=0.3) # 选举
        
    # ---  心训练逻辑重构 ---
    def update_routing(self):
        """
        执行: 决策(所有) -> 执行(区分) -> 测量 -> 存储(所有) -> 学习(核心) -> 健康检查
        """
        
        all_actions_to_execute = {} # 仅用于核心节点
        all_transitions = {}      # 用于所有节点 (S, A, logP)
        
        # --- 步骤 1: 决策 (仅核心节点参与，降低计算开销) ---
        for node in self.core_nodes: 
            agent = self.all_node_agents.get(node.name)
            if not agent: continue 
            
            node_index = self.nodes.index(node)
            state_seq, neighbor_map_indices = self.get_state_sequence(node.name)
            actions, split_actions, old_logprobs, old_split_logprobs = agent.select_action(
                state_seq, node_index, neighbor_map_indices)

            all_transitions[node.name] = (state_seq, actions, split_actions, old_logprobs, old_split_logprobs)

            # (关键) 只存储 *核心节点* 的动作, 用于执行
            if node in self.core_nodes:
                all_actions_to_execute[node.name] = (actions, split_actions)
            
        # --- 步骤 2: 执行 (核心节点PPO, 边缘节点默认) ---
        for node_name, (node_actions, node_split_actions) in all_actions_to_execute.items():
            core_node = self.node_dict[node_name]
            self.apply_node_routing(core_node, node_actions, node_split_actions)

        for edge_node in self.edge_nodes:
            self.apply_default_routing(edge_node)
        
        time.sleep(0.1) 
            
        # --- 步骤 3: 测量 (获取 E2E 结果) ---
        self.test_all_links_concurrent()
        
        
        # --- 步骤 4: 收集经验 (仅核心节点) ---
        self.global_steps += 1
        current_step_core_rewards = [] # 用于健康检查
        
        for node in self.core_nodes:
            agent = self.all_node_agents.get(node.name)
            if not agent or node.name not in all_transitions:
                continue

            (state_seq, actions, split_actions, old_logprobs, old_split_logprobs) = all_transitions[node.name]
            new_state_seq, _ = self.get_state_sequence(node.name)
            done = False 

            rewards_list = []
            for dest_index in range(self.num_dests):
                dest_node = self.nodes[dest_index]
                if dest_node == node:
                    rewards_list.append(0) 
                    continue
                
                e2e_link_info = node.get_link_quality_by_mac(dest_node.mac)
                reward = self._calculate_reward_from_link(e2e_link_info)
                rewards_list.append(reward)
                
                if dest_node.name not in self.reward_history[node.name]:
                    self.reward_history[node.name][dest_node.name] = []
                self.reward_history[node.name][dest_node.name].append(reward)

            # (关键) 所有节点都 *存储* 经验, 保证数据是最新的
            agent.store((state_seq, actions, split_actions, rewards_list, new_state_seq, old_logprobs, old_split_logprobs, done))
            
            if node in self.core_nodes and rewards_list:
                current_step_core_rewards.append(np.mean(rewards_list))

        # --- 步骤 5: 触发全局学习 (v0.2.0 优化: 只学习核心节点) ---
        if self.global_steps > self.batch_size: 
            if self.global_steps % 10 == 0: 
                print(f"\n--- [全局学习步骤 {self.global_steps} (仅核心节点)] ---")

            # (!!!) 优化点: 只训练核心节点以节省时间
            for core_node in self.core_nodes:
                agent = self.all_node_agents.get(core_node.name)
                if agent:
                    # 智能体内部会检查 memory > batch_size
                    agent.learn() 

            if self.global_steps % 10 == 0:
                print("--- [学习步骤完成] ---\n")
        
        # --- 步骤 6 (v0.2.0): 检查网络健康并触发 (事件触发) ---
        if current_step_core_rewards:
            avg_reward = np.mean(current_step_core_rewards)
            self.avg_core_reward_history.append(avg_reward)
            
            if avg_reward < self.reelection_trigger_threshold:
                self.bad_network_health_counter += 1
                print(f"  [健康检查] 网络平均奖励 {avg_reward:.2f} (低于 {self.reelection_trigger_threshold}), 差评计数: {self.bad_network_health_counter}/{self.reelection_patience}")
            else:
                if self.bad_network_health_counter > 0:
                     print(f"  [健康检查] 网络平均奖励 {avg_reward:.2f}, 恢复正常。")
                self.bad_network_health_counter = 0 

            # (新) 事件触发重选举
            if self.bad_network_health_counter >= self.reelection_patience:
                print(f"  [!!! 事件触发] 网络健康度连续 {self.reelection_patience} 次不佳, 触发核心节点重选举!")
                self.test_all_links_concurrent() 
                self.select_core_nodes(num_nodes_rate=0.3)
                # (选举函数在内部会重置计数器)
    
    # --- (v5) 边缘路由逻辑 ---
    def apply_default_routing(self, edge_node):
        """
        (v5 - MODIFIED)
        为边缘节点（非核心节点）设置默认路由：
        1. 优先路由到“邻居中”的“质量最好的”核心节点。
        2. 如果没有核心邻居，则回退到路由到“质量最好的”*任何*邻居。
        """
        best_core_neighbor = None
        best_core_rssi = -float('inf')
        
        best_neighbor = None
        best_neighbor_rssi = -float('inf')

        # 1. 寻找所有可达邻居
        neighbors = []
        for n in self.nodes: 
            if n == edge_node: continue
            
            link_info = edge_node.get_link_quality_by_mac(n.mac)
            # 必须是可达的邻居 (rssi > -100 且 loss < 100)
            if link_info and link_info.get('rssi', -100) > -100 and link_info.get('loss', 100) < 100: 
                neighbors.append((n, link_info))
        
        # 2. 遍历邻居，寻找最佳核心节点和最佳(回退)邻居
        for neighbor_node, link_info in neighbors:
            rssi = link_info.get('rssi', -100)
            
            # 检查是否是最佳邻居 (回退选项)
            if rssi > best_neighbor_rssi:
                best_neighbor_rssi = rssi
                best_neighbor = neighbor_node
            
            # 检查是否是最佳 *核心* 邻居 (优先选项)
            if neighbor_node in self.core_nodes: 
                if rssi > best_core_rssi:
                    best_core_rssi = rssi
                    best_core_neighbor = neighbor_node
        
        # 3. 下发默认路由
        # 使用 proto 189 以区别于核心路由 (188)
        # 我们只管理 default 路由，不flush table main，避免删除本地/链路路由
        
        if best_core_neighbor:
            # 优先：路由到最佳核心邻居
            core_ip = best_core_neighbor.ip.split('/')[0]
            edge_node.cmd(f"ip route replace default via {core_ip} dev wlan{edge_node.name} proto 189")
        elif best_neighbor:
            # 回退：路由到最佳邻居 (它可能知道如何去核心)
            neighbor_ip = best_neighbor.ip.split('/')[0]
            edge_node.cmd(f"ip route replace default via {neighbor_ip} dev wlan{edge_node.name} proto 189")
        else:
            # 孤立：没有邻居，清除旧的默认路由，防止路由循环
            try:
                # 只删除我们自己（proto 189）添加的路由
                edge_node.cmd(f"ip route del default proto 189")
            except Exception as e:
                pass # 忽略错误 (例如路由不存在)
            print(f"[WARN] Edge node {edge_node.name} has no neighbors. Default route removed.")

    # --- (v5) 核心路由应用 ---
        # --- (v5) 核心路由应用 ---
    def apply_node_routing(self, core_node, actions_matrix, split_ratios_matrix=None):
        """
        应用核心节点的路由决策。
        split_ratios_matrix: list[int], 分流档位索引 (0=100/0, 1=75/25, 2=50/50, 3=25/75, 4=0/100)
        """
        core_node.cmd("ip route flush all proto 188")

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

            # 1. 无效动作: 下一跳是自己
            if primary_hop_ip == core_ip:
                continue

            # 2. 直连路由: 下一跳即目的地, 无需分流
            elif primary_hop_ip == dest_ip:
                cmd = f"ip route replace {dest_ip} dev wlan{core_iface} metric 10 proto 188"
                core_node.cmd(cmd)

            # 3. 网关路由: 根据 split_ratios_matrix 决定是否 ECMP 分流
            else:
                split_level = split_ratios_matrix[dest_index] if split_ratios_matrix else 0
                w_primary, w_backup = SPLIT_WEIGHTS[split_level]

                can_split = (
                    w_primary > 0 and w_backup > 0
                    and primary_hop_ip != backup_hop_ip
                    and backup_hop_ip != core_ip
                    and backup_hop_ip != dest_ip
                )

                if can_split:
                    # ECMP 加权分流
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


    # 取消注释 select_core_nodes 并实现 PPOAgent 的创建 ---_>
    def select_core_nodes(self, num_nodes_rate=0.3):
        """
        选择核心节点，并初始化共享的 Critic 和所有 PPO 智能体
        """
        # print("\n=== 初始化核心节点和 PPO 智能体 ===")
        
        # 1. 选择核心节点 (按局部连通性 + 平均RSSI评分)
        node_scores = {}
        for node in self.nodes:
            reachable = 0
            rssi_sum = 0.0
            for other in self.nodes:
                if other == node:
                    continue
                link_info = node.get_link_quality_by_mac(other.mac)
                if not link_info:
                    continue
                rssi = link_info.get('rssi', -100.0)
                loss = link_info.get('loss', 100.0)
                if rssi > -100 and loss < 100:
                    reachable += 1
                    rssi_sum += rssi

            avg_rssi = (rssi_sum / reachable) if reachable > 0 else -100.0
            # 连通邻居数量权重更高，RSSI作为细粒度打分
            node_scores[node.name] = reachable * 10.0 + (avg_rssi + 100.0)

        num_core = max(1, int(len(self.nodes) * num_nodes_rate))

        core_nodes_sorted = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)[:num_core]

        core_name_set = {name for name, _ in core_nodes_sorted}
        self.core_nodes = [node for node in self.nodes if node.name in core_name_set]
        self.edge_nodes = [node for node in self.nodes if node.name not in core_name_set]
        
        # <--- MODIFIED: 创建一个 节点名 -> 节点对象 的映射，方便查找 ---_>
        self.node_dict = {node.name: node for node in self.nodes}

        print(f"已选择 {len(self.core_nodes)} 个核心节点: {[node.name for node in self.core_nodes]}")

        # 2. 定义状态和动作空间的维度
        # 状态维度 = (节点数 - 1) * 5 (distance, latency, loss, rssi, doppler)
        self.state_dim = (len(self.nodes) - 1) * 6  # 6 features per neighbor
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
        self.global_critic = PPOCritic(self.state_dim, self.num_dests, self.gru_hidden_dim, self.dest_embedding_dim)
        self.global_critic_optimizer = torch.optim.Adam(self.global_critic.parameters(), lr=1e-3)

        # 4. 为每个核心节点创建 PPOAgent (独立 Actor)
        for core_node in self.core_nodes:
            name = core_node.name
            if name not in self.all_node_agents:
                agent = PPOAgent(
                    state_dim=self.state_dim,
                    num_dests=self.num_dests,
                    num_next_hops=self.num_next_hops,
                    gru_hidden_dim=self.gru_hidden_dim,
                    dest_embedding_dim=self.dest_embedding_dim,
                    actor_lr=1e-4,
                    critic_lr=1e-3,
                    critic=self.global_critic,
                    critic_optimizer=self.global_critic_optimizer
                )
                self.all_node_agents[name] = agent
                self.reward_history[name] = {} # 初始化奖励历史
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
    
    def plot_node_link_quality(self,src_node_name=None,target_node_name=None,save_path=None,
                               rssi_range=(-100, -50), bitrate_range=(0, 100), loss_range=(0, 100), latency_range=(0, 500)):
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
            if record.get('target') == target_node_name:
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
                src.update_link_quality_cache(
                    dst.mac,
                    latency=final_entry.get('latency'),
                    loss=final_entry.get('loss'),
                    rssi=final_entry.get('rssi'),
                    bitrate=final_entry.get('bitrate')
                )

    def wmediumd_config(self, path='test.cfg'):
        """
        根据 self.nodes 中的节点信息生成 wmediumd 配置文件（默认 test.cfg）。
        读取每个节点的 mac, position, direction, txpower 字段。
        """
        ids = [f'\"{n.mac}\"' for n in self.nodes]

        positions = [
            f"({n.position[0]:.1f}, {n.position[1]:.1f}, {n.position[2]:.1f})"
            for n in self.nodes
        ]

        directions = []
        for n in self.nodes:
            d = n.direction
            if isinstance(d, (list, tuple)) and len(d) >= 3:
                directions.append(f"({d[0]:.1f}, {d[1]:.1f})")
            elif isinstance(d, (list, tuple)) and len(d) == 2:
                directions.append(f"({d[0]:.1f}, {d[1]:.1f})")
            else:
                directions.append("(0.0, 0.0)")

        txpowers = [f"{getattr(n, 'txpower', 30.0):.1f}" for n in self.nodes]

        lines = []
        lines.append("ifaces :")
        lines.append("{")
        lines.append(f"    count = {len(ids)};")
        lines.append("    ids = [")
        for i, mid in enumerate(ids):
            sep = "," if i < len(ids) - 1 else ""
            lines.append(f"        {mid}{sep}")
        lines.append("    ];")
        lines.append("};")
        lines.append("")
        lines.append("model :")
        lines.append("{")
        lines.append('    type = "path_loss";')
        lines.append("    positions = (")
        for i, pos in enumerate(positions):
            sep = "," if i < len(positions) - 1 else ""
            lines.append(f"        {pos}{sep}")
        lines.append("    );")
        lines.append("    directions = (")
        for i, d in enumerate(directions):
            sep = "," if i < len(directions) - 1 else ""
            lines.append(f"        {d}{sep}")
        lines.append("    );")
        lines.append("    tx_powers = (" + ", ".join(txpowers) + ");")
        lines.append('    model_name = "log_distance";')
        lines.append("    path_loss_exp = 3.5;")
        lines.append("    xg = 0.0;")
        lines.append("};")

        cfg_text = "\n".join(lines) + "\n"

        try:
            with open(path, "w") as f:
                f.write(cfg_text)
            print(f"wmediumd 配置已写入: {path}")
        except Exception as e:
            print(f"[ERROR] 写入 wmediumd 配置失败: {e}")

    def end_test(self):
        """
        清理测试环境：停止 wmediumd、删除所有 Docker 容器、卸载 mac80211_hwsim 模块。
        """
        print("\n=== 清理测试环境 ===")

        # 1. 停止 wmediumd
        try:
            subprocess.run(["sudo", "pkill", "-f", "wmediumd"], check=False)
            print("  wmediumd 已停止。")
        except Exception as e:
            print(f"  停止 wmediumd 失败: {e}")

        # 2. 删除所有容器节点
        for node in self.nodes:
            try:
                node.remove_node()
            except Exception as e:
                print(f"  删除节点 {node.name} 失败: {e}")

        # 3. 卸载 mac80211_hwsim 内核模块
        try:
            subprocess.run(["sudo", "rmmod", "mac80211_hwsim"], check=False,
                           stderr=subprocess.DEVNULL)
            print("  mac80211_hwsim 模块已卸载。")
        except Exception as e:
            print(f"  卸载 mac80211_hwsim 失败: {e}")

        print("=== 清理完成 ===")
