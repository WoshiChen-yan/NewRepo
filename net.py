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
from dqn_agent import DQNAgent
from ppo_agent import PPOAgent , PPOCritic
import matplotlib.pyplot as plt
import threading
import torch
import math

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
         self.node_states={}#用于存储核心节点的状态
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
            
            # # 重新加载模块,不创建任何模块
            # subprocess.run(['sudo', 'modprobe', 'mac80211_hwsim', 'radios=0'])
            # time.sleep(1)
            
            # # 验证接口
            # result = subprocess.run(['iw', 'dev'], 
            #                     capture_output=True, 
            #                     text=True)
            # print("Current wireless interfaces:")
            # print(result.stdout)
            
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
         
    # def start_networkx(self):
    #     self.graph=nx.Graph(name=self.id)
    #     for node in self.nodes_list:
    #         mac_address = node['mac_address']
    #         position = node['position']
    #         direction = node['direction']
    #         txpower=node['txpower']
    #         self.graph.add_node(mac_address, position= position,direction=direction,txpower=txpower)
             
    #         # self.graph.add_node(node.mac, position= node.postion,direction=node.direction)
    
    # def calculate_relative_velocity(node1,node2):
    #     pos1=np.array(node1.position)
    #     pos2=np.array(node2.position)
    #     vel1=np.array(node1.direction)
    #     vel2=np.array(node2.direction) 
        
    #     direction=pos2-pos1
    #     norm=np.linalg.norm(direction)
    #     if norm==0:
    #         return 0
    #     direction=direction/norm
    #     v_rel=np.dot(vel1-vel2,direction)
    #     return v_rel
    
    # def calculate_doppler_shift(node1,node2,carrier_freq=2.4e9):
    #     c=3e8
    #     v_rel= calculate_relative_velocity(node1,node2)
    #     if v_rel==0:
    #         return 0
    #     doppler_shift=(carrier_freq/c)*v_rel
    #     return doppler_shift
    
    
        
        # self.update_wmediumd_config()
    
    def get_node_state(self,node):
        state=[]
        for target in self.nodes:
            if target != node:
                link_info=node.get_link_quality(target)
                self.graph.clear_edges()
                if link_info['distance']<800 :
                    self.graph.add_edge(node.name,target.name,
                                        weight=0,
                                        **link_info)
                state.extend([
                    link_info['distance']/1000.0,
                    link_info['latency']/100.0,
                    link_info['loss']/100.0,
                    # link_info['snr']/100.0,
                    link_info['rssi']/100.0,
                    link_info['doppler_shift']/1000.0
                    
                ])
                # 获取相邻核心节点网络状态
        for other_node in self.core_nodes:
            if other_node != node:
                 state.extend(self.node_states.get(other_node.name, [0]*len(state)//(len(self.nodes)-1)))
        return state

    def compute_node_reward(self, node):
        """计算指定节点的奖励"""
        reward = 0
        for _, target in self.graph.edges(node.name):
            edge = self.graph[node.name][target]
            # 考虑该节点相关的所有链路质量
            base_reward = 10
            reward = (base_reward -
                edge['latency'] * 0.4 -
                edge['loss'] * 0.3 -
                abs(edge['doppler_impact']) / 1000.0 * 0.3
            )
        return reward
    
    def update_routing(self):
        self.get_all_links_info()
        self.select_core_nodes()      
        for core_node in self.core_nodes:
            agent=self.node_agents[core_node.name]
            
            
            dpid=int(core_node.name)
            if dpid in self.controllers:
                 state=self.get_node_state(core_node)
                 self.node_states[core_node.name]=state
            
            action=agent.select_action(state)
            
            self.apply_node_routing(core_node,action)
            new_state=self.get_node_state(core_node)
            reward=self.compute_node_reward(core_node)
            done=0
            # agent.store(state,action,reward,new_state)        
            # DQN的存储方式
            
            agent.store(state, action, reward, new_state, done)
            # PPO的存储方式
            agent.learn()
        
            
            
            
            
        # for u, v in self.graph.edges():
        #     edge = self.graph[u][v]
        #     edge['weight'] = edge['latency'] * 0.7 + edge['loss'] * 0.3
        # #生成路由策略 可以使用多种路由策略  这里简单使用 dijkstra
        # all_paths = dict(nx.shortest_path(
        #     self.graph,
        #     weight="weight",
        #     method='dijkstra')
        # print("\n=== 路由表 ===")
        # for src_name, paths in all_paths.items():
        #     print(f"\n源节点: {src_name}")
        #     for dst_name, path in paths.items():
        #         path_str = " -> ".join(path)
        #         print(f"  目标: {dst_name:<20} 路径: {path_str}")
        # print("\n=== 路由表结束 ===\n")
        # # 下发路由规则
        # for src_name, paths in all_paths.items():
        #     src_node = self.node_dict[src_name]
        #     for dst_name, path in paths.items():
        #         if len(path) >= 2:
        #             next_hop = self.node_dict[path[1]].ip
        #             src_node.cmd(
        #                 f"ip route replace {self.node_dict[dst_name].ip} via {next_hop}"
        #             )
        # self.wmediumd_config()
             
    
    def apply_node_routing(self, core_node, action):
        """应用核心节点的路由决策"""
        try:
            target_node = self.nodes[action]
            
            # 将流量引导经过选中的目标节点
            for src in self.nodes:
                for dst in self.nodes:
                    if src != dst:
                        if src == core_node:
                            src.cmd(f"ip route replace {dst.ip} via {target_node.ip}")
                            print(f"核心节点 {core_node.name} 的路由已更新: {src.name} -> {dst.name} 经过 {target_node.name}")
                            # 通过核心节点和目标节点的路径
                        else:
                            src.cmd(f"ip route replace {dst.ip} via {core_node.ip}")
                            print(f"核心节点 {core_node.name} 的路由已更新: {src.name} -> {dst.name} 经过 {core_node.name}")
                            #如果源节点不是核心节点  线路由经过核心节点 
            
            
                       
        except Exception as e:
            print(f"应用路由决策时出错: {e}")
        

    def end_test(self):
        self.cleanup_interfaces()
        # subprocess.Popen(['sudo', 'pkill', 'wmediumd'])
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
        """
        Generate a CFG format configuration file based on network data
        Args:
            output_path: Path where to save the CFG file
            append: If True, append to existing file; if False, create new file
        """
        while Net._next_cfg_id in Net._used_cfg_ids:
            Net._next_cfg_id += 1
            
        net_id = Net._next_cfg_id
        Net._used_cfg_ids.add(net_id)
        Net._next_cfg_id += 1
        
        cfg_content = f"""medium =
(
        {{
                # required
                id = {net_id};
                # required
                interfaces = ["""
        
        # Add interfaces
        interfaces = [node.mac for node in self.nodes]
        cfg_content += ",\n                              ".join([f'"{mac}"' for mac in interfaces])
        
        cfg_content += f"""];enable_interference = true;
                # required
                model =
                {{
                        # required
                        type = "path_loss";
                        simulate_interference = true;
                        noise_level = -91;
                        fading_coefficient = 1;
                        # positions (x,y,z)
                        positions = ("""
        
        # Add positions
        positions = [(float(x),float(y),float(z))for x,y,z in [node.position for node in self.nodes]]
        cfg_content += ",\n                                     ".join([f"({x:.1f}, {y:.1f}, {z:.1f})" for x,y,z in positions])
    
        
        cfg_content += """);
                        # Movement directions
                        move_interval = 1.0;
                        directions = ("""
        
        # Add directions
        directions = [(float(x),float(y),float(z))for x,y,z in [node.direction for node in self.nodes]]
        cfg_content += ",\n                                      ".join([f"({x:.1f}, {y:.1f}, {z:.1f})" for x,y,z in directions])
    
        
        cfg_content += """);
                        # TX powers
                        tx_powers = ["""
        
        # Add TX powers
        tx_powers = [node.txpower for node in self.nodes]
        cfg_content += ", ".join([str(power) for power in tx_powers])
        
        cfg_content += f"""];
                        antenna_gain = [3,3];
                        
                        model_name = "free_space";
                        
                        model_params = #free_space
                        {{
                            system_loss = 1;
                        }}
                }}
        }}
);"""
        print(cfg_content)
        # Write to file
        mode = 'a' if append else 'w'
        with open(output_path, mode) as f:
            if append:
                f.write("\n\n")  # Add separation between networks
            f.write(cfg_content)
            
            
    def wmediumd_config(self):
        # Generate configuration  
    
        config = """ifaces :
{
    ids = [
"""
        # add mac address into config
        for i, node in enumerate(self.nodes):
            line = f"    \"{node.mac}\""
            if i < len(self.nodes) - 1:
                line += ","
            config += line + "\n"
        config += "    ];\n"
        
            # Add model configuration
        config += """}
        
model :
{
    type = "path_loss";
    positions = (
"""
        for i, node in enumerate(self.nodes):
            pos = node.position
            line = f"        ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})"
            if i < len(self.nodes) - 1:
                line += ","
            config += line + "\n"
        config += "    );"

        config += """
    directions = (
"""
        for i, node in enumerate(self.nodes):
            pos = node.direction
            line = f"        ({pos[0]:.1f}, {pos[1]:.1f})"
            if i < len(self.nodes) - 1:
                line += ","
            config += line + "\n"
        config += "    );"

            # Add tx_powers
        config += '\n    tx_powers = ('
        powers = [node.txpower for node in self.nodes]
        config += ', '.join(f'{power:.1f}' for power in powers)
        config += ');'

            # Add model parameters
        config += """    
    model_name = "log_distance";
    path_loss_exp = 3.5;
    xg = 0.0;
};\n"""
        
        
        # print(config)
        # Write configuration to file
        config_path = 'test.cfg'
        with open(config_path, 'w') as f:
            f.write(config)
    
    
    def test_connectivity(self, count=10, timeout=1):
        """
        Test connectivity between all nodes using ping
        Args:
        count: Number of ping packets to send
        timeout: Timeout in seconds for each ping
        """
        print("\n=== 连通性测试开始 ===")
        for i, source in enumerate(self.nodes):
            for target in self.nodes[i+1:]:  # Test each pair only once
                print(f"\n测试 {source.name} -> {target.name}:")
                target_ip=target.ip.split('/')[0]
                # Construct ping command
                cmd = f"ping -c {count} {target_ip}"
                print(f"命令: {cmd}")
                
                try:
                    
                    # Execute ping command from source node
                    result = source.cmd(cmd)
                    print(result)
                    
                    if "100% packet loss" in result:
                        print(f"❌ 连接失败: {source.name} 无法连接到 {target.name}")
                    
                    else:
                    # 解析丢包率
                        
                            packet_loss = None
                            for line in result.split('\n'):
                                if "packets transmitted" in line:
                                    packet_loss = line.split(',')[2].strip().split()[0]  
                                if 'round-trip' in line:
                                    # Extract average latency value
                                    avg = line.split('=')[1].strip().split('/')[1]
                                    rtt = float(avg)
                                    print(rtt)
                                
                                
                            print(f"✓ 连接成功: {source.name} -> {target.name}")
                            print(f"  丢包率: {packet_loss}%")
                            print(f"  平均延迟: {rtt} ms")
                    time.sleep(5)    
                    
                except Exception as e:
                    print(f"❌ 测试失败: {str(e)}")
                        
        print("\n=== 连通性测试结束 ===")


    # def select_core_nodes(self, num_nodes_rate=0.3):
    #     """
    #     Select core nodes based on DQN policy
    #     Args:
    #         num_nodes: Number of core nodes to select
    #     """
        
    #     old_core_nodes=set(node.name for node in self.core_nodes)
    #     # 计算节点的介数中心性和度中心性
    #     betweens=nx.betweenness_centrality(self.graph)
    #     degrees=nx.degree_centrality(self.graph)
        
    #     node_scores = {}
    #     for node in self.nodes:
    #         score=(
    #             betweens.get(node.name,0) * 0.6+
    #             degrees.get(node.name,0) * 0.4
    #         )
    #         node_scores[node.name]=score

    #     num_core=max(1,int(len(self.nodes)* num_nodes_rate))
    #     #选择得分最高的百分之三至十作为核心节点
    #     core_nodes=sorted(node_scores.items(),key=lambda x:x[1],reverse=True)[:num_core]
        
        
    #     # 此处可以添加策略来随机挑选非最优节点 例如 epsilon-greedy策略
        
        
    #     self.core_nodes=[
    #         node for node in self.nodes
    #         if node.name in [name for name, _ in core_nodes]
    #     ]
        
    #     new_core_nodes=set(node.name for node in self.core_nodes)
    #     print(f"以选择 {len(self.core_nodes)} 个核心节点: {[node.name for node in self.core_nodes]}")
        
    #     for name in new_core_nodes:
    #         if name not in old_core_nodes:
    #             if old_core_nodes:
    #                 old_node=min(old_core_nodes,
    #                              key=lambda x:node_scores[x]-node_scores[name])
    #                 self.node_agents[name]=self.node_agents.pop(old_node)
    #                 print(f"核心节点智能体 {old_node} 被替换为 {name}的智能体")
                
                
    #             else:
    #                 state_dim=(len(self.nodes)-1)* 5
    #                 action_dim=len(self.nodes)
    #                 if self.global_critic == None:
    #                     self.global_critic=PPOCritic(state_dim=state_dim)
    #                     self.global_critic_optimizer=torch.optim.Adam(self.global_critic.parameters(), lr=1e-3)
    #                 # self.node_agents[name]=DQNAgent(state_dim=state_dim,action_dim=action_dim)
    #                 self.node_agents[name]=PPOAgent(state_dim=state_dim,action_dim=action_dim,
    #                                                 critic=self.global_critic)
    #                 self.node_agents[name].optimizer=self.global_critic_optimizer
    #                 print(f"核心节点智能体 {name} 被添加")
         
                    
    #     for name in list(self.node_agents.keys()):
    #         if name not in new_core_nodes:
    #             del self.node_agents[name]
    #             print(f"核心节点智能体 {name} 被删除")
        
    #     for core_node in self.core_nodes:
    #         core_node.get_neighbor()
    
    
    def select_core_nodes(self, num_nodes_rate=0.3):
        node_scores = {}
        for node in self.nodes:
            neighbors = node.get_neighbor()
            neighbor_count = len(neighbors)
            # 统计平均RSSI
            avg_rssi = 0
            if neighbors:
                rssis = [node.get_link_quality_by_mac(mac)['rssi'] for mac in neighbors]
                avg_rssi = sum(rssis) / len(rssis)
            # 统计流量（假设有 node.traffic）
            traffic = getattr(node, 'traffic', 0)
            # 综合打分
            score = neighbor_count * 0.5 + avg_rssi * 0.3 + traffic * 0.2
            node_scores[node.name] = score

        num_core = max(1, int(len(self.nodes) * num_nodes_rate))
        core_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)[:num_core]
        self.core_nodes = [
            node for node in self.nodes
            if node.name in [name for name, _ in core_nodes]
        ]
                
                
    def validate_agent_performance(self, episodes=3):
        """
        验证智能体的训练效果
        Args:
            episodes: 验证轮数
        Returns:
            performance_metrics: 包含各项性能指标的字典
        """
        print("\n=== 开始验证智能体性能 ===")
        
        performance_metrics = {
            'average_reward': [],
            'routing_success_rate': [],
            'average_latency': [],
            'path_optimality': []
        }

        for episode in range(episodes):
            print(f"\n第 {episode + 1} 轮验证:")
            self.select_core_nodes()
            if episode > 0:
                self.move_nodes()
            # 记录该轮的指标
            episode_rewards = []
            successful_routes = 0
            total_routes = 0
            latencies = []
            path_lengths = []
            optimal_lengths = []

            # 对每个核心节点进行验证
            for core_node in self.core_nodes:
                agent = self.node_agents[core_node.name]
                
                # 获取当前状态
                state = self.get_node_state(core_node)
                
                # 使用智能体选择动作(不进行探索)
                action = agent.select_action(state)
                
                try:
                    # 应用路由决策
                    self.apply_node_routing(core_node, action)
                    
                    # 测试路由效果
                    for src in self.nodes:
                        for dst in self.nodes:
                            if src != dst:
                                total_routes += 1
                                
                                # 使用ping测试连通性和延迟
                                target_ip = dst.ip.split('/')[0]
                                result = src.cmd(f"ping -c 1 -W 1 {target_ip}")
                                
                                if "1 received" in result:
                                    successful_routes += 1
                                    
                                    # 提取延迟
                                    for line in result.split('\n'):
                                        if 'time=' in line:
                                            latency = float(line.split('time=')[1].split()[0])
                                            latencies.append(latency)
                                    
                                    # 获取当前路径长度
                                    current_path = self._get_current_path(src, dst)
                                    if current_path:
                                        path_lengths.append(len(current_path))
                                    
                                    # 计算最优路径长度
                                    try:
                                        optimal_path = nx.shortest_path(self.graph, 
                                                                        src.name,
                                                                        dst.name,
                                                                        weight="weight")
                                        print(f"最优路径{src.name} -> {dst.name}: {optimal_path}")
                                        optimal_lengths.append(len(optimal_path))
                                    except:
                                        pass
                    
                    # 计算奖励
                    reward = self.compute_node_reward(core_node)
                    print(f"轮数: {episode + 1}, 节点: {core_node.name}, 奖励: {reward:.2f}")
                    episode_rewards.append(reward)
                    
                except Exception as e:
                    print(f"验证过程出错: {e}")

            # 计算该轮的平均指标
            avg_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
            success_rate = successful_routes / total_routes if total_routes > 0 else 0
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            path_optimality = sum(l1/l2 for l1, l2 in zip(optimal_lengths, path_lengths)) / len(path_lengths) if path_lengths else 0
            
            # 存储指标
            performance_metrics['average_reward'].append(avg_reward)
            performance_metrics['routing_success_rate'].append(success_rate)
            performance_metrics['average_latency'].append(avg_latency)
            performance_metrics['path_optimality'].append(path_optimality)
            
            print(f"平均奖励: {avg_reward:.2f}")
            print(f"路由成功率: {success_rate*100:.2f}%")
            print(f"平均延迟: {avg_latency:.2f}ms")
            print(f"路径最优率: {path_optimality*100:.2f}%")
        
        print("\n=== 验证完成 ===")
        return performance_metrics

    def _get_current_path(self, src, dst):
        """
        获取当前src到dst的实际路由路径
        """
        try:
            # 使用traceroute获取实际路径
            result = src.cmd(f"traceroute -n {dst.ip.split('/')[0]} -m 15 -w 1")
            path = []
            for line in result.split('\n')[1:]:  # 跳过第一行
                if line.strip():
                    # 提取IP地址
                    ip = line.split()[1]
                    if ip != '*':  # 忽略无响应的跳数
                        for node in self.nodes:
                            if node.ip.split('/')[0] == ip:
                                path.append(node.name)
                                break
            return path
        except:
            return None
        
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
        for node in self.nodes:
            if node.name == src_node_name:
                src_node = node
                break
        for record in src_node.link_quality_history:
            if record['target'] == target_node_name:
                times.append(record['time'])
                rssis.append(record['rssi'])
                bitrates.append(record['bitrate'])
                loss.append(record['loss'])
                latency.append(record['latency'])
                
        if not times:
            print("时间  无数据")
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
        
        # plt.subplot(5,1,5)
        # plt.plot(times,exec_times, marker='o', color='purple')
        # plt.title(f"Node {src_node.name} to Node{ target_node_name} Execution Time over time")
        # plt.ylabel("Execution Time (s)")
        # plt.xlabel("Time")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.savefig(f"test/{src_node_name} -> {target_node_name} link quality.png")
        plt.close() 
            
    def plot_all_nodes(self):
        n=len(self.nodes)
        for i in range(n):
            for j in range(i+1,n):
                src= self.nodes[i]
                target= self.nodes[j]
                self.plot_node_link_quality(src.name, target.name)
            

    def test_all_links_concurrent(self):
        threads1 = []
        threads2 = []
        n = len(self.nodes)
        for i in range(n):
            ips=[]
            macs=[]
            for j in range(i+1, n):
                src = self.nodes[i]
                dst = self.nodes[j]
                # t = threading.Thread(target=src.get_link_quality, args=(dst,))
                src.link_quality_history.append({
                    'target': dst.name,
                    'mac': dst.mac,
                    'ip': dst.ip.split('/')[0],
                    'time': time.time(),
                    'rssi': -100 ,  # 默认值
                    'bitrate': 0 ,
                    'loss': 100 ,
                    'latency': 9999 ,
                })
                ips.append(dst.ip.split('/')[0])
                macs.append(dst.mac)
                # threads1.append(t)
                # t.start()
            print(f"节点 {self.nodes[i].name} 的即将进行的fping的IP地址: {ips}")
            def node_task(node, ips, macs):
                """线程任务函数"""
                node.get_latency(ips)
                node.get_rssi(macs)
            t= threading.Thread(target=node_task, args=(self.nodes[i], ips, macs))
            threads1.append(t)  
            t.start()
            # src.fping_multi(ips)
            # src.get_rssi(macs)
            # 等待所有线程结束
        for t in threads1:
            t.join()
        
   