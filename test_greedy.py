#!/usr/bin/env python3
"""
贪心路由算法测试文件

该脚本创建一个无线网络拓扑，并实现基于链路质量的贪心路由算法，
然后与现有路由算法进行性能比较。
"""

from net import Net
import time
import numpy as np
import matplotlib.pyplot as plt
import threading
import re

class GreedyRouter:
    """贪心路由算法实现"""
    
    def __init__(self, network):
        self.network = network
        self.routing_mode = "greedy"  # 贪心路由模式标记
        
    def apply_greedy_routing(self):
        """为网络中的所有节点应用贪心路由策略"""
        print("应用贪心路由算法...")
        
        # 先确保所有节点的IP转发已启用
        for node in self.network.nodes:
            node.cmd("sysctl -w net.ipv4.ip_forward=1")
        
        # 为每个节点计算贪心路由
        for source_node in self.network.nodes:
            self._compute_routes_for_node(source_node)
        
        print("贪心路由算法应用完成")
    
    def _compute_routes_for_node(self, source_node):
        """为单个节点计算贪心路由表"""
        # 清除旧的路由表（只清除proto 190的路由，这是我们的贪心路由标记）
        source_node.cmd("ip route flush all proto 190")
        
        # 为每个目标节点计算最佳下一跳
        for dest_node in self.network.nodes:
            if source_node == dest_node:
                continue  # 跳过自己
                
            # 找到到目标节点的最佳下一跳
            best_next_hop = self._find_best_next_hop(source_node, dest_node)
            
            if best_next_hop:
                # 安装路由
                dest_ip = dest_node.ip.split('/')[0]
                next_hop_ip = best_next_hop.ip.split('/')[0]
                
                # 使用proto 190来标记贪心路由
                source_node.cmd(
                    f"ip route replace {dest_ip} via {next_hop_ip} "
                    f"dev wlan{source_node.name} metric 10 proto 190"
                )
            else:
                print(f"警告: 节点 {source_node.name} 无法到达节点 {dest_node.name}")
    
    def _find_best_next_hop(self, source_node, dest_node):
        """
        贪心策略：找到到目标节点的最佳邻居节点
        评估标准：综合考虑RSSI（信号强度）、延迟和丢包率
        """
        best_hop = None
        best_score = -float('inf')
        
        # 获取源节点的所有邻居
        for neighbor in self.network.nodes:
            if neighbor == source_node or neighbor == dest_node:
                continue
                
            # 检查邻居是否可达
            link_info = source_node.get_link_quality_by_mac(neighbor.mac)
            if not link_info:
                continue
                
            # 只有当邻居节点可达时才考虑 (RSSI > -100, 丢包率 < 100%)
            if (link_info.get('rssi', -100) > -100 and 
                link_info.get('loss', 100) < 100):
                
                # 计算链路质量分数
                # - RSSI权重0.6：信号越强越好
                # - 延迟权重0.2：延迟越小越好
                # - 丢包率权重0.2：丢包率越低越好
                rssi = link_info.get('rssi', -100)
                latency = link_info.get('latency', 9999)
                loss = link_info.get('loss', 100)
                
                # 归一化各项指标
                # RSSI范围通常在-30到-90之间，我们将其归一化到0-1
                rssi_norm = (rssi + 90) / 60 if rssi > -90 else 0
                
                # 延迟通常在0-100ms之间，我们将其归一化到1-0
                latency_norm = 1 - (min(latency, 100) / 100)
                
                # 丢包率范围0-100%，我们将其归一化到1-0
                loss_norm = 1 - (loss / 100)
                
                # 计算综合分数
                score = (0.6 * rssi_norm) + (0.2 * latency_norm) + (0.2 * loss_norm)
                
                # 更新最佳邻居
                if score > best_score:
                    best_score = score
                    best_hop = neighbor
        
        # 如果找不到最佳邻居，检查是否可以直接到达目标节点
        direct_link_info = source_node.get_link_quality_by_mac(dest_node.mac)
        if (direct_link_info and 
            direct_link_info.get('rssi', -100) > -100 and 
            direct_link_info.get('loss', 100) < 100):
            # 直接到达目标节点，不需要下一跳
            dest_ip = dest_node.ip.split('/')[0]
            source_node.cmd(
                f"ip route replace {dest_ip} dev wlan{source_node.name} "
                f"metric 5 proto 190"
            )
            return None  # 直连路由，没有下一跳
        
        return best_hop
    
    def measure_performance(self, duration=30):
        """测量贪心路由的网络性能"""
        print(f"测量贪心路由性能，持续时间: {duration}秒...")
        
        # 启动iperf服务器
        server_node = self.network.nodes[0]
        server_node.cmd("iperf -s -u -p 5001 > /dev/null 2>&1 &")
        
        # 在其他节点上运行iperf客户端
        start_time = time.time()
        throughput_sum = 0
        latency_sum = 0
        packet_loss_sum = 0
        node_count = 0
        
        for client_node in self.network.nodes[1:]:
            server_ip = server_node.ip.split('/')[0]
            try:
                # 运行iperf测试
                result = client_node.cmd(f"iperf -c {server_ip} -u -t {duration} -b 10M")
                
                # 解析吞吐量结果
                throughput_match = re.search(r'Bytes\s+([\d.]+)\s+([KMG]?)bits/sec', result)
                if throughput_match:
                    throughput = float(throughput_match.group(1))
                    unit = throughput_match.group(2)
                    if unit == 'K':
                        throughput *= 1000
                    elif unit == 'M':
                        throughput *= 1000000
                    throughput_sum += throughput
                
                # 获取延迟和丢包
                ping_result = client_node.cmd(f"ping -c 10 {server_ip}")
                latency_match = re.search(r'rtt min/avg/max/mdev = [\d.]+/([\d.]+)/', ping_result)
                loss_match = re.search(r'([\d.]+)% packet loss', ping_result)
                
                if latency_match:
                    latency_sum += float(latency_match.group(1))
                if loss_match:
                    packet_loss_sum += float(loss_match.group(1))
                    
                node_count += 1
            except Exception as e:
                print(f"性能测试失败: {e}")
        
        # 计算平均值
        avg_throughput = throughput_sum / node_count if node_count > 0 else 0
        avg_latency = latency_sum / node_count if node_count > 0 else 0
        avg_packet_loss = packet_loss_sum / node_count if node_count > 0 else 0
        
        return {
            'throughput': avg_throughput,
            'latency': avg_latency,
            'packet_loss': avg_packet_loss
        }

def compare_routing_algorithms():
    """比较贪心路由和现有路由算法"""
    
    # 创建网络拓扑
    print("创建网络拓扑...")
    Net_1 = Net(name="greedy_test", interval=1)
    
    # 添加节点 - 6个节点形成网格拓扑
    Net_1.add_node("11", "02:00:00:00:00:01", "10.10.10.1/24", (0, 0, 0), (0.1, 0.1, 0.1))
    Net_1.add_node("12", "02:00:00:00:00:02", "10.10.10.2/24", (50, 0, 0), (0.1, 0.1, 0.1))
    Net_1.add_node("13", "02:00:00:00:00:03", "10.10.10.3/24", (100, 0, 0), (0.1, 0.1, 0.1))
    Net_1.add_node("14", "02:00:00:00:00:04", "10.10.10.4/24", (0, 50, 0), (0.1, 0.1, 0.1))
    # Net_1.add_node("15", "02:00:00:00:00:05", "10.10.10.5/24", (50, 50, 0), (0.1, 0.1, 0.1))
    # Net_1.add_node("16", "02:00:00:00:00:06", "10.10.10.6/24", (100, 50, 0), (0.1, 0.1, 0.1))
    
    # 启动网络
    Net_1.start_network()
    
    # 初始化贪心路由器
    greedy_router = GreedyRouter(Net_1)
    
    # 测试场景1: 静态网络
    print("\n=== 测试场景1: 静态网络 ===")
    
    # 等待网络稳定
    print("等待网络稳定...")
    time.sleep(10)
    
    # 测量现有路由算法性能
    print("测量现有路由算法性能...")
    existing_performance = greedy_router.measure_performance(duration=20)
    
    # 应用贪心路由并测量性能
    greedy_router.apply_greedy_routing()
    time.sleep(5)  # 等待路由生效
    greedy_performance = greedy_router.measure_performance(duration=20)
    
    # 测试场景2: 移动网络
    print("\n=== 测试场景2: 移动网络 ===")
    
    # 移动节点
    print("移动节点...")
    Net_1.move_nodes(10)
    time.sleep(5)
    
    # 更新贪心路由
    print("更新贪心路由...")
    greedy_router.apply_greedy_routing()
    time.sleep(5)
    
    # 测量移动场景下的性能
    mobile_performance = greedy_router.measure_performance(duration=20)
    
    # 输出比较结果
    print_comparison_results(existing_performance, greedy_performance, mobile_performance)
    
    # 结束测试
    print("\n结束测试...")
    Net_1.end_test()

def print_comparison_results(existing_perf, greedy_perf, mobile_perf):
    """打印性能比较结果"""
    print("\n=== 路由算法性能比较 ===")
    print(f"{'指标':<15} {'现有路由':<15} {'贪心路由(静态)':<20} {'贪心路由(移动)':<20}")
    print("-" * 70)
    
    # 吞吐量比较
    print(f"{'吞吐量(bps)':<15} {existing_perf['throughput']:<15.2f} "
          f"{greedy_perf['throughput']:<20.2f} {mobile_perf['throughput']:<20.2f}")
    
    # 延迟比较
    print(f"{'平均延迟(ms)':<15} {existing_perf['latency']:<15.2f} "
          f"{greedy_perf['latency']:<20.2f} {mobile_perf['latency']:<20.2f}")
    
    # 丢包率比较
    print(f"{'丢包率(%)':<15} {existing_perf['packet_loss']:<15.2f} "
          f"{greedy_perf['packet_loss']:<20.2f} {mobile_perf['packet_loss']:<20.2f}")
    
    # 计算提升百分比
    if existing_perf['throughput'] > 0:
        tp_improvement = ((greedy_perf['throughput'] - existing_perf['throughput']) / 
                          existing_perf['throughput']) * 100
    else:
        tp_improvement = 0
    
    if existing_perf['latency'] > 0:
        latency_improvement = ((existing_perf['latency'] - greedy_perf['latency']) / 
                              existing_perf['latency']) * 100
    else:
        latency_improvement = 0
    
    if existing_perf['packet_loss'] > 0:
        loss_improvement = ((existing_perf['packet_loss'] - greedy_perf['packet_loss']) / 
                           existing_perf['packet_loss']) * 100
    else:
        loss_improvement = 0
    
    print("\n=== 贪心路由相对现有路由的提升 ===")
    print(f"吞吐量提升: {tp_improvement:+.2f}%")
    print(f"延迟改善: {latency_improvement:+.2f}%")
    print(f"丢包率改善: {loss_improvement:+.2f}%")

if __name__ == "__main__":
    compare_routing_algorithms()