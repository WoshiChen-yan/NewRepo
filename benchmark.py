"""
基于 NetworkX 的 SPF 和 ECMP 对比算法
用于与现有 benchmark.py 中的手工实现进行对标测试
"""

import networkx as nx
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from net import Net


class  NetworkXAlgorithmComparison:
    """
    基于 NetworkX 的路由算法对标框架
    用于对比 SPF 和 ECMP 的 NetworkX 实现 vs 手工实现
    """
    
    def __init__(self, net, metric_type="combined"):
        """
        初始化对比工具
        
        Args:
            net (Net): 网络实例，包含所有节点和链路信息
            metric_type (str): 成本度量方式
                - "hops": 跳数
                - "latency": 延迟
                - "loss": 丢包率
                - "combined": 综合成本 (延迟 + 丢包率)
        """
        self.net = net
        self.metric_type = metric_type
        self.results = {
            "spf": [],
            "ecmp": [],
            "comparison": {}
        }
        
    # ======================== 成本计算 ========================
    
    def _get_link_cost(self, src_node, dst_node):
        """
        计算 src_node 到 dst_node 的链路成本
        与 benchmark.py 中的 ECMPRouting._get_link_cost() 保持一致
        """
        if src_node == dst_node:
            return 0
        
        # 从缓存中获取链路质量
        link_info = src_node.link_quality_cache.get(dst_node.mac, {})
        
        # 处理无链接情况 (>95% 丢包)
        loss = link_info.get('loss', 100)
        if loss >= 95:
            return float('inf')
        
        if self.metric_type == "hops":
            return 1
        
        elif self.metric_type == "latency":
            latency = link_info.get('latency', 9999.0)
            return min(latency, 9999.0)
        
        elif self.metric_type == "loss":
            return loss
        
        elif self.metric_type == "combined":
            # 综合成本 = 1.0 + (latency/100) + (loss/100)*2.0
            latency = link_info.get('latency', 1000.0)
            loss = link_info.get('loss', 100.0)
            
            latency = min(latency, 1000.0)
            loss = min(loss, 100.0)
            
            cost = 1.0 + (latency / 100.0) + (loss / 100.0) * 2.0
            return cost
        
        return float('inf')
    
    # ======================== 图构建 ========================
    
    def _build_graph(self):
        """
        从当前网络状态构建 NetworkX 有向图
        返回: (graph, node_map)
        """
        G = nx.DiGraph()
        node_map = {}  # {node_obj: node_id_in_graph}
        
        # 添加所有节点
        for i, node in enumerate(self.net.nodes):
            G.add_node(node.name)
            node_map[node] = node.name
        
        # 添加所有边及其权重
        for src_node in self.net.nodes:
            for dst_node in self.net.nodes:
                if src_node == dst_node:
                    continue
                
                cost = self._get_link_cost(src_node, dst_node)
                
                if cost < float('inf'):
                    G.add_edge(src_node.name, dst_node.name, weight=cost)
        
        return G, node_map
    
    # ======================== SPF (Shortest Path First) ========================
    
    def spf_routing(self, src_node, dst_node):
        """
        使用 NetworkX 计算最短路径
        
        Args:
            src_node (Node): 源节点
            dst_node (Node): 目标节点
        
        Returns:
            dict: {
                'path': [node_names],
                'length': total_cost,
                'next_hop': next_hop_name or None,
                'time': computation_time
            }
        """
        start_time = time.time()
        
        try:
            G, node_map = self._build_graph()
            
            # 使用 NetworkX 计算最短路径
            path = nx.shortest_path(
                G, 
                source=src_node.name, 
                target=dst_node.name,
                weight='weight'
            )
            
            # 计算路径长度
            if len(path) > 1:
                length = nx.shortest_path_length(
                    G,
                    source=src_node.name,
                    target=dst_node.name,
                    weight='weight'
                )
            else:
                length = 0
            
            # 获取下一跳
            next_hop = path[1] if len(path) > 1 else None
            
            elapsed = time.time() - start_time
            
            return {
                'path': path,
                'length': length,
                'next_hop': next_hop,
                'time': elapsed,
                'hops': len(path) - 1
            }
        
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            elapsed = time.time() - start_time
            return {
                'path': None,
                'length': float('inf'),
                'next_hop': None,
                'time': elapsed,
                'error': str(e)
            }
    
    # ======================== ECMP (Equal-Cost Multi-Path) ========================
    
    def ecmp_routing(self, src_node, dst_node, max_paths=4, load_balance_policy='flow_hash'):
        """
        使用 NetworkX 计算等成本多路径
        
        Args:
            src_node (Node): 源节点
            dst_node (Node): 目标节点
            max_paths (int): 最大路径数
            load_balance_policy (str): 负载均衡策略
                - 'flow_hash': 流哈希 (确定性)
                - 'random': 随机
                - 'weighted': 基于链路质量权重
        
        Returns:
            dict: {
                'paths': [[list of nodes], ...],
                'unique_nexthops': [next_hop_names],
                'selected_nexthop': selected_nexthop_name,
                'path_count': number_of_paths,
                'time': computation_time
            }
        """
        start_time = time.time()
        
        try:
            G, node_map = self._build_graph()
            
            # 计算所有最短路径
            all_paths = list(nx.all_shortest_paths(
                G,
                source=src_node.name,
                target=dst_node.name,
                weight='weight'
            ))
            
            # 限制路径数
            all_paths = all_paths[:max_paths]
            
            # 提取所有可能的下一跳
            unique_nexthops = set()
            for path in all_paths:
                if len(path) > 1:
                    unique_nexthops.add(path[1])
            
            unique_nexthops = list(unique_nexthops)
            
            # 根据负载均衡策略选择下一跳
            if not unique_nexthops:
                elapsed = time.time() - start_time
                return {
                    'paths': [],
                    'unique_nexthops': [],
                    'selected_nexthop': None,
                    'path_count': 0,
                    'time': elapsed
                }
            
            selected_nexthop = self._select_nexthop_by_policy(
                src_node, unique_nexthops, load_balance_policy
            )
            
            elapsed = time.time() - start_time
            
            return {
                'paths': all_paths,
                'unique_nexthops': unique_nexthops,
                'selected_nexthop': selected_nexthop,
                'path_count': len(all_paths),
                'time': elapsed
            }
        
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            elapsed = time.time() - start_time
            return {
                'paths': [],
                'unique_nexthops': [],
                'selected_nexthop': None,
                'path_count': 0,
                'time': elapsed,
                'error': str(e)
            }
    
    def _select_nexthop_by_policy(self, src_node, nexthops, policy):
        """
        根据负载均衡策略从候选下一跳中选择一个
        """
        if not nexthops:
            return None
        
        if policy == 'random':
            return np.random.choice(nexthops)
        
        elif policy == 'flow_hash':
            # 确定性哈希：基于源节点名称
            idx = hash(src_node.name) % len(nexthops)
            return nexthops[idx]
        
        elif policy == 'weighted':
            # 基于链路质量的权重选择
            weights = []
            for nexthop_name in nexthops:
                # 从网络中找到对应的节点
                nexthop_node = None
                for node in self.net.nodes:
                    if node.name == nexthop_name:
                        nexthop_node = node
                        break
                
                if nexthop_node:
                    link_info = src_node.link_quality_cache.get(nexthop_node.mac, {})
                    rssi = link_info.get('rssi', -100)
                    # 权重 = RSSI + 120 (转为正数)
                    weight = max(rssi + 120, 1)
                else:
                    weight = 1
                
                weights.append(weight)
            
            # 概率选择
            total = sum(weights)
            if total > 0:
                probs = [w / total for w in weights]
                idx = np.random.choice(len(nexthops), p=probs)
                return nexthops[idx]
            
            return nexthops[0]
        
        return nexthops[0]
    
    # ======================== 对比评估 ========================
    
    def benchmark_comparison(self, num_tests=10):
        """
        运行 SPF 和 ECMP 对比基准测试
        
        Returns:
            pd.DataFrame: 对比结果
        """
        results = []
        
        for test_num in range(num_tests):
            # 重新测量所有链路
            self.net.test_all_links_concurrent()
            
            # 随机选择源和目标节点
            if len(self.net.nodes) < 2:
                print("[ERROR] 网络节点数不足")
                break
            
            src_idx = np.random.randint(0, len(self.net.nodes))
            dst_idx = np.random.randint(0, len(self.net.nodes))
            
            # 确保源和目标不同
            while dst_idx == src_idx:
                dst_idx = np.random.randint(0, len(self.net.nodes))
            
            src_node = self.net.nodes[src_idx]
            dst_node = self.net.nodes[dst_idx]
            
            # 运行 SPF
            spf_result = self.spf_routing(src_node, dst_node)
            
            # 运行 ECMP
            ecmp_result = self.ecmp_routing(src_node, dst_node, max_paths=4)
            
            # 处理 hops 的 None 值
            spf_hops = spf_result.get('hops') if 'error' not in spf_result else None
            ecmp_paths = ecmp_result.get('path_count') if 'error' not in ecmp_result else None
            
            # 记录结果
            result = {
                'test_num': test_num + 1,
                'src': src_node.name,
                'dst': dst_node.name,
                'spf_next_hop': spf_result.get('next_hop'),
                'spf_hops': spf_result.get('hops'),
                'spf_cost': spf_result.get('length'),
                'spf_time': spf_result.get('time'),
                'ecmp_nexthops_count': len(ecmp_result.get('unique_nexthops', [])),
                'ecmp_selected_hop': ecmp_result.get('selected_nexthop'),
                'ecmp_time': ecmp_result.get('time'),
                'ecmp_paths_count': ecmp_result.get('path_count')
            }
            
            results.append(result)
            print(f"[测试 {test_num+1}] {src_node.name} -> {dst_node.name}")
            print(f"  SPF: next_hop={result['spf_next_hop']}, "
                  f"hops={result['spf_hops']}, "
                  f"cost={result['spf_cost']:.3f}, "
                  f"time={result['spf_time']*1000:.2f}ms")
            print(f"  ECMP: nexthops={result['ecmp_nexthops_count']}, "
                  f"paths={result['ecmp_paths_count']}, "
                  f"time={result['ecmp_time']*1000:.2f}ms")
        
        return pd.DataFrame(results)
    
    # ======================== 结果可视化 ========================
    
    def plot_comparison(self, results_df, save_path="test/networkx_comparison.png"):
        """
        绘制对比结果
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. SPF vs ECMP 计算时间对比
        ax = axes[0, 0]
        ax.bar(['SPF', 'ECMP'], 
               [results_df['spf_time'].mean() * 1000, 
                results_df['ecmp_time'].mean() * 1000],
               color=['blue', 'green'])
        ax.set_ylabel('Computation Time (ms)')
        ax.set_title('Average Computation Time Comparison')
        ax.grid(axis='y')
        
        # 2. 时间分布箱线图
        ax = axes[0, 1]
        ax.boxplot([results_df['spf_time'] * 1000, 
                    results_df['ecmp_time'] * 1000],
                   tick_labels=['SPF', 'ECMP'])
        ax.set_ylabel('Computation Time (ms)')
        ax.set_title('Computation Time Distribution')
        ax.grid(axis='y')
        
        # 3. ECMP 下一跳多样性
        ax = axes[1, 0]
        ecmp_hops_valid = results_df['ecmp_nexthops_count'].dropna()
        if len(ecmp_hops_valid) > 0:
            ax.hist(ecmp_hops_valid, bins=10, color='purple', edgecolor='black')
        ax.set_xlabel('Number of Equal-Cost Next-Hops')
        ax.set_ylabel('Frequency')
        ax.set_title('ECMP Next-Hop Diversity')
        ax.grid(axis='y')
        
        # 4. SPF 跳数统计
        ax = axes[1, 1]
        spf_hops_valid = results_df['spf_hops'].dropna()
        if len(spf_hops_valid) > 0:
            ax.hist(spf_hops_valid, bins=10, color='orange', edgecolor='black')
        ax.set_xlabel('Number of Hops')
        ax.set_ylabel('Frequency')
        ax.set_title('SPF Path Length Distribution')
        ax.grid(axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"[结果保存] {save_path}")
        plt.close()
    
    def print_summary(self, results_df):
        """
        打印对比总结
        """
        print("\n" + "="*60)
        print("NetworkX 路由算法对标测试总结")
        print("="*60)
        
        print(f"\n[测试配置]")
        print(f"  度量类型: {self.metric_type}")
        print(f"  测试轮次: {len(results_df)}")
        print(f"  网络节点数: {len(self.net.nodes)}")
        
        # 提取有效数据（排除 None）
        spf_hops_valid = results_df['spf_hops'].dropna()
        ecmp_paths_valid = results_df['ecmp_paths_count'].dropna()
        
        print(f"\n[SPF (Shortest Path First)]")
        print(f"  平均计算时间: {results_df['spf_time'].mean() * 1000:.3f} ms")
        print(f"  最小计算时间: {results_df['spf_time'].min() * 1000:.3f} ms")
        print(f"  最大计算时间: {results_df['spf_time'].max() * 1000:.3f} ms")
        if len(spf_hops_valid) > 0:
            print(f"  平均路径跳数: {spf_hops_valid.mean():.2f}")
        else:
            print(f"  平均路径跳数: N/A (无有效路径)")
        spf_cost_valid = results_df['spf_cost'].dropna()
        if len(spf_cost_valid) > 0:
            print(f"  平均路径成本: {spf_cost_valid.mean():.3f}")
        else:
            print(f"  平均路径成本: N/A (无有效路径)")
        
        print(f"\n[ECMP (Equal-Cost Multi-Path)]")
        print(f"  平均计算时间: {results_df['ecmp_time'].mean() * 1000:.3f} ms")
        print(f"  最小计算时间: {results_df['ecmp_time'].min() * 1000:.3f} ms")
        print(f"  最大计算时间: {results_df['ecmp_time'].max() * 1000:.3f} ms")
        if len(ecmp_paths_valid) > 0:
            print(f"  平均多路径数: {ecmp_paths_valid.mean():.2f}")
        else:
            print(f"  平均多路径数: N/A (无有效路径)")
        print(f"  平均下一跳选项: {results_df['ecmp_nexthops_count'].mean():.2f}")
        
        print(f"\n[性能对比]")
        spf_avg = results_df['spf_time'].mean()
        ecmp_avg = results_df['ecmp_time'].mean()
        overhead = ((ecmp_avg - spf_avg) / spf_avg) * 100
        print(f"  ECMP vs SPF 计算时间开销: {overhead:+.1f}%")
        
        print(f"\n[成本度量类型]")
        print(f"  当前使用: {self.metric_type}")
        cost_types = {
            'hops': '跳数 (每条边权重=1)',
            'latency': '延迟 (毫秒)',
            'loss': '丢包率 (百分比)',
            'combined': '综合成本 (1.0 + latency/100 + loss/100*2.0)'
        }
        print(f"  说明: {cost_types.get(self.metric_type)}")
        
        print("="*60 + "\n")
    
    def save_results_to_csv(self, results_df, save_path="test/networkx_benchmark.csv"):
        """
        保存对比结果到 CSV 文件
        """
        results_df.to_csv(save_path, index=False)
        print(f"[结果已保存] {save_path}")


# ======================== 主程序 ========================

def main():
    """
   运行 NetworkX 对标测试
    """
    print("\n" + "="*60)
    print("NetworkX 路由算法对标工具")
    print("="*60)
    
    # 1. 创建网络
    print("\n[1/4] 初始化网络...")
    net = Net(name='Comparison_Net', interval=1)
    
    # 添加测试节点
    net.add_node("11", "02:00:00:00:01:00", "10.10.10.1/24", position=(10, 0, 0), direction=(-1, 0, 0))
    net.add_node("12", "02:00:00:00:02:00", "10.10.10.2/24", position=(-10, 0, 0), direction=(1, 0, 0))
    net.add_node("13", "02:00:00:00:03:00", "10.10.10.3/24", position=(0, 10, 0), direction=(0, 0, 0))
    net.add_node("14", "02:00:00:00:04:00", "10.10.10.4/24", position=(0, -10, 0), direction=(-1, 0, 0))
    net.add_node("15", "02:00:00:00:05:00", "10.10.10.5/24", position=(50, 5, 0), direction=(0, 0, 0))
    net.add_node("16", "02:00:00:00:06:00", "10.10.10.6/24", position=(50, -5, 0), direction=(0, 0, 0))
    net.add_node("17", "02:00:00:00:07:00", "10.10.10.7/24", position=(50, 0, 0), direction=(0, 0, 0))
    
    print(f"  已创建 {len(net.nodes)} 个节点")
    
    # 2. 启动网络
    print("\n[2/4] 启动网络（...")
    net.start_network()
    
    # 3. 创建对比工具并运行测试
    print("\n[3/4] 运行对标测试...")
    comparator = NetworkXAlgorithmComparison(net, metric_type="combined")
    
    # 确保测试目录存在
    import os
    os.makedirs("test", exist_ok=True)
    
    # 运行对比基准测试
    results_df = comparator.benchmark_comparison(num_tests=100)
    
    # 4. 生成报告
    print("\n[4/4] 生成报告...")
    comparator.print_summary(results_df)
    comparator.plot_comparison(results_df)
    comparator.save_results_to_csv(results_df)
    
    # 5. 清理
    print("\n[清理] 结束测试...")
    net.end_test()
    
    print("\n对标测试完成！")
    print(f"  结果文件: test/networkx_benchmark.csv")
    print(f"  图表文件: test/networkx_comparison.png")


if __name__ == "__main__":
    main()
