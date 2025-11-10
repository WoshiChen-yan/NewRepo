
from net import Net
import subprocess
import time
from wmediumdConnector import w_server
from connect import Interface
import matplotlib.pyplot as plt 
import os 

# 1. 初始化网络和节点
Net_1 = Net(name='Net_1', interval=1) 
Net_1.add_node(name="11", mac="02:00:00:00:01:00", ip="10.10.10.1/24",position= (140, 0, 0), direction=(-10, 1, 1))
# Net_1.add_node(name="12", mac="02:00:00:00:02:00", ip="10.10.10.2/24",position= (-140, 0, 0),direction= (10, 1, 1))
# Net_1.add_node(name="13", mac="02:00:00:00:03:00", ip="10.10.10.3/24",position=(-200, 0, 0),direction=(0, 0, 1))
# Net_1.add_node("14", "02:00:00:00:00:04", "10.10.10.4/24",(250, 0, 30),(-10, -10, 1))
# Net_1.add_node("15", "02:00:00:00:00:05","10.10.10.5/24",(50,50,0),(0, 0, 100))
# Net_1.add_node("16", "02:00:00:00:00:06","10.10.10.6/24",(1,20,300),(10,10,10))
# Net_1.add_node("17", "02:00:00:00:00:07","10.10.10.7/24",(10,20,30),(1,1,100))
# docker stop 11&&docker rm 11&&docker stop 12&&docker rm 12&&docker stop 13&&docker rm 13&&docker stop 14&&docker rm 14
# 确保 test 目录存在
os.makedirs("test", exist_ok=True)

# 2. 启动网络
Net_1.start_network()
time.sleep(5)

# 3. (可选) 启动背景流量
print("启动背景流量 (iperf)...")
try:
    node_17 = Net_1.node_dict["17"]
    node_11 = Net_1.node_dict["11"]
    
    print(f"在 {node_17.name} 上安装 iperf...")
    print(node_17.cmd("apt-get update && apt-get install -y iperf"))
    print(f"在 {node_17.name} 上启动 iperf 服务器...")
    node_17.cmd("iperf -s -u &") 
    
    print(f"在 {node_11.name} 上安装 iperf...")
    print(node_11.cmd("apt-get update && apt-get install -y iperf"))
    print(f"在 {node_11.name} 上启动 iperf 客户端...")
    node_11.cmd(f"iperf -c {node_17.ip.split('/')[0]} -u -b 1m -t 300 &") 
    print(f"已启动从 {node_11.name} 到 {node_17.name} 的 1Mbps UDP 流量")
except Exception as e:
    print(f"启动 iperf 失败: {e}。")


print("=== 智能路由训练与测试开始 ===")
times={}
time_all=time.time()
training_steps = 80 

# <--- MODIFIED: 4. 主训练/仿真循环 (v6) ---_>
for i in range(training_steps): 
    
    time2 = time.time()
    print(f"===第{i+1}/{training_steps}次迭代===")

    # 1. 移动节点 (模拟动态拓扑)
    Net_1.move_nodes()
    
    # 2. (v6) 智能体迁移 (核心)
    #    定期重选核心节点
    #    (必须在 测量 和 决策 之前)
    if i % Net_1.agent_migration_interval == 0 or i == 0:
        # 选举前必须先测量一次，否则没有邻居数据
        Net_1.test_all_links_concurrent() 
        Net_1.select_core_nodes_distributed(num_nodes_rate=0.3) # 选择 30% 的核心节点
    
    # 3. (v6) 执行路由步骤
    #    此函数现在内部包含了 决策(t) -> 执行(t) -> 测量(t+1) -> 学习(t+1)
    Net_1.update_routing()
    
    time1= time.time()-time2
    times[i]={'第次测试':i+1,'耗时':{time1}}
    
    print(f"迭代 {i+1} 完成，耗时 {time1:.2f} 秒。等待 {Net_1.interval} 秒...")
    sleep_time = max(0, Net_1.interval - time1)
    if sleep_time > 0:
        time.sleep(sleep_time) 
# <--- MODIFIED 结束 ---_>
        

print("===测试全部结束===")
time1=time.time()-time_all
print(f"总共测试时间：{time1}秒") 
time.sleep(5)  

# 5. 绘制结果图
Net_1.plot_all_nodes()
Net_1.plot_reward_history() # 绘制新的奖励图

print(times)

# 6. 清理
Net_1.end_test()