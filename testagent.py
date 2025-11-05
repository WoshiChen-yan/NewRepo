from net import Net
import subprocess
import time
from wmediumdConnector import w_server
from connect import Interface
import matplotlib.pyplot as plt # <--- MODIFIED: 导入 matplotlib

# 1. 初始化网络和节点
Net_1 = Net(name='Net_1', interval=1) # 1秒一个时间步
Net_1.add_node(name="11", mac="02:00:00:00:01:00", ip="10.10.10.1/24",position= (140, 0, 0), direction=(-10, 1, 1))
Net_1.add_node(name="12", mac="02:00:00:00:02:00", ip="10.10.10.2/24",position= (-140, 0, 0),direction= (10, 1, 1))
Net_1.add_node(name="13", mac="02:00:00:00:03:00", ip="10.10.10.3/24",position=(-200, 0, 0),direction=(0, 0, 1))
Net_1.add_node("14", "02:00:00:00:00:04", "10.10.10.4/24",(250, 0, 30),(-10, -10, 1))
Net_1.add_node("15", "02:00:00:00:00:05","10.10.10.5/24",(50,50,0),(0, 0, 100))
Net_1.add_node("16", "02:00:00:00:00:06","10.10.10.6/24",(1,20,300),(10,10,10))
Net_1.add_node("17", "02:00:00:00:00:07","10.10.10.7/24",(10,20,30),(1,1,100))


# 2. 启动网络
Net_1.start_network()
time.sleep(5)

# <--- MODIFIED: 3. 初始化 PPO 智能体 ---_>
# 这将创建共享 Critic 和所有 Agent
Net_1.select_core_nodes() 
print("核心节点和 PPO 智能体已初始化。")

# <--- MODIFIED: 4. (可选) 启动背景流量 (项目3) ---_>
# 这对于产生有意义的“奖励”非常重要
print("启动背景流量 (iperf)...")
try:
    node_17 = Net_1.node_dict["17"]
    node_11 = Net_1.node_dict["11"]
    
    # 在 17 上启动 iperf UDP 服务器
    node_17.cmd("apt-get update && apt-get install -y iperf")
    node_17.cmd("iperf -s -u &") 
    
    # 在 11 上启动 iperf UDP 客户端
    node_11.cmd("apt-get update && apt-get install -y iperf")
    node_11.cmd(f"iperf -c {node_17.ip.split('/')[0]} -u -b 1m -t 300 &") 
    print(f"已启动从 {node_11.name} 到 {node_17.name} 的 1Mbps UDP 流量")
except Exception as e:
    print(f"启动 iperf 失败: {e}")
# <--- MODIFIED 结束 ---_>


print("=== 智能路由训练与测试开始 ===")
times={}
time_all=time.time()
training_steps = 80 # <--- MODIFIED: 定义训练步数

# <--- MODIFIED: 5. 更改为主训练/仿真循环 ---_>
for i in range(training_steps): 
    
    time2 = time.time()
    print(f"===第{i+1}/{training_steps}次迭代===")

    # 1. 移动节点 (模拟动态拓扑)
    Net_1.move_nodes()
    
    # 2. 测量当前网络状态 (更新链路质量历史)
    Net_1.test_all_links_concurrent()
    
    # 3. 执行 PPO 训练步骤 (核心)
    #    (获取状态 -> 选动作 -> 执行路由 -> 存经验 -> 学习)
    Net_1.update_routing()
    
    time1= time.time()-time2
    times[i]={'第次测试':i+1,'耗时':{time1}}
    
    print(f"迭代 {i+1} 完成，耗时 {time1:.2f} 秒。等待 {Net_1.interval} 秒...")
    time.sleep(Net_1.interval) # 等待一个时间步
# <--- MODIFIED 结束 ---_>
        

print("===测试全部结束===")
time1=time.time()-time_all
print(f"总共测试时间：{time1}秒") 
time.sleep(5)  

# <--- MODIFIED: 6. 绘制结果图 ---_>
# 绘制链路质量图
Net_1.plot_all_nodes()
# 绘制奖励曲线图 (!!!)
Net_1.plot_reward_history()
# <--- MODIFIED 结束 ---_>

print(times)

# 7. 清理
Net_1.end_test()