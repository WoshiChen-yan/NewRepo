import os
import time

from lstm_gru_ppo.net_dual_timescale_shm import NetDualTimeScaleSHM


Net_1 = NetDualTimeScaleSHM(name='Net_1', interval=1, t_slow=1.0, t_fast=0.01)
Net_1.add_node(name="11", mac="02:00:00:00:01:00", ip="10.10.10.1/24", position=(15, 0, 0), direction=(-0.1, 0, 0))
Net_1.add_node(name="12", mac="02:00:00:00:02:00", ip="10.10.10.2/24", position=(-15, 0, 0), direction=(0.1, 0, 0))
Net_1.add_node(name="13", mac="02:00:00:00:03:00", ip="10.10.10.3/24", position=(0, 10, 0), direction=(0, 0, 0))
Net_1.add_node("14", "02:00:00:00:00:04", "10.10.10.4/24", (0, -100, 0), (-1, 0, 0))
Net_1.add_node("15", "02:00:00:00:00:05", "10.10.10.5/24", (50, 50, 0), (0, 0, 0))

os.makedirs("test", exist_ok=True)
Net_1.start_network()

print("=== 双时间尺度(多进程共享内存)测试开始 ===")

training_steps = 15
last_iteration_time = Net_1.interval

try:
    Net_1.start_slow_predictor()

    total_begin = time.time()
    for i in range(training_steps):
        step_begin = time.time()
        print(f"=== 第{i + 1}/{training_steps}次迭代 ===")

        Net_1.move_nodes(last_iteration_time)

        if i % Net_1.agent_migration_interval == 0 or i == 0:
            Net_1.test_all_links_concurrent()
            Net_1.select_core_nodes(num_nodes_rate=0.3)

        fast_window_begin = time.time()
        while time.time() - fast_window_begin < 1.0:
            Net_1.test_all_links_concurrent()
            Net_1.update_routing()
            time.sleep(Net_1.T_fast)

        elapsed = time.time() - step_begin
        last_iteration_time = round(elapsed, 1)
        print(f"第{i + 1}次测试耗时: {elapsed:.1f}秒")

    print("=== 测试全部结束 ===")
    print(f"总耗时: {time.time() - total_begin:.1f}秒")

    Net_1.plot_all_nodes()
    Net_1.plot_reward_history(save_path="test/reward_history_dual_timescale_shm.png")

except KeyboardInterrupt:
    print("测试被用户中断。")
finally:
    Net_1.stop_slow_predictor()
    Net_1.end_test()
