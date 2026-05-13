import os
import random
import time

from net_dual_timescale_shm_topk_continuous import NetDualTimeScaleSHMTopKContinuous


def make_mac(index):
    # Locally administered unicast MAC.
    return f"02:00:00:{(index >> 16) & 0xFF:02x}:{(index >> 8) & 0xFF:02x}:{index & 0xFF:02x}"


def make_ip(index):
    # Spread addresses across /24 blocks to avoid edge cases when node count grows.
    third = (index // 254) + 10
    fourth = (index % 254) + 1
    return f"10.10.{third}.{fourth}/24"


def random_direction(max_speed=1.2):
    # Keep mobility moderate to avoid overly unstable links in one step.
    dx = random.uniform(-max_speed, max_speed)
    dy = random.uniform(-max_speed, max_speed)
    return (round(dx, 2), round(dy, 2), 0.0)


def random_position(xy_bound=200.0):
    x = random.uniform(-xy_bound, xy_bound)
    y = random.uniform(-xy_bound, xy_bound)
    return (round(x, 1), round(y, 1), 0.0)



def main():
    random.seed(20260415)

    num_nodes = 32
    training_steps = 20
    core_rate = 0.30
    fast_window_sec = 1.0

    net = NetDualTimeScaleSHMTopKContinuous(
        name="Net_1",
        interval=1,
        t_slow=5.0,
        t_fast=0.05,
        top_k=3,
        lstm_window=12,
        lstm_ckpt_path=None,
    )

    # Build random topology with 32 nodes.
    for i in range(num_nodes):
        node_name = f"n{i + 1}"
        mac = make_mac(i + 1)
        ip = make_ip(i + 1)
        pos = random_position(xy_bound=220.0)

        # 20% static nodes, 80% mobile nodes.
        if random.random() < 0.2:
            direction = (0.0, 0.0, 0.0)
        else:
            direction = random_direction(max_speed=1.0)

        net.add_node(name=node_name, mac=mac, ip=ip, position=pos, direction=direction)

    os.makedirs("test", exist_ok=True)

    print("=== LSTM + PPO 双时间尺度随机拓扑测试开始 (32 节点) ===")
    print(f"节点数: {num_nodes}, 训练步数: {training_steps}, 核心比例: {core_rate}")

    last_iteration_time = net.interval
    total_begin = time.time()

    try:
        net.start_network()
        net.start_slow_predictor()

        for step in range(training_steps):
            step_begin = time.time()
            print(f"=== 第 {step + 1}/{training_steps} 次迭代 ===")

            # Dynamic topology evolution.
            net.move_nodes(last_iteration_time)

            # Re-elect core nodes periodically.
            if step % net.agent_migration_interval == 0 or step == 0:
                net.test_all_links_concurrent()
                net.select_core_nodes(num_nodes_rate=core_rate)

            # Fast stream scheduling window.
            fast_begin = time.time()
            while time.time() - fast_begin < fast_window_sec:
                net.test_all_links_concurrent()
                net.update_routing()
                time.sleep(net.T_fast)

            elapsed = time.time() - step_begin
            last_iteration_time = round(elapsed, 1)
            print(f"第 {step + 1} 次迭代耗时: {elapsed:.1f} 秒")

        total_elapsed = time.time() - total_begin
        print("=== 测试结束 ===")
        print(f"总耗时: {total_elapsed:.1f} 秒")

        net.plot_reward_history(save_path="test/reward_history_lstm_random32.png")

    except KeyboardInterrupt:
        print("测试被用户中断")
    finally:
        try:
            net.stop_slow_predictor()
        except Exception:
            pass
        net.end_test()


if __name__ == "__main__":
    main()
