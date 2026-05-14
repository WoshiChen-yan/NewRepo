import os
import sys
import random
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PARENT_ROOT = PACKAGE_ROOT.parent
for candidate in (str(PACKAGE_ROOT), str(PARENT_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from net_dual_timescale_shm_topk_continuous import NetDualTimeScaleSHMTopKContinuous


def make_mac(index):
    # Locally administered unicast MAC.
    return f"02:00:00:{(index >> 16) & 0xFF:02x}:{(index >> 8) & 0xFF:02x}:{index & 0xFF:02x}"


def make_ip(index):
    # Spread addresses across /24 blocks to avoid edge cases when node count grows.
    third = (index // 254) + 10
    fourth = (index % 254) + 1
    return f"10.10.{third}.{fourth}/24"


def random_direction(max_speed=1):
    # Keep mobility moderate to avoid overly unstable links in one step.
    dx = random.uniform(-max_speed, max_speed)
    dy = random.uniform(-max_speed, max_speed)
    return (round(dx, 2), round(dy, 2), 0.0)


def random_position(xy_bound=30.0):
    x = random.uniform(-xy_bound, xy_bound)
    y = random.uniform(-xy_bound, xy_bound)
    return (round(x, 1), round(y, 1), 0.0)


def moving_average(values, window=5):
    if not values:
        return []
    if len(values) < window:
        return [sum(values) / len(values)]
    out = []
    for i in range(window - 1, len(values)):
        window_values = values[i - window + 1 : i + 1]
        out.append(sum(window_values) / window)
    return out


def print_banner():
    print("=" * 92)
    print("LSTM + GRU + PPO 双时间尺度连续分流实验")
    print("目标: 用 reward 提升、趋势稳定性、以及最终曲线来证明算法有效性")
    print("=" * 92)


def print_config(num_nodes, training_steps, core_rate, fast_window_sec, lstm_window, top_k):
    print(f"节点数        : {num_nodes}")
    print(f"训练步数      : {training_steps}")
    print(f"核心比例      : {core_rate:.2f}")
    print(f"快周期窗口    : {fast_window_sec:.1f}s")
    print(f"LSTM 窗口     : {lstm_window}")
    print(f"top_k         : {top_k}")
    print("-" * 92)


def summarize_iteration(net, step, elapsed, warmup_seconds):
    avg_reward = net.avg_core_reward_history[-1] if net.avg_core_reward_history else 0.0
    trend_matrix, trend_ts, conf, _ = net._read_trend()
    alpha, age = net._compute_trend_alpha(time.time(), trend_ts, conf)

    print(
        f"[Step {step:02d}]  "
        f"time={elapsed:6.2f}s  "
        f"avg_reward={avg_reward:8.3f}  "
        f"trend_alpha={alpha:4.2f}  "
        f"trend_age={age:4.1f}s  "
        f"warmup={warmup_seconds:.1f}s"
    )

    if trend_matrix is not None:
        mean_risk = float(trend_matrix.mean())
        max_risk = float(trend_matrix.max())
        print(
            f"           trend_stats -> mean_risk={mean_risk:.3f}, max_risk={max_risk:.3f}, confidence={conf:.2f}"
        )


def print_final_report(net, total_elapsed, training_steps):
    print("=" * 92)
    print("实验总结")
    print(f"总耗时        : {total_elapsed:.2f}s")
    print(f"迭代次数      : {training_steps}")
    print(f"奖励点数      : {len(net.avg_core_reward_history)}")

    if net.avg_core_reward_history:
        rewards = net.avg_core_reward_history
        mean_reward = sum(rewards) / len(rewards)
        first_reward = rewards[0]
        last_reward = rewards[-1]
        best_reward = max(rewards)
        head_avg = sum(rewards[: max(1, len(rewards) // 3)]) / max(1, len(rewards) // 3)
        tail_avg = sum(rewards[-max(1, len(rewards) // 3):]) / max(1, len(rewards) // 3)
        gain = last_reward - first_reward

        print(f"平均核心奖励  : {mean_reward:.3f}")
        print(f"初始核心奖励  : {first_reward:.3f}")
        print(f"最后一次奖励  : {last_reward:.3f}")
        print(f"最佳核心奖励  : {best_reward:.3f}")
        print(f"前段平均奖励  : {head_avg:.3f}")
        print(f"后段平均奖励  : {tail_avg:.3f}")
        print(f"奖励提升幅度  : {gain:.3f}")

    print(f"核心节点数    : {len(net.core_nodes)}")
    print(f"边缘节点数    : {len(net.edge_nodes)}")
    print("策略说明      : LSTM 趋势感知 + GRU/PPO 决策 + top-k 连续流量分配")
    print("提示          : 若后段平均奖励高于前段平均奖励，说明算法在学习和适应拓扑变化")
    print("=" * 92)


def main():
    random.seed(20260514)

    output_dir = PACKAGE_ROOT / "test"
    output_dir.mkdir(parents=True, exist_ok=True)

    num_nodes = 5
    training_steps = 10
    core_rate = 0.30
    fast_window_sec = 1.0

    net = NetDualTimeScaleSHMTopKContinuous(
        name="Net_1",
        interval=1,
        t_slow=5.0,
        t_fast=0.5,
        top_k=3,
        lstm_window=12,
        lstm_ckpt_path=None,
    )

    # Build random topology with 32 nodes.
    for i in range(num_nodes):
        node_name = f"n{i + 1}"
        mac = make_mac(i + 1)
        ip = make_ip(i + 1)
        pos = random_position()

        # 20% static nodes, 80% mobile nodes.
        if random.random() < 0.2:
            direction = (0.0, 0.0, 0.0)
        else:
            direction = random_direction(max_speed=1.0)

        net.add_node(name=node_name, mac=mac, ip=ip, position=pos, direction=direction)

    print_banner()
    print_config(
        num_nodes=num_nodes,
        training_steps=training_steps,
        core_rate=core_rate,
        fast_window_sec=fast_window_sec,
        lstm_window=12,
        top_k=3,
    )

    last_iteration_time = net.interval
    total_begin = time.time()

    try:
        net.start_network()
        net.start_slow_predictor()

        for step in range(training_steps):
            step_begin = time.time()
            print(
                f"--- Iteration {step + 1:02d}/{training_steps:02d}  "
                f"topology -> slow LSTM -> GRU/PPO -> routing -> learning ---"
            )

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
            summarize_iteration(net, step + 1, elapsed, fast_window_sec)

        total_elapsed = time.time() - total_begin
        print_final_report(net, total_elapsed, training_steps)

        if net.avg_core_reward_history:
            smoothed = moving_average(net.avg_core_reward_history, window=3)
            print("平滑奖励序列   : " + ", ".join(f"{v:.3f}" for v in smoothed[-5:]))

        net.plot_all_nodes()
        net.plot_reward_history(save_path=str(output_dir / "reward_history_lstm_random32.png"))

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
